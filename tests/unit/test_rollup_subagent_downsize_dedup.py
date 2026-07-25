"""The rollup must count a subagent's tokens exactly once.

``downsize`` aggregates per session and ``subagent`` aggregates per
(session, sub_agent_id) over the SAME spans. Their signatures are structurally
different (``cost:downsize:<agent>`` vs ``cost:subagent[:<name>]``), so
``estimated_recoverable_rollup``'s dedup-by-signature can't catch an overlap —
the populations have to be disjoint at the source. These tests pin that: the
same class of guard ``_per_agent_cache_recoverable_by_model`` provides for the
cache family, applied to downsize/subagent.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import analyze_model_downgrade, build_report
from tokenjam.core.optimize.cost_proposals import (
    cost_proposals_from_report,
    estimated_recoverable_rollup,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span

# One session: a small main thread plus a single premium-model Task dispatch.
# The whole-session aggregate (4_500 input / 350 output / 0 tools) sits under
# every downsize threshold, which is exactly what used to make BOTH analyzers
# claim the subagent's tokens.
MAIN_INPUT, MAIN_OUTPUT, MAIN_COST = 500, 50, 0.02
SUB_INPUT, SUB_OUTPUT, SUB_COST = 4_000, 300, 0.30
SESSION_TOKENS = MAIN_INPUT + MAIN_OUTPUT + SUB_INPUT + SUB_OUTPUT


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _insert_session_with_one_task_dispatch(db, session_id: str = "s1") -> None:
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=MAIN_INPUT, output_tokens=MAIN_OUTPUT, cost_usd=MAIN_COST,
        session_id=session_id, sub_agent_id=None, start_time=start,
    ))
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=SUB_INPUT, output_tokens=SUB_OUTPUT, cost_usd=SUB_COST,
        session_id=session_id, sub_agent_id="researcher", start_time=start,
    ))


def test_downsize_excludes_subagent_tokens_from_its_candidate_figure(db):
    # The candidate figure is main-thread only: the Task dispatch's tokens
    # belong to `subagent`, which prices the identical swap over them.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.candidate_sessions == 1
    assert finding.estimated_recoverable_tokens == MAIN_INPUT + MAIN_OUTPUT
    assert finding.actual_cost_usd == pytest.approx(MAIN_COST, abs=1e-6)


def test_denominators_stay_window_wide(db):
    # Only the CANDIDATE side narrows to the main thread. The shares the card
    # reports ("% of sessions", "% of tokens") must still be against the whole
    # window, or excluding subagent spans would silently inflate them.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.total_sessions == 1
    assert finding.window_total_tokens == SESSION_TOKENS


def test_rollup_counts_the_subagent_tokens_exactly_once(db):
    # End to end: build the report, derive every cost proposal, roll them up.
    # `downsize` and `subagent` both fire on this session; their token claims
    # must partition the session, not overlap it.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize", "subagent"],
    )
    proposals = cost_proposals_from_report(report, None, window_days=30.0)
    analyzers = {p.analyzer for p in proposals}
    assert "downsize" in analyzers, "expected the session to trip downsize"
    assert "subagent" in analyzers, "expected the dispatch to trip subagent"

    rollup = estimated_recoverable_rollup(proposals)
    # Pre-fix this summed 4_850 (downsize, whole session) + 4_300 (subagent) =
    # 9_150 — nearly 2x the tokens the session actually spent.
    assert rollup["estimated_recoverable_tokens"] == SESSION_TOKENS
    # The dollar side can only ever claim the session's real spend back.
    assert rollup["estimated_recoverable_usd"] <= MAIN_COST + SUB_COST
