"""relearn's horizon, its ungated past-tense cost, and its rollup presence.

Four defects, one module:

1. relearn read Claude Code's on-disk transcripts, which Claude Code rotates
   (``cleanupPeriodDays``), so the one analyzer whose premise is long-horizon
   recurrence could never accumulate history. The archive lane recovers the
   sessions tokenjam retained telemetry for after the transcript was rotated.
2. A cluster with no fix template in our library reported ``$0`` — an
   action-availability gate zeroing an observed cost.
3. A cluster whose rule is uneconomic to keep reported ``$0`` for the same
   wrong reason: "is codifying this worth it?" is not "did this cost anything?"
4. A future fix's standing cost was netted out of a PAST figure.

Plus the rollup: relearn produced ``RelearnCluster``s, never ``CostProposal``s,
so no aggregate surface could see it at all.

All spans/sessions go through ``tests/factories`` (Critical Rule 8); the backend
is in-memory and no real ``~/.tj`` or ``~/.claude`` is touched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.relearn import (
    GROUNDED_TOKENS_PER_OCCURRENCE,
    MIN_RECURRING_SESSIONS,
    FailureEpisode,
    RelearnCluster,
    RelearnFinding,
    _corpus_window_days,
    analyze_relearns,
    build_proposals,
    compute_relearn_finding,
)
from tokenjam.core.optimize.projection import build_projection_basis
from tests.factories import make_llm_span, make_session, make_tool_span

BASE = datetime(2026, 5, 10, tzinfo=timezone.utc)
CODING_AGENT = "claude-code-demo"

#: Long enough that the transcript lane could never have reached it: Claude Code
#: rotates at 30 days by default, so an episode this old exists ONLY in the
#: archive.
BEYOND_TRANSCRIPT_RETENTION_DAYS = 75


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _episode(
    session_id: str, error: str, *, ts: str | None = None, tool: str = "Bash",
) -> FailureEpisode:
    return FailureEpisode(
        session_id=session_id, repo="demo", ts=ts, tool_name=tool, label="",
        error_text=error, kind="act", is_retry=False, depth=0,
    )


def _priced_session(db, session_id: str, *, agent_id: str = CODING_AGENT, at=BASE):
    """A session row plus one priced LLM span, so a rate profile can be blended."""
    db.upsert_session(make_session(
        agent_id=agent_id, session_id=session_id, started_at=at, ended_at=at,
    ))
    db.insert_span(make_llm_span(
        agent_id=agent_id, session_id=session_id, model="claude-sonnet-4-5",
        input_tokens=2_000, output_tokens=200, start_time=at,
    ))


# --------------------------------------------------------------------------- #
# 1. The horizon is tokenjam's archive, not Claude Code's rotation
# --------------------------------------------------------------------------- #

def test_episode_older_than_transcript_retention_still_counts(db, tmp_path):
    """An episode whose transcript Claude Code already rotated away must still
    reach a cluster. This is the whole premise of the analyzer and it used to be
    structurally impossible: with no ``.jsonl`` on disk the session was invisible
    however long tokenjam had been retaining its telemetry."""
    old = BASE - timedelta(days=BEYOND_TRANSCRIPT_RETENTION_DAYS)
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"archived-{i}"
        _priced_session(db, session_id, at=old)
        span = make_tool_span(
            agent_id=CODING_AGENT, tool_name="Bash", status="error",
            session_id=session_id, start_time=old + timedelta(minutes=i),
        )
        span.status_message = "(eval):cd:1: no such file or directory: orchestrator"
        db.insert_span(span)

    # tmp_path is an EMPTY projects root: not one of these sessions has a
    # transcript, exactly as it would be after Claude Code's rotation ran.
    finding = compute_relearn_finding(
        db.conn, projects_root=tmp_path, distill_enabled=False,
    )

    assert finding.transcript_sessions_scanned == 0
    assert finding.archived_sessions_scanned == MIN_RECURRING_SESSIONS
    assert finding.clusters, "an archived-only recurrence produced no cluster"
    assert finding.clusters[0].occurrences == MIN_RECURRING_SESSIONS
    assert "rotated" in finding.corpus_basis


def test_archive_lane_never_double_counts_a_session_with_a_transcript(db, tmp_path):
    """A session that still HAS a transcript belongs to the transcript lane
    alone. The two lanes are disjoint by session id, so a failure is extracted
    once, never twice."""
    from tokenjam.core.optimize.relearn_otel import extract_archived_coding_failures

    _priced_session(db, "still-on-disk")
    span = make_tool_span(
        agent_id=CODING_AGENT, tool_name="Bash", status="error",
        session_id="still-on-disk", start_time=BASE,
    )
    span.status_message = "no such file or directory: orchestrator"
    db.insert_span(span)

    # The caller hands the archive lane only the transcript-LESS ids.
    assert extract_archived_coding_failures(db.conn, set()) == []
    assert extract_archived_coding_failures(db.conn, {"still-on-disk"})


def test_sentinel_timestamp_does_not_stretch_the_observed_window():
    """A ``1970-01-01`` sentinel must not become the corpus MIN. Widening the
    horizon to the DB brings sentinel-stamped rows into range, and one of them
    reaching the span derivation would report a ~56-year window and crush every
    monthly figure to nearly zero."""
    real = [
        _episode("s1", "boom", ts="2026-06-01T00:00:00Z"),
        _episode("s2", "boom", ts="2026-06-11T00:00:00Z"),
    ]
    assert _corpus_window_days(real) == pytest.approx(10.0)

    with_sentinel = [*real, _episode("s3", "boom", ts="1970-01-01T00:00:00Z")]
    assert _corpus_window_days(with_sentinel) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 2 + 3. No gate may zero an observed cost
# --------------------------------------------------------------------------- #

def _no_fix_template_clusters(db):
    """Three sessions sharing an error no known family matches, so the fix falls
    back to the generic "Review examples" placeholder."""
    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"s{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id, "Unclassifiable widget explosion in the frobnicator",
            ts=(BASE + timedelta(days=i)).isoformat(),
        ))
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    return list(cluster_failures(failures).values())


def test_no_fix_template_cluster_still_carries_its_observed_cost(db):
    """Absence of a fix template is a gap in OUR library, not a property of the
    waste. The cluster claims nothing (correct — there is no fix to claim) and
    still reports what it cost (previously: $0 on every basis)."""
    proposals, _ = build_proposals(
        _no_fix_template_clusters(db), conn=db.conn, window_days=30.0,
        persona="claude-code",
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert "no known fix template" in p.proposed_fix.lower() or "review examples" in p.proposed_fix.lower()

    # The CLAIM is correctly zero: nothing is recoverable without a fix.
    assert p.estimated_recoverable_tokens == 0
    assert p.estimated_monthly_tokens == 0

    # The OBSERVATION is not.
    assert p.past_overspend_tokens >= MIN_RECURRING_SESSIONS * GROUNDED_TOKENS_PER_OCCURRENCE
    assert p.past_overspend_usd is not None and p.past_overspend_usd > 0
    assert p.past_overspend_basis
    # The past figure can never be smaller than any claim made off the same
    # cluster, or a card would read "cost you $X, of which $Y>X was avoidable".
    assert p.past_overspend_tokens >= p.estimated_recoverable_tokens


def test_net_negative_write_budget_does_not_zero_the_past_cost(db):
    """A rung-1 rule that costs more to keep than it saves is correctly not
    offered, and correctly claims nothing. It must NOT retroactively make the
    failures it describes free."""
    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"n{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    clusters = list(cluster_failures(failures).values())
    # A huge projected session count makes any standing rule wildly underwater:
    # the rule is re-sent on every one of them, the saving is not.
    underwater = build_projection_basis(30.0, 30, 500_000)
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        projection=underwater, repo_cwd_map={"demo": "/tmp/demo"},
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert p.net_negative, "expected the write budget to suppress this rule"
    assert p.estimated_recoverable_tokens == 0        # nothing worth doing
    # It still cost real money, at the FULL observed rate.
    assert p.past_overspend_tokens >= MIN_RECURRING_SESSIONS * GROUNDED_TOKENS_PER_OCCURRENCE
    assert p.past_overspend_usd is not None and p.past_overspend_usd > 0


def test_past_overspend_is_never_netted_against_standing_cost(db):
    """The past figure is ``occurrences x per-occurrence cost``, full stop. It
    does not move when a fix's FUTURE maintenance cost changes, because a
    forward cost cannot be subtracted from money already spent."""
    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"p{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    clusters = list(cluster_failures(failures).values())
    observed = set()
    for basis in (None, build_projection_basis(30.0, 30, 10),
                  build_projection_basis(30.0, 30, 500_000)):
        proposals, _ = build_proposals(
            clusters, conn=db.conn, window_days=30.0, persona="claude-code",
            projection=basis, repo_cwd_map={"demo": "/tmp/demo"},
        )
        p = proposals[0]
        assert p.past_overspend_tokens >= MIN_RECURRING_SESSIONS * GROUNDED_TOKENS_PER_OCCURRENCE
        observed.add((p.past_overspend_tokens, p.past_overspend_usd))

    # One value across every standing-cost basis: the past figure does not move
    # when the FUTURE maintenance cost of a fix does.
    assert len(observed) == 1, observed


def test_recurrence_gate_residue_is_counted_not_silently_dropped(db):
    """A one-off failure is not a relearn, so it gets no cluster and no claim —
    but it is not free either, and the finding says so instead of dropping it."""
    _priced_session(db, "solo")
    finding = analyze_relearns(
        [], conn=db.conn, distill_enabled=False,
        extra_failures=[_episode("solo", "a one-off boom", ts=BASE.isoformat())],
    )

    assert finding.clusters == []
    assert finding.below_threshold_clusters == 1
    assert finding.below_threshold_occurrences == 1
    assert finding.below_threshold_past_overspend_tokens == GROUNDED_TOKENS_PER_OCCURRENCE


# --------------------------------------------------------------------------- #
# 4. relearn reaches the rollup
# --------------------------------------------------------------------------- #

class _Window:
    days = 30.0
    active_days = 30
    sessions = 100


class _Report:
    persona = "claude-code"
    downgrade = None
    window = _Window()

    def __init__(self, finding):
        self.findings = {"relearn": finding}


def test_relearn_reaches_the_cost_proposals_and_the_past_overspend_rollup(db):
    from tokenjam.core.optimize.cost_proposals import (
        COST_ANALYZERS,
        cost_proposals_from_report,
        past_overspend_rollup,
    )

    assert "relearn" in COST_ANALYZERS

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"r{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    finding = analyze_relearns(
        [], conn=db.conn, distill_enabled=False, extra_failures=failures,
        persona="claude-code",
    )
    assert finding.clusters and finding.past_overspend_usd

    proposals = cost_proposals_from_report(_Report(finding))
    relearn_cards = [p for p in proposals if p.analyzer == "relearn"]

    assert len(relearn_cards) == 1, "one aggregate card, never one per cluster"
    card = relearn_cards[0]
    # relearn carries no avoidable/forward claim (see below), so its observed
    # past cost lands on `observed_cost_usd`, never on the avoidable-only
    # `past_overspend_usd` headline field (`_with_past_overspend` maps that
    # from `estimated_recoverable_usd`, which this card deliberately omits).
    assert card.observed_cost_usd == pytest.approx(finding.past_overspend_usd)
    assert card.observed_cost_basis
    # coverage_note is REQUIRED whenever observed_cost_usd is set (this
    # module's own stated contract) — this card carries a cost with no
    # avoidable figure beside it, exactly the shape that must never ship
    # unexplained.
    assert card.coverage_note
    assert "COVERAGE" in card.coverage_note
    assert card.past_overspend_usd is None
    assert not card.apply_capable and card.advise_only
    # No forward claim on this card: relearn's re-read tail is the same re-sent
    # context `resend` already claims, and two analyzers claiming one span in
    # `estimated_recoverable_rollup` is CLAUDE.md rule 27.
    assert card.estimated_recoverable_usd is None
    assert card.estimated_recoverable_tokens is None
    assert card.baseline["relearn_claim_usd"] == pytest.approx(
        finding.estimated_recoverable_usd)

    rollup = past_overspend_rollup(proposals)
    # Cost, not avoidable: relearn's figure sums into the separate
    # `observed_cost_usd` rollup total, never into the waste-labelled
    # `past_overspend_usd` headline.
    assert rollup["observed_cost_usd"] == pytest.approx(finding.past_overspend_usd)
    assert "relearn" in {a["analyzer"] for a in rollup["by_analyzer"]}


def test_relearn_card_survives_a_finding_whose_clusters_all_lack_a_fix(db):
    """The rollup case that motivated the whole ticket: every cluster gated out
    of a claim, so the OLD code contributed exactly nothing to any total, while
    the observed cost was real."""
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        _priced_session(db, f"g{i}")
        failures.append(_episode(
            f"g{i}", "Unclassifiable widget explosion",
            ts=(BASE + timedelta(days=i)).isoformat(),
        ))
    finding = analyze_relearns(
        [], conn=db.conn, distill_enabled=False, persona="claude-code",
        extra_failures=failures,
    )
    assert finding.estimated_recoverable_tokens == 0     # nothing claimable
    assert finding.past_overspend_usd and finding.past_overspend_usd > 0

    cards = [p for p in cost_proposals_from_report(_Report(finding)) if p.analyzer == "relearn"]
    assert len(cards) == 1
    # The observed cost is real even though nothing was claimable.
    assert cards[0].observed_cost_usd > 0
    # The second number (what a fix would return) is correctly zero/absent —
    # they are different quantities and only one of them is gated.
    assert not cards[0].past_overspend_usd
    # And the card says why there is no avoidable figure at all, rather than
    # leaving that gap to imply the cost was unavoidable.
    assert cards[0].coverage_note


def test_relearn_coverage_note_breaks_down_gated_clusters_by_reason(db):
    """The durable half of the fix: `_relearn_to_proposals` must name WHY the
    gated clusters carry no fix, not just that the card has a `coverage_note`
    at all. Mirrors the founder's own measurement (55 clusters, 50 gated: 29
    with no fix template, 17 net-negative, 4 budget-deferred) with a smaller
    fixture of the same three reasons.
    """
    from tokenjam.core.optimize.cost_proposals import _relearn_to_proposals
    from tokenjam.core.optimize.write_budget import (
        REASON_BUDGET_FULL,
        REASON_NET_NEGATIVE,
        REASON_PLACEHOLDER,
    )

    def _cluster(sig, reason, offered=False):
        return RelearnCluster(
            signature=sig, family_key=None, title=sig, sessions=1,
            occurrences=1, repos=["demo"], rung=1, scope="project",
            proposed_fix="fix" if offered else "",
            write_offered=offered, write_blocked_reason=reason,
        )

    clusters = (
        [_cluster(f"nofix{i}", REASON_PLACEHOLDER) for i in range(2)]
        + [_cluster(f"neg{i}", REASON_NET_NEGATIVE) for i in range(3)]
        + [_cluster("budget0", REASON_BUDGET_FULL)]
        + [_cluster("offered0", "", offered=True)]
    )
    finding = RelearnFinding(
        clusters=clusters, past_overspend_usd=46.30, past_overspend_tokens=1_000,
        past_overspend_basis="observed",
    )
    proposals = _relearn_to_proposals(finding)
    assert len(proposals) == 1
    note = proposals[0].coverage_note
    assert note
    assert "2 have no derived fix template" in note
    assert "3 are net-negative" in note
    assert "1 are budget-deferred" in note
    # The load-bearing closing sentence: absence of a figure is not evidence
    # of necessity.
    assert "not a measurement of what was unavoidable" in note
