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


def _priced_session(
    db, session_id: str, *, agent_id: str = CODING_AGENT, at=BASE,
    input_tokens: int = 2_000,
):
    """A session row plus one priced LLM span, so a rate profile can be blended.

    ``input_tokens`` is a knob because a test asserting on the WRITE OFFER (as
    opposed to the clustering) has to fund it: `write_budget.MIN_NET_WRITE_USD`
    declines to spend a permanent block on a rule returning under $5, so a
    cent-scale fixture is correctly refused a write and cannot exercise the
    offered path.
    """
    db.upsert_session(make_session(
        agent_id=agent_id, session_id=session_id, started_at=at, ended_at=at,
    ))
    db.insert_span(make_llm_span(
        agent_id=agent_id, session_id=session_id, model="claude-sonnet-4-5",
        input_tokens=input_tokens, output_tokens=200, start_time=at,
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

    # No forward claim exists at all any more — the one canonical dollar
    # field is `past_overspend_*`, no carve-out.
    assert not hasattr(p, "estimated_recoverable_tokens")
    assert not hasattr(p, "estimated_monthly_tokens")
    assert not p.write_offered

    # The OBSERVATION is not zero.
    assert p.past_overspend_tokens >= MIN_RECURRING_SESSIONS * GROUNDED_TOKENS_PER_OCCURRENCE
    assert p.past_overspend_usd is not None and p.past_overspend_usd > 0
    assert p.past_overspend_basis


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
    assert not p.write_offered        # nothing worth doing, no forward field exists
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
    # Priced on the SAME head basis the kept clusters use — the measured cost
    # of the recovery turn, not the error text's size. Asserted as the
    # invariant rather than a literal, because the head is corpus-derived now:
    # pinning it to the constant is what let the residue silently keep the old
    # ~15-20x-low basis when the head term moved.
    from tokenjam.core.optimize.analyzers.relearn import _measured_turn_tokens
    from tokenjam.core.optimize.rate_profile import blended_rate_profile

    expected = _measured_turn_tokens(
        db.conn, {"solo"}, blended_rate_profile(db.conn, session_ids={"solo"}),
    )
    assert expected is not None and expected >= GROUNDED_TOKENS_PER_OCCURRENCE
    assert finding.below_threshold_past_overspend_tokens == expected


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
    # relearn carries no avoidable claim (see below), so its observed past
    # cost lands on `observed_cost_usd` and the avoidable-only
    # `past_overspend_usd` headline field stays None — "not priced", never a
    # 0.0 that would read as "worth nothing".
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
    # `past_overspend_rollup` is CLAUDE.md rule 27.
    assert card.past_overspend_usd is None
    assert card.past_overspend_tokens is None
    # relearn no longer carries a forward claim at all — a relearn cluster
    # shows its PAST figure only, so there is no `relearn_claim_*` key on the
    # card's `baseline` any more (retired alongside `estimated_recoverable_*`
    # / `estimated_monthly_*`).
    assert "relearn_claim_usd" not in card.baseline
    assert "relearn_claim_tokens" not in card.baseline

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
    assert not hasattr(finding, "estimated_recoverable_tokens")
    assert all(not c.write_offered for c in finding.clusters)  # nothing claimable
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
    # "modelled as", not a bare "are": the net-negative verdict is arithmetic
    # over quantities we can count, and it does not count what the rule saves
    # by preventing the failure (write_budget's module docstring).
    assert "3 are modelled as net-negative" in note
    assert "1 are budget-deferred" in note
    # The load-bearing closing sentence: absence of a figure is not evidence
    # of necessity.
    assert "not a measurement of what was unavoidable" in note


# --------------------------------------------------------------------------- #
# The verdict has to be VISIBLE, not merely computed (#346).
#
# Every quantity below already existed on the dataclass; none of it reached a
# renderer. A card whose write was suppressed showed a bare, unexplained
# "$0.00 est./mo" — on a real corpus that was ~44 of 55 rows — while those same
# clusters had already spent real money. The gate is a gap in OUR fix library,
# never a finding that the failure was harmless, and the surfaces now have to
# say so.
# --------------------------------------------------------------------------- #


def test_suppressed_cluster_carries_a_short_gate_label(db):
    """A suppressed write stamps BOTH the long sentence and the short label.

    The short form is what a dense list can afford to render; without it the
    CLI and the inbox row would each invent their own abbreviation of the
    budget's sentence, which is the drift `write_budget` owns the wording to
    prevent.
    """
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"s{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    clusters = list(cluster_failures(failures).values())
    underwater = build_projection_basis(30.0, 30, 500_000)
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        projection=underwater, repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    assert not p.write_offered
    assert p.write_blocked_reason, "the long sentence must survive for the card"
    assert p.write_blocked_short, "the short label must exist for the list"
    # Short really is short: a list renders one line per row, not a paragraph.
    assert len(p.write_blocked_short) < len(p.write_blocked_reason)
    assert "\n" not in p.write_blocked_short


def test_short_reason_covers_every_suppression_reason():
    """Every reason `write_budget` can emit has a short form.

    A reason added upstream without one degrades to a generic label rather
    than rendering blank — but the five it ships with are named, so a silent
    "no permanent fix offered" on a known reason is a regression.
    """
    from tokenjam.core.optimize import write_budget as wb

    known = [
        wb.REASON_PLACEHOLDER, wb.REASON_NET_NEGATIVE, wb.REASON_BUDGET_FULL,
        wb.REASON_FAMILY_MERGED, wb.REASON_CEILING_REACHED,
    ]
    for reason in known:
        short = wb.short_reason(reason)
        assert short and short != "no permanent fix offered", reason
    # Falsy in, falsy out: an offered write has nothing to say.
    assert wb.short_reason("") == ""
    # Unknown reason still reports THAT something gated the write.
    assert wb.short_reason("something new") == "no permanent fix offered"


def test_cli_relearn_row_leads_with_observed_cost_not_the_gated_claim(db):
    """The CLI row states what the recurrence already cost, and never states a
    suppressed write's gross saving as though it were claimable."""
    from tokenjam.cli.cmd_optimize import (
        _relearn_observed_cost,
        _relearn_write_gate_line,
    )
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"c{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    clusters = list(cluster_failures(failures).values())
    underwater = build_projection_basis(30.0, 30, 500_000)
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        projection=underwater, repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    assert p.net_negative
    # No write is offered — no forward field exists to be zero...
    assert not p.write_offered
    # ...and yet the row still shows a non-empty observed cost.
    cost = _relearn_observed_cost(p, pricing_mode="api")
    assert cost, "a suppressed cluster must still render what it cost"
    assert cost not in ("$0.00", "~0 tok")
    # And it says WHY no fix is offered, with the arithmetic.
    gate = _relearn_write_gate_line(p)
    assert gate
    assert "no permanent fix offered" in gate
    assert "payback" in gate
    # A subscription user sees tokens, never a price they were not charged.
    assert _relearn_observed_cost(p, pricing_mode="subscription").endswith("tok")


def test_offered_cluster_has_no_gate_line(db):
    """The gate line is strictly for suppressed writes; an offered fix renders
    none, so the list does not accuse every row of being un-actionable."""
    from tokenjam.cli.cmd_optimize import _relearn_write_gate_line

    offered = RelearnCluster(
        signature="ok", title="ok", family_key="fam", sessions=3, occurrences=3,
        repos=["demo"], rung=1, scope="project", proposed_fix="do the thing",
        write_offered=True, write_blocked_reason="",
    )
    assert _relearn_write_gate_line(offered) == ""


def test_net_negative_wording_is_modelled_not_observed():
    """#355: the verdict is arithmetic over quantities we can count, and the
    measurement that would settle it returned null. No surface may state it as
    an observed result, and the blind spot is named in the module docstring."""
    from tokenjam.core.optimize import write_budget as wb

    assert "quantities we can count" in wb.REASON_NET_NEGATIVE
    assert "modelled" in wb.REASON_NET_NEGATIVE
    # The prevention blind spot is stated to the user, not just to the reader
    # of the source.
    assert "preventing" in wb.REASON_NET_NEGATIVE
    doc = wb.__doc__ or ""
    assert "blind spot" in doc.lower()
    assert "repeat-task-codification-measurement" in doc


def test_short_label_is_derived_on_read_for_an_older_cache():
    """The Review inbox reads the STORED proposal cache, and the row renders
    only the short label. A cache written before that field existed must still
    produce one, or every gated row silently loses its explanation for a whole
    upgrade cycle — the exact invisibility the label was added to fix.

    Mirrors the existing on-read stamping of `proposal_id` / `advise_only_reason`.
    """
    from tokenjam.core.optimize import write_budget as wb
    from tokenjam.core.optimize.relearn_proposals import stamp_proposal_ids

    legacy = {
        "clusters": [
            # No `write_blocked_short` key at all, and the PRE-hedging wording
            # of the net-negative sentence — exactly what the previous build
            # wrote to disk.
            {
                "signature": "old", "write_offered": False,
                "write_blocked_reason": wb.LEGACY_REASON_NET_NEGATIVE,
            },
            {
                "signature": "current", "write_offered": False,
                "write_blocked_reason": wb.REASON_PLACEHOLDER,
            },
        ],
    }
    out = stamp_proposal_ids(legacy)["clusters"]
    assert out[0]["write_blocked_short"] == wb.short_reason(wb.REASON_NET_NEGATIVE)
    assert out[0]["write_blocked_short"] != "no permanent fix offered", (
        "the legacy net-negative wording must not degrade to the generic label"
    )
    assert out[1]["write_blocked_short"] == wb.short_reason(wb.REASON_PLACEHOLDER)
    # Pure: the input dict is never mutated (the stamper runs inside a cache
    # write and must never surprise its caller).
    assert "write_blocked_short" not in legacy["clusters"][0]


def test_relearn_finding_round_trips_its_observed_dollar_figure():
    """`report_to_dict` / `report_from_dict` must preserve relearn's observed
    figure. relearn's reader was the one in `runner.py` restoring the token
    field but not its USD twin, so a rehydrated report kept ~16.7M tokens and
    silently lost the $46 that went with them -- and the dollar field is what
    every leading surface renders.
    """
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict

    finding = RelearnFinding(
        clusters=[], sessions_scanned=3,
        past_overspend_tokens=1_234_567,
        past_overspend_usd=46.62,
        past_overspend_basis="observed over the scanned corpus",
    )
    # Round-trip through the report envelope the cache actually stores.
    from tokenjam.core.optimize.runner import OptimizeReport, WindowSummary

    window = WindowSummary(
        since=BASE - timedelta(days=30), until=BASE, days=30, sessions=3,
        spans=10, total_tokens=1, total_cost_usd=1.0, thin_data=False,
        active_days=28,
    )
    report = OptimizeReport(window=window, downgrade=None)
    report.findings["relearn"] = finding
    back = report_from_dict(report_to_dict(report)).findings["relearn"]

    assert back.past_overspend_tokens == 1_234_567
    assert back.past_overspend_usd == 46.62
    assert back.past_overspend_basis == "observed over the scanned corpus"
    # A payload written before the fields existed rehydrates to the declared
    # defaults, never to None on a non-optional int.
    legacy = report_from_dict({"findings": {"relearn": {"sessions_scanned": 1}}})
    assert legacy.findings["relearn"].past_overspend_tokens == 0


# --------------------------------------------------------------------------- #
# The head term is MEASURED, and the claim is disjoint from `resend`.
#
# relearn priced a pothole at a flat 1,500 tokens -- an unmeasured guess for
# "one extra assistant turn's overhead". Measured on a real corpus, one turn
# costs ~140k tokens (836 fresh + 138k cache-read + 795 out), because a coding
# turn re-sends the whole context. The constant was ~15-20x low, which is why
# relearn read as noise: 70% of the inbox cards carrying 0.8% of the money.
#
# The fix splits one conflated constant into the two quantities it was standing
# in for: the forced RETRY TURN (measured, and relearn's own claim) and the
# error TEXT re-read on later calls (~1,500 tokens really is right for a block
# of error text, and it is `resend`'s money, so it is observed but not claimed).
# --------------------------------------------------------------------------- #


def test_head_term_is_measured_from_the_corpus_not_a_constant(db):
    """The recovery turn is priced from what a call actually cost in the
    contributing sessions, so a corpus of expensive calls yields a bigger
    figure than one of cheap calls off the identical failure count."""
    from tokenjam.core.optimize.analyzers.relearn import (
        _measured_turn_tokens,
        cluster_failures,
    )
    from tokenjam.core.optimize.rate_profile import blended_rate_profile

    failures = []
    for i in range(MIN_RECURRING_SESSIONS):
        session_id = f"m{i}"
        _priced_session(db, session_id)
        failures.append(_episode(
            session_id,
            "File has not been read yet. Read it first before writing to it.",
            ts=(BASE + timedelta(days=i)).isoformat(), tool="Edit",
        ))
    sessions = {f.session_id for f in failures}
    profile = blended_rate_profile(db.conn, session_ids=sessions)
    measured = _measured_turn_tokens(db.conn, sessions, profile)

    assert measured is not None, "a priced corpus must yield a measured turn"
    # Never BELOW the error-text constant: a forced retry carries the error
    # text, so it cannot cost less than the text alone.
    assert measured >= GROUNDED_TOKENS_PER_OCCURRENCE
    # And it must actually be derived, not the constant echoed back: these
    # sessions carry 2,000-token calls, well above the 1,500 floor.
    assert measured > GROUNDED_TOKENS_PER_OCCURRENCE

    clusters = list(cluster_failures(failures).values())
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    # The head alone already exceeds what the OLD flat model charged for head
    # plus tail combined.
    assert p.past_overspend_tokens > p.occurrences * GROUNDED_TOKENS_PER_OCCURRENCE


def test_relearn_claims_the_retry_turn_and_never_the_reread_tail(db):
    """The disjointness rule between the two analyzers, asserted.

    `resend` prices redundant context inside calls that had to happen.
    `relearn` prices a call that should not have happened at all. The error
    text's re-read tail belongs to the first, so it appears in relearn's
    OBSERVED figure and never in its CLAIM -- otherwise the same tokens are
    priced on two cards (CLAUDE.md rule 27).
    """
    from tokenjam.core.optimize.analyzers.relearn import cluster_failures

    # Sized to clear `write_budget.MIN_NET_WRITE_USD`, since this test asserts
    # on the OFFERED path: 5 sessions of ~100k-token turns hitting the same
    # pothole 4 times each, which is an ordinary shape on a real coding corpus.
    failures = []
    for i in range(MIN_RECURRING_SESSIONS + 2):
        session_id = f"d{i}"
        _priced_session(db, session_id, input_tokens=100_000)
        for occurrence in range(4):
            failures.append(_episode(
                session_id,
                "File has not been read yet. Read it first before writing to it.",
                ts=(BASE + timedelta(days=i, minutes=occurrence)).isoformat(),
                tool="Edit",
            ))
    clusters = list(cluster_failures(failures).values())
    proposals, _ = build_proposals(
        clusters, conn=db.conn, window_days=30.0, persona="claude-code",
        repo_cwd_map={"demo": "/tmp/demo"},
    )
    p = proposals[0]
    assert p.write_offered, "this family has a real fix; expected a claim"

    # No forward claim field exists any more — a relearn cluster shows its
    # PAST figure only. The disjointness rule this test used to assert on
    # `estimated_recoverable_tokens` now lives entirely inside the observed
    # figure: the re-read tail is a COMPONENT of `past_overspend_tokens` (not
    # an addend), broken out separately as `past_reread_tokens`, and never
    # exceeds the total it is a component of.
    assert not hasattr(p, "estimated_recoverable_tokens")
    assert p.past_reread_tokens <= p.past_overspend_tokens
    # The one basis string says the overlap is disclosed, never claimed twice.
    from tokenjam.core.optimize.analyzers.relearn import PAST_OVERSPEND_BASIS

    assert "must never be added together" in PAST_OVERSPEND_BASIS
