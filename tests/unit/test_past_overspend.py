"""The past-overspend contract: the number the product leads with.

Every user-facing figure on the Dashboard hero and the Review inbox is
PAST TENSE and window-OBSERVED — priced at the rates it actually billed at,
over a window that has already happened. These tests pin the four properties
that make that figure safe to show, because each of them failing turns the
feature net-negative rather than merely wrong:

  0. the figure a reader will call waste/overspend is the AVOIDABLE amount,
     never the full observed cost. Waste is only what could have been avoided;
     unavoidable spend is cost. Leading with the cost figure implicitly
     asserted that everything it exceeded the avoidable figure by had been
     shown to be unavoidable — a claim nothing in the analyzer supports, since
     the two are computed over DIFFERENT POPULATIONS;
  1. neither figure is ever summed into a recoverable total, and the two are
     never summed with each other (the avoidable figure is a subset of the
     cost figure, so adding them double-counts);
  2. neither is multiplied by the central 30-day pacing ratio — both sides of
     the pair read the same raw analysed window, so their difference is
     attributable to avoidability alone and never to a time-basis artifact;
  3. the surfaces read them off the payload rather than deriving them in JS
     (two derivations of one number drift the moment either side is edited).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core.optimize.cost_proposals import (
    CostProposal,
    _with_past_overspend,
    backfill_legacy_past_overspend_fields,
    compute_projection_ratio,
    cost_proposals_from_report,
    estimated_recoverable_rollup,
    past_overspend_rollup,
)
from tokenjam.core.optimize.types import (
    DowngradeFinding,
    OptimizeReport,
    WindowSummary,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)

UI = Path(__file__).resolve().parents[2] / "tokenjam" / "ui" / "index.html"


def _proposal(**kw) -> CostProposal:
    base = dict(
        kind="cost", analyzer="downsize", signature="cost:downsize",
        title="t", target_key={}, evidence="e", baseline={}, advise_text="a",
    )
    base.update(kw)
    return CostProposal(**base)


def _resend_finding():
    from tokenjam.core.optimize.analyzers.context_resend import ResendFinding

    return ResendFinding(
        sessions_examined=40, repeat_share=0.93, repeat_tokens=10_000,
        estimated_recoverable_tokens=1_400_000_000,
        estimated_recoverable_usd=703.78,
        estimate_basis="resend basis",
        fix_compaction="Run /compact.",
        cost_of_waste_usd=7_038.85,
        cost_of_waste_tokens=14_382_971_851,
        cost_of_waste_basis="observed; do NOT read this as a saving",
        # The real analyzer always emits this alongside a cost figure (pinned
        # by test_context_resend.py); stated here because this fixture builds
        # the finding directly rather than running the analyzer.
        coverage_note=(
            "COVERAGE. The cost figure covers every session with repeat volume; "
            "the avoidable figure was computed over 12 of them."
        ),
    )


# --- 0. the waste-labelled figure is the AVOIDABLE one --------------------- #

def test_the_waste_labelled_figure_never_exceeds_the_avoidable_figure():
    """THE invariant this whole surface rests on.

    `past_overspend_usd` is what every waste/overspend-worded surface renders.
    It may never exceed the avoidable figure, because a figure a reader calls
    "wasted" is a claim that it could have been avoided. Before this was
    enforced, resend put its full $6,972 cost of re-sending context here beside
    a $398 avoidable figure — implicitly asserting 94.3% of it was unavoidable,
    when in truth that 94.3% was analysed on another card, filtered out below
    the context floor, or outside the tail definition, and was never shown to
    be unavoidable at all.

    Asserted across a whole report rather than one hand-built proposal, so a
    future analyzer that routes a cost figure into the headline slot fails here
    rather than shipping.
    """
    from tokenjam.core.optimize.analyzers.context_resend import ResendFinding

    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=200,
        spans=1_000, total_tokens=1, total_cost_usd=50.0, thin_data=False,
        active_days=28,
    )
    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        estimated_recoverable_usd=3.0, estimated_recoverable_tokens=1_000,
        percent_of_tokens=35.0, estimate_basis="downsize basis",
    )
    report = OptimizeReport(window=window, downgrade=dg)
    report.findings["resend"] = _resend_finding()

    props = cost_proposals_from_report(report)
    assert props, "fixture produced no proposals — the guard would be vacuous"
    for p in props:
        if p.past_overspend_usd is None:
            continue
        assert p.estimated_recoverable_usd is not None, (
            f"{p.analyzer} renders a waste figure with no avoidable figure behind it"
        )
        assert p.past_overspend_usd <= p.estimated_recoverable_usd + 1e-9, (
            f"{p.analyzer} labels {p.past_overspend_usd} as overspend but only "
            f"{p.estimated_recoverable_usd} of it was shown to be avoidable"
        )
        if p.observed_cost_usd is not None:
            # The cost figure is legitimately larger — it just may never be the
            # one wearing the waste label, and it must arrive with its coverage
            # stated rather than leaving the gap to imply necessity.
            assert p.observed_cost_usd >= p.past_overspend_usd
            assert p.coverage_note, f"{p.analyzer} ships a cost figure with no coverage note"

    block = past_overspend_rollup(props)
    assert block["past_overspend_usd"] <= block["observed_cost_usd"]


def test_the_rollup_headline_is_avoidable_and_the_cost_total_stays_separate():
    props = [
        _with_past_overspend(_proposal(
            analyzer="resend", signature="cost:resend",
            estimated_recoverable_usd=398.41,
            cost_of_waste_usd=6_972.80, cost_of_waste_basis="observed",
            coverage_note="COVERAGE. ...",
        )),
        _with_past_overspend(_proposal(signature="cost:downsize",
                                       estimated_recoverable_usd=40.0)),
    ]
    block = past_overspend_rollup(props)
    # The headline sums avoidable figures only — one meaning across analyzers.
    assert block["past_overspend_usd"] == pytest.approx(438.41)
    # The cost total ships, separately, and is never folded into the headline.
    assert block["observed_cost_usd"] == pytest.approx(6_972.80)
    assert block["past_overspend_usd"] != pytest.approx(6_972.80 + 438.41)
    by_analyzer = {a["analyzer"]: a for a in block["by_analyzer"]}
    assert by_analyzer["resend"]["usd"] == 398.41
    assert by_analyzer["resend"]["observed_cost_usd"] == 6_972.80
    assert by_analyzer["downsize"]["observed_cost_usd"] is None


# --- 1. never summed into a recoverable total ------------------------------ #

def test_past_overspend_is_never_summed_into_the_recoverable_rollup():
    # The two answer different questions ("what of this could I have avoided"
    # vs "what does the fix return going forward"). The recoverable rollup must
    # read only the estimated_* fields no matter how large the observed cost is.
    prop = _with_past_overspend(_proposal(
        analyzer="resend", signature="cost:resend",
        estimated_recoverable_usd=703.78, estimated_recoverable_tokens=1_400_000_000,
        cost_of_waste_usd=7_038.85, cost_of_waste_tokens=14_382_971_851,
        cost_of_waste_basis="observed",
    ))
    assert prop.past_overspend_usd == 703.78
    assert prop.observed_cost_usd == 7_038.85

    rollup = estimated_recoverable_rollup([prop])
    assert rollup["estimated_recoverable_usd"] == 703.78
    assert rollup["estimated_recoverable_tokens"] == 1_400_000_000
    assert rollup["projected_usd_30d"] == 703.78          # ratio blocked, thin window
    # No key of the recoverable rollup carries the observed figure, under any
    # name — this is the assertion that survives someone adding a field later.
    for key, value in rollup.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert value != pytest.approx(7_038.85), f"{key} leaked the observed figure"
            assert value != pytest.approx(7_038.85 + 703.78), f"{key} summed the two"


def test_past_overspend_rollup_and_recoverable_rollup_read_disjoint_fields():
    # A proposal carrying ONLY a past figure contributes nothing to the
    # recoverable rollup, and one carrying ONLY a recoverable figure
    # contributes nothing to the past rollup. Structural separation, not a
    # convention a caller has to remember.
    past_only = _proposal(signature="a", past_overspend_usd=100.0, past_overspend_tokens=10)
    rec_only = _proposal(signature="b", estimated_recoverable_usd=7.0,
                         estimated_recoverable_tokens=3)

    assert estimated_recoverable_rollup([past_only])["estimated_recoverable_usd"] == 0.0
    assert past_overspend_rollup([rec_only])["past_overspend_usd"] == 0.0
    assert past_overspend_rollup([past_only])["past_overspend_usd"] == 100.0


def test_the_two_totals_are_never_added_to_each_other():
    # They overlap by construction — the avoidable total is a SUBSET of the
    # cost total — so any surface adding them double-counts the avoidable part.
    props = [
        _with_past_overspend(_proposal(
            analyzer="resend", signature="cost:resend",
            estimated_recoverable_usd=703.78,
            cost_of_waste_usd=7_038.85, cost_of_waste_basis="observed",
        )),
        _with_past_overspend(_proposal(signature="cost:downsize",
                                       estimated_recoverable_usd=40.0)),
    ]
    block = past_overspend_rollup(props)
    combined = 7_038.85 + 743.78
    for key, value in block.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert value != pytest.approx(combined), f"{key} summed the two totals"


# --- 2. never paced ---------------------------------------------------------#

def test_no_pacing_ratio_is_applied_to_a_past_overspend_figure():
    # The ONE sanctioned projection in this product is the central 30-day
    # active-day pace, applied centrally to the monthly fields. Applying it to
    # an observation would turn the one figure that needs no trust into one
    # that needs the most.
    ratio, label = compute_projection_ratio(window_days=30, active_days=10, n_sessions=200)
    assert ratio == 3.0 and label == "per 30 days"          # a live, non-trivial ratio

    dg = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-sonnet-5"},
        estimated_recoverable_usd=3.0, estimated_recoverable_tokens=1_000,
        percent_of_tokens=35.0, estimate_basis="downsize basis",
    )
    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=200, spans=1_000,
        total_tokens=1, total_cost_usd=50.0, thin_data=False, active_days=10,
    )
    props = cost_proposals_from_report(OptimizeReport(window=window, downgrade=dg))
    prop = next(p for p in props if p.analyzer == "downsize")

    # The projection DID happen on the monthly fields...
    assert prop.estimated_monthly_usd == pytest.approx(9.0)
    # ...and did NOT touch the observed one, which stays the raw window figure.
    assert prop.past_overspend_usd == pytest.approx(3.0)
    assert prop.past_overspend_tokens == 1_000

    # And the rollup can't pace it either: it takes no pace to project from.
    block = past_overspend_rollup(props)
    assert block["past_overspend_usd"] == pytest.approx(3.0)
    assert "projection_ratio" not in block
    with pytest.raises(TypeError):
        past_overspend_rollup(props, active_days=10, n_sessions=200)  # type: ignore[call-arg]


def test_past_overspend_reads_the_netted_figure_not_the_gross_one():
    # A write-bearing card is netted against what its rule costs to KEEP. The
    # observed figure must follow the netting, not the pre-net gross, or the
    # card would state an overspend larger than the product's own arithmetic
    # says it is.
    prop = _with_past_overspend(_proposal(
        signature="cost:reuse", analyzer="reuse",
        estimated_recoverable_usd=4.0, gross_recoverable_usd=9.0,
    ))
    assert prop.past_overspend_usd == 4.0


# --- the single-number vs paired-number rule ------------------------------- #

def test_only_a_finding_with_its_own_observed_cost_renders_two_numbers():
    # Verified against the analyzer sources: for 6 of the 7 CC-eligible
    # analyzers the "recoverable" figure IS the observed window avoidable
    # spend, so rendering both would show one quantity twice. Only resend
    # computes a separate TOTAL cost of the behaviour it flags, so only resend
    # gets a second number.
    single = _with_past_overspend(_proposal(estimated_recoverable_usd=12.0,
                                            estimated_recoverable_tokens=99,
                                            estimate_basis="downsize basis"))
    assert single.past_overspend_usd == 12.0
    assert single.observed_cost_usd is None
    assert single.observed_cost_tokens is None
    assert "downsize basis" in single.past_overspend_basis

    paired = _with_past_overspend(_proposal(
        analyzer="resend", estimated_recoverable_usd=703.78,
        estimate_basis="resend basis",
        cost_of_waste_usd=7_038.85, cost_of_waste_basis="Do NOT read this as a saving.",
    ))
    assert paired.past_overspend_usd == 703.78
    assert paired.observed_cost_usd == 7_038.85
    # Each basis travels WITH the figure it qualifies, unweakened.
    assert "resend basis" in paired.past_overspend_basis
    assert "Do NOT read this as a saving." in paired.observed_cost_basis


def test_resend_adapter_carries_both_observed_figures_end_to_end():
    from tokenjam.core.optimize.cost_proposals import _resend_to_proposals

    prop = _with_past_overspend(
        _resend_to_proposals(_resend_finding(), persona="claude-code")[0]
    )
    assert prop.past_overspend_usd == 703.78
    assert prop.past_overspend_tokens == 1_400_000_000
    assert prop.observed_cost_usd == 7_038.85
    assert prop.past_overspend_usd < prop.observed_cost_usd
    # The evidence line states the observation without recovery vocabulary,
    # and without asserting that the rest of the cost was unavoidable.
    assert "recoverable" not in prop.evidence
    assert "as cost, not waste" in prop.evidence
    assert "inherently re-sends" not in prop.evidence
    # The cost figure is stated exactly once on the card, on its own COST line.
    assert "7,038.85" not in prop.evidence


def test_legacy_cached_proposal_backfills_the_same_derivation_on_read():
    # A cache written before these fields existed would otherwise render an em
    # dash where the page's headline number belongs, for up to a scheduled
    # recompute interval.
    single = backfill_legacy_past_overspend_fields(
        {"analyzer": "cache", "estimated_recoverable_usd": 4.5,
         "estimated_recoverable_tokens": 700, "estimate_basis": "cache basis"}
    )
    assert single["past_overspend_usd"] == 4.5
    assert single.get("observed_cost_usd") is None

    paired = backfill_legacy_past_overspend_fields(
        {"analyzer": "resend", "estimated_recoverable_usd": 703.78,
         "cost_of_waste_usd": 7_038.85, "cost_of_waste_basis": "observed"}
    )
    assert paired["past_overspend_usd"] == 703.78
    assert paired["observed_cost_usd"] == 7_038.85

    # Never overwrites a current entry.
    current = {"past_overspend_usd": 1.0, "estimated_recoverable_usd": 99.0}
    assert backfill_legacy_past_overspend_fields(current)["past_overspend_usd"] == 1.0


# --- 3. the UI reads the payload, it does not recompute -------------------- #
# No JS runner in CI (see CLAUDE.md -> Testing the UI), so the guard is a
# static grep over the single-file SPA, same as every other Lens regression.

@pytest.fixture(scope="module")
def ui() -> str:
    return UI.read_text()


def test_ui_renders_the_observed_figure_at_all(ui):
    # The gap this closes: the backend computed the figure, priced it per
    # token class, wrote an honesty basis for it, shipped it on the payload,
    # and handed it to a dashboard that referenced it zero times.
    assert ui.count("past_overspend_usd") > 0
    assert "PastOverspendBand" in ui


def test_ui_never_derives_a_past_overspend_figure_client_side(ui):
    # Single-compute-path: if the UI needs a number, the endpoint provides it.
    # No pricing, no pacing, no window arithmetic in JS.
    for forbidden in (
        "past_overspend_usd *", "past_overspend_tokens *",
        "* past_overspend", "past_overspend_usd /", "cost_of_waste",
    ):
        assert forbidden not in ui, f"UI derives its own figure: {forbidden}"


def test_both_headline_surfaces_read_the_same_server_block(ui):
    # The Dashboard hero and the Review inbox headline render the SAME
    # component over the SAME payload key, so they cannot disagree on basis,
    # window, or number.
    assert ui.count("<${PastOverspendBand}") == 2
    assert "setCostPastOverspend(r.past_overspend || null)" in ui
    assert "setHeroPast((r && r.past_overspend) || null)" in ui
    # The hero fetches on its OWN effect rather than inside the Dashboard's
    # triage Promise.all: that batch resolves only when its slowest member
    # does (a 30-day analyzer sweep), and a headline that waits on an analyzer
    # sweep is a headline nobody sees. Verified live: the triage band was
    # still showing its loading shimmer minutes after the page settled.
    assert "api('/relearn/cost-proposals')\n      .then(r => { if (live) setHeroPast" in ui


def test_ui_labels_are_past_tense_and_carry_no_recovery_vocabulary(ui):
    band = ui[ui.index("function PastOverspendBand"):]
    band = band[:band.index("\n}")]
    # The headline is the AVOIDABLE amount, and says so — it is no longer
    # labelled as everything the behaviour cost.
    assert "What you could have avoided" in band
    assert "was avoidable" in band
    assert "recoverable" not in band
    assert "could save" not in ui
    # No ratio framing ("recovering $X of a $Y problem") anywhere.
    assert "recovering $" not in ui
    # The cost line is present, past tense, and worded as COST — never waste.
    assert "cost ${fmtUsd(cost)} in total" in band
    assert "That is cost, not waste" in band
    fn = ui[ui.index("function observedCostSentence"):]
    fn = fn[:fn.index("\n}")]
    assert "this behaviour cost " in fn
    assert "recoverable" not in fn
    assert "wasted" not in fn


def test_no_ui_surface_labels_the_total_cost_figure_as_waste(ui):
    # The `observed_cost_*` fields carry the unavoidable-inclusive total. No
    # rendering of them may sit next to waste/overspend wording, because the
    # part they exceed the avoidable figure by was never shown to be avoidable.
    for marker in ("observed_cost_usd", "observed_cost_tokens"):
        for idx in _all_indices(ui, marker):
            around = ui[max(0, idx - 400): idx + 400].lower()
            for forbidden in ("wasted", "waste you", "overspent", "overspend you"):
                assert forbidden not in around, (
                    f"{marker} is rendered within 400 chars of '{forbidden}'"
                )


def test_the_card_states_the_two_figures_differing_coverage(ui):
    # A cost spanning every session shown beside an avoidable figure spanning a
    # filtered subset must never be presented as two views of one quantity: the
    # ratio between them would otherwise read as "94% of this was unavoidable".
    card = ui[ui.index("function CostProposalCard"):]
    card = card[:card.index("\n// The headline band")]
    assert "${prop.coverage_note}" in card
    assert "observedCostSentence(prop)" in card


def _all_indices(haystack: str, needle: str):
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return
        yield i
        start = i + 1


def test_the_basis_is_reachable_from_the_card_not_only_on_hover(ui):
    # "Do NOT read this as a saving" only protects a reader who can reach it —
    # a hover title is unreachable on touch, so the card carries an expandable
    # block as well.
    card = ui[ui.index("function CostProposalCard"):]
    card = card[:card.index("\n// The headline band")] if "\n// The headline band" in card else card
    assert 'title=${prop.past_overspend_basis' in card
    assert "How this number was derived" in card
    assert "${prop.past_overspend_basis}" in card


def test_the_observed_figure_is_visually_separated_from_recoverable_tiles(ui):
    # Not the same colour treatment, not the same row: every "what you could
    # get back" surface is accent-blue (.rec-amount) or success-green; this
    # one renders in body text in its own full-width band.
    css = ui[ui.index(".po-band {"):ui.index(".po-basis {")]
    assert "var(--accent)" not in css
    assert "var(--success)" not in css
    assert "color: var(--text);" in css
