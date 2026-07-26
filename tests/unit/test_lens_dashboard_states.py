"""The Dashboard triage band must never report absence when nothing answered.

Background: ``GET /api/v1/optimize`` runs the analyzer sweep server-side and,
measured against a real 794MB corpus over a 30-day window, answers HTTP 200 with
five findings after 356 seconds (480s while the page was also polling). The
Dashboard fetched it inside the triage ``Promise.all`` as
``api('/optimize', ...).catch(() => null)``, and ``recoverableTiles(null)``
returns ``[]`` exactly like a completed-but-empty report does. The band therefore
rendered "No recoverable candidates." over a corpus holding thousands of dollars
of them. The same collapse sat under all five Health tiles, whose reads each fell
back to an empty default, publishing "0 unread alerts / all clear" for a read
that had failed.

The rule those bugs violate: a surface may only claim a figure it actually has.
The decision of what each band is ALLOWED to say now lives in two pure functions
in the served ``index.html`` (``recoverableBandState`` and ``healthValueFor``),
which this module extracts and runs under node -- the same trick
``test_lens_select_all_behaviour.py`` uses, and the reason those functions are
pure. A string match on the source would still pass if the state logic were
wrong, and "wrong" here means telling a user they have no waste when the answer
is unknown.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Behavioural: the state machine, run under node
# --------------------------------------------------------------------------- #
_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS evaluation"
)


def _state_machine_source() -> str:
    """The two pure deciders, lifted straight out of the served page.

    ``UNKNOWN_FIGURE`` is declared up with the formatters (the KPI row needs it
    too, and one page must not show two different kinds of "unknown"), so it is
    lifted separately rather than falling inside the deciders' slice.
    """
    src = _UI.read_text(encoding="utf-8")
    glyph_line = "const UNKNOWN_FIGURE = '?';"
    assert glyph_line in src, "the unknown-figure glyph moved; update this extractor"
    start = src.index("function recoverableBandState")
    end = src.index("function HealthTile", start)
    return glyph_line + "\n" + src[start:end]


def _run_js(expr: str):
    script = _state_machine_source() + "\nconsole.log(JSON.stringify(" + expr + "));"
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


def _band(phase: str, data, tile_count: int) -> str:
    return _run_js(
        "recoverableBandState(%s, %s, %d)"
        % (json.dumps(phase), "null" if data is None else json.dumps(data), tile_count)
    )


def _health_value(status: str, value):
    return _run_js("healthValueFor(%s, %s)" % (json.dumps(status), json.dumps(value)))


# --- the reported bug, pinned ---------------------------------------------- #
@_node
def test_a_timeout_with_no_answer_is_not_the_empty_state():
    # THE bug. A sweep that never came back used to reach the tiles as an empty
    # report; 'none' is the only state permitted to say there is nothing here.
    assert _band("timeout", None, 0) == "timeout"


@_node
def test_a_failed_read_with_no_answer_is_not_the_empty_state():
    assert _band("error", None, 0) == "failed"


@_node
def test_an_unfinished_read_is_not_the_empty_state():
    assert _band("loading", None, 0) == "scanning"


@_node
def test_timeout_and_failure_stay_distinguishable():
    # They call for different next moves (keep waiting on a sweep that is still
    # running, vs. retry something that broke), so they must not share a state.
    assert _band("timeout", None, 0) != _band("error", None, 0)


@_node
def test_only_a_completed_response_may_claim_there_is_nothing_recoverable():
    # The one legitimate route to "No recoverable candidates.": a real payload
    # that really carried no candidates.
    assert _band("ready", {"findings": {}}, 0) == "none"
    # And every non-answer is excluded from it.
    for phase in ("loading", "timeout", "error"):
        assert _band(phase, None, 0) != "none"


@_node
def test_a_completed_response_with_candidates_renders_tiles():
    assert _band("ready", {"findings": {"resend": {}}}, 5) == "tiles"


# --- holding a previous answer --------------------------------------------- #
@_node
def test_a_held_answer_survives_a_refresh_that_does_not_land():
    # Showing the last complete result, labelled as such, beats blanking numbers
    # we do have. It must NOT read as a fresh answer, hence its own state.
    held = {"findings": {"resend": {}}}
    assert _band("timeout", held, 5) == "stale"
    assert _band("error", held, 5) == "stale"


@_node
def test_a_quiet_refresh_over_a_held_answer_keeps_showing_the_tiles():
    # In-flight refresh with numbers already on screen: no warning, no blanking.
    assert _band("loading", {"findings": {"resend": {}}}, 5) == "tiles"


@_node
def test_a_held_answer_that_was_empty_still_reads_as_empty_not_as_tiles():
    assert _band("loading", {"findings": {}}, 0) == "none"


# --- the health tiles' figures --------------------------------------------- #
@_node
def test_an_unknown_health_figure_is_never_rendered_as_zero():
    # A zero on this band reads as an all-clear ("no unread alerts", "no agents
    # drifting"). Only an answered read may put a number in the slot.
    assert _health_value("error", 0) != "0"
    assert _health_value("error", 0) == "?"
    assert _health_value("error", 7) == "?"


@_node
def test_a_timed_out_health_figure_is_unknown_not_zero_and_not_a_shimmer():
    # "Budgets at risk" rides on the same minutes-long sweep as the band. An
    # indefinite shimmer would read as progress past the point where the page has
    # already admitted it does not know, so this state gets the unknown mark.
    assert _health_value("timeout", 0) == "?"
    assert _health_value("timeout", 0) is not None


@_node
def test_a_loading_health_figure_yields_no_text_at_all():
    # null means "render the shimmer here", so not even a placeholder glyph
    # occupies the number slot while the read is outstanding.
    assert _health_value("loading", 0) is None


@_node
def test_an_answered_health_figure_renders_its_number_including_a_real_zero():
    # A genuine zero from a read that answered is a fact and must still show.
    assert _health_value("ready", 0) == "0"
    assert _health_value("ready", 12) == "12"


# --------------------------------------------------------------------------- #
# Static: the wiring that feeds those deciders
# --------------------------------------------------------------------------- #
def test_optimize_is_no_longer_swallowed_into_an_empty_report(html):
    # The exact line that manufactured the false claim.
    assert "api('/optimize', { since, fast: 'true' }).catch(() => null)" not in html


def test_recoverable_tiles_are_derived_only_from_an_answer_we_hold(html):
    # recoverableTiles() flattens a missing report and an empty one to the same
    # [], so the null case must never reach it.
    assert "const tiles = optData ? recoverableTiles(optData) : [];" in html
    assert "recoverableTiles(d.opt)" not in html


def _dashboard(html: str) -> str:
    start = html.index("function DashboardView")
    end = html.index("// Two lenses, one router", start)
    return html[start:end]


def test_no_panel_waits_on_another_panel_s_endpoint(html):
    # There is no shared batch left. A Promise.all resolves only when its SLOWEST
    # member does, and these members range from ~13.6s (/traces, /status) through
    # 27.0s (/cost) to 356s (the /optimize sweep) on a real 794MB corpus, so one
    # batch made every panel as slow as the worst of them and left the page with
    # nothing on it in the meantime.
    dash = _dashboard(html)
    assert "await Promise.all([" not in dash
    for read in (
        "const costRead = useTriageRead(",
        "const statusRead = useTriageRead(",
        "const alertsRead = useTriageRead(",
        "const driftRead = useTriageRead(",
        "const tracesRead = useTriageRead(",
        "const relearnRead = useTriageRead(",
    ):
        assert read in dash


def test_no_read_stacks_queries_behind_itself(html):
    # A 30s poll in front of a 27-second endpoint would otherwise keep opening new
    # queries against a DB the previous one still holds.
    assert "if (inFlight.current) return;" in html      # useTriageRead
    assert "if (optInFlight.current) return;" in html   # the sweep


def test_the_poll_interval_is_installed_once(html):
    # Listing the six refresh callbacks as effect deps would tear the timer down
    # and reinstall it on every window change, so a 30s timer could keep being
    # reset before it ever fired.
    dash = _dashboard(html)
    assert "const tick = () => { if (document.visibilityState === 'visible') pollRef.current(); };" in dash
    assert "document.addEventListener('visibilitychange', tick);" in dash


def test_a_window_change_drops_the_previous_window_s_answer(html):
    # Last window's figures under this window's label would be a fresh lie rather
    # than a stale truth, so a deps change clears the held answer.
    hook = html[html.index("function useTriageRead(run, deps) {"):]
    hook = hook[:hook.index("function readStatus")]
    assert "setSt({ phase: 'loading', data: null, error: null });" in hook


def test_the_band_has_a_bounded_wait_that_clears_the_measured_worst_case(html):
    # Bounded so 'loading' cannot be forever, but above the measured 480s so the
    # notice does not fire on a scan that was merely slow.
    assert "const OPTIMIZE_WAIT_MS = 10 * 60 * 1000;" in html


def test_the_timeout_state_says_it_timed_out_and_offers_a_way_forward(html):
    assert "Still scanning after ${OPTIMIZE_WAIT_MIN} minutes." in html
    assert "Keep waiting" in html
    # It must explicitly refuse the all-clear reading.
    assert "that is not the same as an all-clear" in html


def test_the_failure_state_says_it_failed_and_offers_a_retry(html):
    assert "Couldn't scan for recoverable waste." in html
    assert "Try again" in html


def test_the_degradable_triage_reads_no_longer_fall_back_to_empty_defaults(html):
    # Each of these published a zero for a read that never answered.
    for gone in (
        "api('/status').catch(() => ({ agents: [] }))",
        "api('/alerts', { since, unread: 'true' }).catch(() => ({ alerts: [] }))",
        "api('/drift').catch(() => ({ agents: [] }))",
        "api('/traces', { since, limit: 6 }).catch(() => ({ traces: [] }))",
        "api('/relearn/proposals').catch(() => ({ finding: null }))",
    ):
        assert gone not in html


def test_an_unreported_kpi_field_is_unknown_not_zero(html):
    # fmtCount(null) returns the string "0" and fmtCost(null) returns "$0.0000".
    # Both are shared formatters used by every screen, so the guard sits at the
    # KPI call site instead: a field the payload omitted must not become a figure.
    assert "const kpiFigure = (v, fmt) => (v == null ? UNKNOWN_FIGURE : fmt(v));" in html
    assert "value: kpiFigure(kpis.tokens, fmtTokens)" in html
    assert "value: kpiFigure(kpis.sessions, fmtCount)" in html
    assert "value: kpiFigure(kpis.events, fmtCount)" in html
    # And the bare formatters are no longer handed those fields directly.
    assert "fmtCount(kpis.sessions)" not in html
    assert "fmtCount(kpis.events)" not in html
    assert "fmtTokens(kpis.tokens)" not in html


def test_an_unreported_spend_field_does_not_become_a_zero_spend_tile(html):
    # `(null || 0) / fee` rendered "0.0× plan value" and fmtCost(null) rendered
    # "$0.0000"; on a spend tile a zero reads as "this window cost you nothing".
    spend = html[html.index("function spendTileDisplay"):]
    spend = spend[:spend.index("function PlanBadge")]
    assert "const unknown = spendUsd == null;" in spend
    assert "if (unknown) return { label: 'Implied value', value: UNKNOWN_FIGURE };" in spend
    assert "value: unknown ? UNKNOWN_FIGURE : fmtCost(spendUsd)" in spend
    assert "(spendUsd || 0) / framing.plan_monthly_usd" not in spend


def test_one_page_shows_one_kind_of_unknown(html):
    # The Health tiles and the KPI row share the glyph, so a half-loaded page does
    # not show a "?" in one band and something else in another.
    assert html.count("const UNKNOWN_FIGURE = '?';") == 1
    assert "HEALTH_UNKNOWN = " not in html


def test_an_unknown_health_tile_says_which_kind_of_unknown_it_is(html):
    # "still scanning" and "couldn't load" decide whether waiting is worth
    # anything, so the tile must not collapse them into one caption.
    assert "timeout: 'still scanning'," in html
    assert "error: \"couldn't load\"," in html
    # The budgets tile inherits the sweep's phase verbatim rather than flattening
    # a timeout into a failure.
    assert "const budgetStatus = optData ? 'ready' : optState.phase;" in html


def test_every_health_tile_declares_the_status_of_its_source(html):
    # A tile without a status= defaults to 'ready' and would silently resume
    # publishing derived zeros, so all five must pass one.
    start = html.index('<div class="band-label">Health at a glance</div>')
    end = html.index("<!-- The HERO", start)
    band = html[start:end]
    assert band.count("<${HealthTile}") == 5
    assert band.count("status=$") == 5


def test_the_front_door_empty_card_requires_its_inputs_to_have_answered(html):
    # "No data yet. TokenJam is listening." is a claim about the user's history;
    # an outstanding or failed /cost, /status or /traces means "could not check",
    # not "nothing there".
    assert "const emptyKnown = !!costData && !!statusRead.data && !!tracesRead.data;" in html
    assert "const isEmpty = emptyKnown && !hasCost && !statusAgents.length && !traceList.length;" in html
    assert "${isEmpty ? html`" in html


def test_a_cost_failure_no_longer_blanks_the_panels_it_does_not_feed(html):
    # It used to replace the whole triage row with "Couldn't load triage", which
    # threw away five health tiles whose reads have nothing to do with /cost.
    dash = _dashboard(html)
    assert "Couldn't load triage" not in dash
    assert "Couldn't load spend for this window." in dash
    assert "everything else on this page is unaffected" in dash


# --- skeletons hold final positions ---------------------------------------- #
def test_the_kpi_row_holds_its_position_while_its_numbers_are_unknown(html):
    # The row used to not exist until /analytics answered, so the chart below it
    # jumped down the page tens of seconds after first paint.
    assert "const KPI_SKELETON_TILES = [0, 1, 2, 3];" in html
    assert "${!error && !kpis ? html`" in html
    assert '<div class="kpi-row" aria-hidden="true">' in html


def test_the_chart_placeholder_matches_the_chart_height(html):
    # 200px standing in for a 220px chart is a 20px jump on every first paint.
    assert 'loading && !resp ? html`<div class="shimmer" style="height:220px"></div>`' in html


def test_the_pricing_qualifier_holds_its_slot_while_unknown(html):
    # It appeared 27 seconds in and shoved both triage bands down the page.
    assert 'class="qualifier qualifier-skel"' in html


def test_no_panel_is_gated_on_a_page_wide_load_flag(html):
    # The single `d` state object is what coupled them; every panel now reads its
    # own useTriageRead result.
    dash = _dashboard(html)
    assert "const [d, setD] = useState(" not in dash
    assert "d.loading" not in dash
    assert "d.empty" not in dash


def test_the_anonymous_whole_band_shimmer_is_gone(html):
    # One unlabelled 90px grey rectangle stood in for BOTH bands until the
    # slowest read resolved: measured at 7+ minutes of a box naming nothing.
    assert 'd.loading ? html`<div class="shimmer" style="height:90px"></div>`' not in html


def test_only_one_analyzer_sweep_runs_at_a_time(html):
    # The 30s poll re-entered this read unconditionally; with a 45s response
    # cache in front of a multi-minute endpoint, every other poll opened another
    # concurrent sweep of the same DuckDB file.
    assert "if (optInFlight.current) return;" in html


def test_the_shimmer_primitive_honors_reduced_motion(html):
    block = html[html.index("@media (prefers-reduced-motion: reduce)"):]
    assert ".shimmer { animation: none;" in block[:400]
