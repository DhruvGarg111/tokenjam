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
    """The two pure deciders, lifted straight out of the served page."""
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function recoverableBandState")
    end = src.index("function HealthTile", start)
    return src[start:end]


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


def test_the_slow_sweep_is_not_inside_the_triage_batch(html):
    # A Promise.all resolves on its slowest member, so folding a minutes-long
    # analyzer sweep in there held the whole band (both columns) hostage.
    dash = html.index("function DashboardView")
    start = html.index("const load = useCallback(async () => {", dash)
    end = html.index("// ---- The recoverable-waste read", start)
    assert "'/optimize'" not in html[start:end]


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
    # a failed /status or /traces means "could not check", not "nothing there".
    assert (
        "const empty = !hasCost && status.ok && !agents.length "
        "&& traces.ok && !traceList.length;"
    ) in html


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
