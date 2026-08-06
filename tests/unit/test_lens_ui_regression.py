"""Static-grep regression guards for Lens web-UI fixes.

There is no JS test runner in the Python CI ``test`` job, so UI behaviour is
guarded here by asserting the buggy pattern is gone and the fix's markers are
present (the approach documented in CLAUDE.md → Web UI → "Testing the UI"), or by
extracting a pure function and running it under node (see
``test_lens_select_all_behaviour.py`` / ``test_lens_dashboard_states.py``).

Each assertion is anchored on the specific string a fix introduced or removed,
not on incidental wording, so harmless copy tweaks around it don't break it.

Guards the polish batch (#654–#657) and the trace-detail fix (#653 plus its #659
follow-ups: opt-in light payload, lazy per-span attributes, capped/pinned rows).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# #654 — Dashboard is a persistent top-level item + the default landing route.
# --------------------------------------------------------------------------- #
def test_dashboard_nav_link_is_persistent_across_lenses(html: str) -> None:
    # The Dashboard link must carry data-lens="all" so the improve/observe hide
    # rules never remove it, and must NOT be scoped to the improve lens anymore.
    assert (
        '<a href="#/dashboard" class="nav-link" data-view="dashboard" data-lens="all">'
        in html
    ), "Dashboard nav link must be data-lens=\"all\" (persistent in both lenses)"
    assert (
        '<a href="#/dashboard" class="nav-link" data-view="dashboard" data-lens="improve">'
        not in html
    ), "Dashboard must no longer be improve-only"


def test_dashboard_link_sits_above_the_lens_switch(html: str) -> None:
    dash = html.index('data-view="dashboard" data-lens="all"')
    switch = html.index('<div class="lens-switch"')
    assert dash < switch, "Dashboard nav link must be ABOVE the Improve/Observe toggle"


def test_persistent_lens_items_have_a_style_rule(html: str) -> None:
    assert '.sidebar a.nav-link[data-lens="all"]' in html


def test_empty_hash_default_route_is_dashboard(html: str) -> None:
    # getRoute()'s default (both the path fallback and the parts[0] fallback)
    # must resolve to dashboard, not review — with no render-time hash redirect.
    assert ": raw) || 'dashboard';" in html
    assert "parts[0] || 'dashboard'" in html
    assert "|| 'review';" not in html, "empty-hash default must be dashboard, not review"


def test_default_route_normalization_agrees_with_getroute(html: str) -> None:
    # Greptile P1-1: the URL-normalization replaceState on an empty hash must
    # write #/dashboard, matching getRoute()'s default. Writing the old #/review
    # here made the page (Dashboard) and address bar disagree, so a refresh or
    # shared bare link opened Review instead.
    assert "history.replaceState(null, '', '#/dashboard');" in html
    assert (
        "history.replaceState(null, '', '#/review');" not in html
    ), "empty-hash normalization must write #/dashboard, not #/review"


def test_dashboard_is_lens_neutral_and_preserves_active_lens(html: str) -> None:
    # Greptile P1-3: Dashboard is data-lens="all", so opening it from Observe
    # must keep the user in Observe. It must therefore be ABSENT from VIEW_LENS
    # (a mapped lens would force a switch), and the route-sync effect must fall
    # back to the sidebar's CURRENT lens for an unmapped view, not to 'improve'.
    assert (
        "dashboard: 'improve'" not in html
    ), "Dashboard must not be classified as improve in VIEW_LENS (it is lens-neutral)"
    # The lens-neutral fallback expression is still present for any future
    # persona that keeps both lenses; both currently-known personas (claude-code
    # and sdk) force 'improve' instead (guarded by
    # test_persona_forces_improve_lens below).
    assert (
        "(VIEW_LENS[view] || (sidebar && sidebar.dataset.lens) || 'improve')" in html
    ), "the lens-neutral fallback expression must remain"


def test_persona_forces_improve_lens(html: str) -> None:
    # Neither the claude-code nor the sdk persona has Observe pages or a
    # lens-switch, so both must be FORCED to 'improve'. Deriving the lens from a
    # lens-neutral view (Dashboard) would preserve a stale 'observe' left by a
    # prior session, collapsing the sidebar to Dashboard-only until a reload —
    # the persona-toggle collapse bug.
    assert "const lens = (persona === 'claude-code' || persona === 'sdk')" in html
    assert "? 'improve'" in html


def test_trace_detail_is_exempt_from_persona_redirect(html: str) -> None:
    # A trace DETAIL (traces/<id>) reached by drilling into a session's Traces
    # tab must NOT be redirected to the Dashboard the way the hidden top-level
    # Traces LIST is. The persona-redirect guard exempts a traces view that
    # carries a trace-id param.
    assert "!(route.view === 'traces' && route.param)" in html


# --------------------------------------------------------------------------- #
# #655 — one shared 30d window default across window-driven screens.
# --------------------------------------------------------------------------- #
def test_shared_default_since_constant_is_30d(html: str) -> None:
    assert "const DEFAULT_SINCE = '30d';" in html


def test_window_driven_screens_use_the_shared_default(html: str) -> None:
    # Traces (was 24h) and Cost (was 7d) now read the shared constant, and no
    # window-driven DEFAULTS object hardcodes a divergent since anymore.
    assert "const DEFAULTS = { since: DEFAULT_SINCE, result:" in html  # Traces
    assert "const DEFAULTS = { since: DEFAULT_SINCE, group_by: 'total'" in html  # Cost
    assert "const DEFAULTS = { since: DEFAULT_SINCE, agent_id: '', compare:" in html  # Optimize
    assert "const DEFAULTS = { since: DEFAULT_SINCE, metric: 'spend'" in html  # Analytics
    assert "const DEFAULTS = { since: '24h', result:" not in html
    assert "const DEFAULTS = { since: '7d', group_by:" not in html


def test_drill_through_hrefs_omit_since_at_the_shared_default(html: str) -> None:
    # Greptile P1-2: drill-through href builders must decide "omit since" against
    # the shared DEFAULT_SINCE, not a hardcoded literal. With the default moved
    # to 30d, tracesHrefForWindow's old `!== '24h'` dropped a real 24h window and
    # silently reset Traces to 30d. Both the Traces and Optimize href builders
    # now compare to DEFAULT_SINCE.
    assert (
        "if (since && since !== DEFAULT_SINCE) sp.set('since', since);" in html
    ), "drill-through builders must omit since only at DEFAULT_SINCE"
    assert (
        "if (since && since !== '24h') sp.set('since', since);" not in html
    ), "tracesHrefForWindow must not hardcode the retired 24h default"
    assert (
        "if (since && since !== '30d') sp.set('since', since);" not in html
    ), "href builders must read DEFAULT_SINCE, not a hardcoded 30d literal"


# --------------------------------------------------------------------------- #
# #656 — Review inbox appliable-count clarity + honest Apply wording.
# --------------------------------------------------------------------------- #
def test_inbox_reports_auto_appliable_count(html: str) -> None:
    assert "const autoApplyCount = bulkRelearn.length;" in html
    assert "const showBulkSelect = autoApplyCount > 1;" in html
    assert "can be applied automatically — the rest are copy-and-apply." in html


def test_bulk_select_only_shows_when_more_than_one_appliable(html: str) -> None:
    # Both the select-all label and the bulk action bar gate on showBulkSelect,
    # not on the old "> 0" which showed a bulk control for a single row.
    assert "${showBulkSelect ? html`" in html
    assert "${bulkRelearn.length > 0 ? html`" not in html


def test_apply_wording_says_next_run_not_enforcement(html: str) -> None:
    # The Approve note and the per-kind hints must make clear the write is
    # effective on the next run and is NOT live enforcement (honesty Rule 14).
    assert "it takes effect on the next run, not as live enforcement" in html
    assert "takes effect on the next run, not live" in html
    # The old ambiguous single-word "Apply change" button must be gone.
    assert "button: 'Apply change'" not in html


# --------------------------------------------------------------------------- #
# #657 — Drift empty-state leads with a scannable headline.
# --------------------------------------------------------------------------- #
def test_drift_empty_state_has_scannable_headline(html: str) -> None:
    assert 'class="drift-empty-lead"' in html
    assert ".drift-empty-lead {" in html
    assert (
        "Drift needs live SDK agents with 10+ completed sessions" in html
    ), "Drift empty-state must lead with the one-line headline"


# --------------------------------------------------------------------------- #
# #653 — large-trace detail must not hang; payload is capped + lazy-attrs
# --------------------------------------------------------------------------- #
def test_trace_detail_has_load_error_state(html: str) -> None:
    """The skeleton must be able to clear into an error state, never spin forever."""
    assert "loadState" in html
    # An explicit error branch with a retry affordance.
    assert "loadState === 'error'" in html
    assert "Retry" in html


def test_trace_detail_has_fetch_timeout(html: str) -> None:
    """The trace-detail fetch is raced against a timeout so it can't hang."""
    assert "Promise.race" in html
    assert "TIMEOUT_MS" in html


def test_trace_detail_handles_truncation(html: str) -> None:
    """A capped large trace must disclose 'showing N of M spans' (no silent drop)."""
    assert "truncated" in html
    assert "Showing " in html and "of " in html and "spans" in html


def test_trace_detail_fetches_attributes_lazily(html: str) -> None:
    """Captured content is fetched per-span on expand, not shipped for all spans."""
    # The lazy per-span endpoint is called from the detail view.
    assert "/spans/" in html
    assert "selAttrs" in html
    # The old bug: rendering sel.attributes straight from the waterfall payload.
    assert "JSON.stringify(sel.attributes" not in html


def test_trace_detail_caps_rendered_rows(html: str) -> None:
    """Thousands of DOM rows freeze the tab; the render is capped + disclosed."""
    assert "RENDER_ROW_CAP" in html


# --------------------------------------------------------------------------- #
# #659 P1-1 — the Lens waterfall must request the OPT-IN light payload so the
# default (full-attributes) response is left intact for exports / the API shim.
# --------------------------------------------------------------------------- #
def test_waterfall_fetch_uses_light_payload_param(html: str) -> None:
    """The waterfall fetch passes ?attributes=false; the default full payload is
    reserved for complete-span consumers (ApiBackend.get_trace_spans)."""
    assert "'/traces/' + traceId + '?attributes=false'" in html


# --------------------------------------------------------------------------- #
# #659 P1-3 — costliest "jump" badges must never target a row hidden by the
# render cap. Beyond-cap costliest spans are pinned into the rendered set, and
# the badge gates on the rendered-row id set so no badge is a dead link.
# --------------------------------------------------------------------------- #
def test_jump_badges_only_target_rendered_rows(html: str) -> None:
    """A jump badge must only render when its target row is actually rendered."""
    # The rendered-row id set exists and the badge gates on it.
    assert "renderedRowIds" in html
    assert "!renderedRowIds.has(sid)" in html
    # Beyond-cap costliest spans are pinned into the rendered set.
    assert "pinnedRows" in html


# --------------------------------------------------------------------------- #
# SDK-persona UX batch — the persona-empty banner renders in the CONTENT region
# below each page's header (never above it), the "switch back" copy is a real
# control that calls the persona setter, and the SDK sidebar is a clean flat
# list with no Improve/Observe lens toggle.
# --------------------------------------------------------------------------- #
def test_persona_empty_gate_renders_banner_below_the_header(html: str) -> None:
    """One shared gate wraps every primary view: it renders the page header
    THEN the banner in place of the body, so the banner never sits above (and
    mangles) the header. The gate must exist and be applied in App()'s view
    loop, and the old unconditional 'banner above the Dashboard header' render
    must be gone."""
    # The shared gate + its header component exist.
    assert "function PersonaEmptyGate(" in html
    assert "function PersonaEmptyHeader(" in html
    # App() wraps each mounted primary view in the gate, feeding it the header.
    assert "<${PersonaEmptyGate} persona=${persona} header=${html`<${PersonaEmptyHeader}" in html
    # The gate renders the header BEFORE the banner (content region below header).
    gate = html[html.index("function PersonaEmptyGate("):]
    gate = gate[: gate.index("function PersonaEmptyHeader(")]
    hdr = gate.index("${header")
    banner = gate.index("PersonaNoDataNotice")
    assert hdr < banner, "gate must render the header above the banner"
    # The old placement — banner rendered as the FIRST child of DashboardView's
    # own return, above its .ov-head header — must be gone.
    assert "<${PersonaNoDataNotice} persona=${persona} />\n    <div class=\"ov-head\">" not in html


def test_switch_back_is_a_clickable_control_calling_the_persona_setter(html: str) -> None:
    """'Switch back to <other>' is a button that calls the SAME persona setter
    the <select> uses (ctx.onChange), toggling to the other persona — not a
    page reload."""
    notice = html[html.index("function PersonaNoDataNotice("):]
    notice = notice[: notice.index("function PersonaEmptyGate(")]
    assert 'class="link-inline"' in notice
    assert "ctx.onChange && ctx.onChange(otherKey)" in notice
    assert "Switch back to ${other}" in notice
    # No reload / navigation to accomplish the switch.
    assert "location.reload" not in notice


def test_sdk_sidebar_differs_from_claude_code_only_where_sdk_has_a_lever(
    html: str,
) -> None:
    """SDK and claude-code share one flat Improve nav; SDK additionally keeps
    the surfaces that only make sense for a deployed service.

    THIS TEST USED TO ENFORCE THE DEFECT. It asserted the two personas' hidden
    lists were byte-for-byte identical — which pinned `traces`, `cost`,
    `alerts`, `drift` and `budget` as hidden under EVERY persona key, i.e. as
    views no user of any kind could reach. The five screens were built, routed
    and populated the whole time; only the gate was wrong, and a green suite was
    defending it. The assertion is INVERTED rather than deleted: the shared
    parts stay pinned as present, and the identical-lists state is now pinned as
    ABSENT so it cannot come back.
    """
    # Still shared: neither persona has a lens toggle, so both hide it.
    assert '.sidebar[data-persona="claude-code"] .lens-switch { display: none; }' in html
    assert '.sidebar[data-persona="sdk"] .lens-switch { display: none; }' in html
    # THE BAD STATE, pinned absent: the two persona keys must not carry the same
    # list, and the observe suite must not be hidden wholesale for SDK.
    assert "'sdk': ['traces', 'cost', 'alerts', 'drift', 'budget']," not in html, (
        "SDK must not hide the same five views claude-code does — that left no "
        "persona able to reach any of them"
    )
    # The correct state, pinned present. Spend is the only view hidden for BOTH,
    # and it carries a recorded reason (see the deliberate-hide test below).
    assert "'claude-code': ['traces', 'cost']," in html
    assert "'sdk': ['cost']," in html
    # Traces is SDK-only, hidden for claude-code by its own per-view rule rather
    # than by a blanket lens hide (per-session traces stay reachable for every
    # persona from the session detail's Traces tab).
    assert (
        '.sidebar[data-persona="claude-code"] a.nav-link[data-view="traces"] '
        '{ display: none !important; }' in html
    )
    # Alerts / Drift are no longer top-level nav entries at all — they moved
    # into the Sessions screen's SDK-services zone.
    assert '<a href="#/alerts" class="nav-link" data-view="alerts"' not in html
    assert '<a href="#/drift" class="nav-link" data-view="drift"' not in html
    # The predecessor's SDK-specific forcing of observe links visible, and its
    # hiding of Review inbox + Sessions, must still be gone.
    assert 'a.nav-link[data-lens="observe"] { display: flex !important; }' not in html
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="review"]' not in html
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="sessions"]' not in html
    # The lens is forced to 'improve' for both personas so the flat nav renders
    # regardless of stale lens state.
    assert "(persona === 'claude-code' || persona === 'sdk')" in html


# --------------------------------------------------------------------------- #
# Optimize IA redesign (PR #665) — summary landing + per-analyzer detail
# sub-pages, wired to the submenu and the Dashboard tiles.
# --------------------------------------------------------------------------- #
def test_optimize_submenu_has_no_new_badges(html: str) -> None:
    # The "NEW" badge is removed from the Summarize and Rules nav-children.
    for child in ("summarize", "rules"):
        # Grab the anchor line for the child and assert no nav-badge on it.
        marker = f'data-param="{child}" data-optimize-static="1"'
        assert marker in html, f"{child} nav-child must be tagged data-optimize-static"
    # No nav-badge "new" anywhere in the Optimize children.
    assert '<span class="nav-badge">new</span>' not in html, (
        "the NEW badge must be removed from the Optimize nav-children"
    )


def test_optimize_analyzer_children_are_injected_dynamically(html: str) -> None:
    # The submenu is populated from the findings, not a hardcoded per-analyzer
    # list of static anchors. The injector reads optimizeDetailAnalyzers off the
    # stored /optimize artifact and inserts one nav-child per finding before the
    # static Summarize/Rules children.
    assert "function optimizeDetailAnalyzers(opt)" in html
    assert 'a.nav-child[data-optimize-dyn]' in html, "dynamic children must be tagged for reconciliation"
    assert "optimizeDetailAnalyzers(d)" in html
    assert "a.dataset.optimizeDyn = '1'" in html
    # The submenu is PERSONA-AWARE (#671): the effect re-derives when the
    # selected persona changes, and a no-data persona yields an empty submenu.
    assert "}, [persona, personaHasNoData]);" in html, (
        "the submenu effect must depend on the selected persona so it re-derives on toggle"
    )
    assert "personaHasNoData" in html
    # The static Summarize/Rules children are gated (not injected
    # unconditionally) via a data-optimize-hidden flag syncNavState honors.
    assert "optimizeHidden" in html
    # Rules stays a submenu item (cross-cutting rule-write surface).
    assert 'href="#/optimize/rules"' in html


def test_optimize_detail_route_renders_a_single_analyzer(html: str) -> None:
    # OptimizeView takes navParam (the path segment after #/optimize/) and, when
    # present, renders only that analyzer's OptimizeFinding detail card plus a
    # back link — not the whole stacked page.
    # `persona` joins the signature: the view reads the report AS the selected
    # persona rather than as whichever one the corpus happens to be dominated by.
    assert "function OptimizeView({ params, navParam, persona })" in html
    assert "const detailName = navParam || null;" in html
    assert "if (detailName) {" in html
    assert "← Back to Optimize" in html
    # App threads the active path param into the view.
    assert "const navParam = isActive ? route.param : null;" in html
    assert "navParam=${navParam}" in html


def test_optimize_summary_no_longer_stacks_every_finding(html: str) -> None:
    # The old long page mapped OptimizeFinding over every analyzer on the summary.
    # That stacked render is gone; the summary is now ONE unified "Recommended
    # actions" list (each row a hint + figure + Review button), which replaced the
    # duplicated waste-list + separate Rules/Summarize/Findings sections.
    assert "${order.map(n => html`<${OptimizeFinding}" not in html, (
        "summary landing must not stack every analyzer's full detail section"
    )
    assert 'id="opt-actions"' in html
    assert 'class="opt-act-list"' in html
    # Each actionable row carries a Review button that opens its detail page.
    assert "class=\"opt-review-btn\"" in html
    # The old three-surface duplication is gone.
    assert 'id="opt-findings"' not in html
    assert 'id="opt-rules"' not in html


def test_dashboard_empty_tiles_are_not_clickable(html: str) -> None:
    # Only a tile with a real finding is an <a> to its detail page; empty-state /
    # at-ceiling tiles render as a non-clickable <div> (.static).
    assert "const hasPage = t.state === 'actionable' && DETAIL_ANALYZER_NAMES.has(t.name);" in html
    assert "const href = '#/optimize/' + t.name;" in html
    assert "? html`<a class=${cls} href=${href}>${inner}" in html
    # data tiles carry a persistent "→" cue inside the link; empty tiles do not
    assert '<span class="rec-go" aria-hidden="true">→</span></a>' in html
    assert "html`<div class=${cls}>${inner}</div>`" in html
    # The .static class strips the clickable affordance (no hover border, default
    # cursor) so an empty tile cannot read as a dead link.
    assert ".rec-tile.static { cursor: default; }" in html
    assert ".rec-tile.static:hover { border-color: var(--border); }" in html


# --------------------------------------------------------------------------- #
# Lens table horizontal-overflow fix — wide .opt-table findings tables (long
# absolute paths, provider/model strings) were pushing the whole page into
# horizontal scroll instead of scrolling inside their own container.
# --------------------------------------------------------------------------- #
def test_body_never_scrolls_horizontally(html: str) -> None:
    # Belt-and-suspenders: no descendant, however wide, may push the PAGE
    # itself into horizontal scroll. Wide content must scroll inside its own
    # .table-wrap instead.
    assert "overflow-x: hidden;" in html
    body = html[html.index("\nbody {"):]
    body = body[: body.index("}")]
    assert "overflow-x: hidden;" in body, "body rule must set overflow-x: hidden"


def test_every_opt_table_is_wrapped_for_horizontal_scroll(html: str) -> None:
    # Every OptimizeFinding detail table (.opt-table) must sit inside a
    # .table-wrap (overflow-x: auto) container so a long unbreakable string
    # (a repo-relative path, a provider/model id) scrolls inside the table
    # instead of forcing the whole card, and the page, wider than the
    # viewport. A bare, unwrapped `<table class="opt-table"` is the bug.
    import re

    for m in re.finditer(r'<table class="opt-table"', html):
        preceding = html[max(0, m.start() - 40): m.start()]
        assert '<div class="table-wrap">' in preceding, (
            f"unwrapped .opt-table at offset {m.start()}: {preceding!r}"
        )


def test_recurring_inclusions_label_is_truncated_with_full_path_in_title(html: str) -> None:
    # The "What's re-included" column in the resend finding's recurring-
    # inclusions table renders absolute paths. Rendering them untruncated
    # forced the table (and the page) wider than the viewport, with the
    # label unreadable on both ends. shortPath() truncates to the last two
    # path segments for display; the full path still rides in title= for a
    # hover tooltip.
    assert (
        '<td class="mono" title=${r.label}>${shortPath(r.label)}</td>' in html
    ), "recurring-inclusions label must be shortPath()-truncated with the full value in title="


def test_cursor_listbox_scrolls_horizontally_and_truncates_paths(html: str) -> None:
    # The Summarize/Rules file pickers (.cur-listbox > .cur-table) render a
    # File column of absolute paths. The listbox must scroll horizontally on
    # its own (overflow-x, alongside its existing overflow-y) instead of
    # relying on the page to scroll, and the path itself must be
    # shortPath()-truncated with the full path in a hover title.
    assert (
        ".cur-listbox { max-height:380px; overflow-y:auto; overflow-x:auto;" in html
    ), "cur-listbox must scroll horizontally, not just vertically"
    assert '<td class="mono" title=${c.path}>${shortPath(c.path)}</td>' in html
    assert "title=${r.path + ' — review the diff'}" in html


# --------------------------------------------------------------------------- #
# Total opportunity tile — a 7th tile, first in the Dashboard's "Opportunities
# to optimize token efficiency" row, summing the six per-analyzer figures.
#
# `totalOpportunityFigure()` is the one pure function that decides the sum and
# its population; a static string match on the source would still pass if that
# arithmetic or exclusion logic were wrong (the exact critique
# test_lens_select_all_behaviour.py levels at grep-only tests), so it is
# extracted straight out of the served index.html and run under node instead,
# the same trick that module and test_lens_dashboard_states.py use.
# --------------------------------------------------------------------------- #
_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available for JS evaluation")


def _round_to_cents_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function roundToCents(n)")
    end = src.index("\n}\n", start) + 2
    return src[start:end]


def _total_figure_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function totalOpportunityFigure")
    end = src.index("// The TOTAL tile itself", start)
    # totalOpportunityFigure calls roundToCents (the same helper fmtDashUsd
    # uses) to round each contributor before summing -- pull it in too so
    # this extraction doesn't drift from the real dependency.
    return _round_to_cents_source() + "\n" + src[start:end]


def _total_figure(tiles: list[dict]):
    script = (
        _total_figure_source()
        + "\nconsole.log(JSON.stringify(totalOpportunityFigure(" + json.dumps(tiles) + ")));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


@_node
def test_total_equals_the_plain_sum_of_actionable_contributors():
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 1528.89, "tokens": 100},
        {"name": "resend", "state": "actionable", "usd": 374.33, "tokens": 200},
        {"name": "downsize", "state": "actionable", "usd": 295.23, "tokens": 300},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "populated"
    assert round(fig["totalUsd"], 2) == round(1528.89 + 374.33 + 295.23, 2)
    assert fig["totalTokens"] == 600
    assert fig["contributorCount"] == 3


@_node
def test_total_reconciles_with_the_sum_of_the_displayed_per_tile_figures():
    # The total must equal what you get from adding the SIX RENDERED figures
    # by hand, not a sum of raw values rounded once at the end -- those two
    # differ whenever contributors' raw cents round independently. Fixture
    # chosen so the two strategies disagree by a whole cent, unaffected by
    # any binary floating-point boundary ambiguity (verified directly: raw
    # sum 3.012 rounds once to 3.01, but each 1.004 individually rounds down
    # to 1.00, so the sum of the three DISPLAYED figures is 3.00).
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 1.004, "tokens": 1},
        {"name": "resend", "state": "actionable", "usd": 1.004, "tokens": 1},
        {"name": "downsize", "state": "actionable", "usd": 1.004, "tokens": 1},
    ]
    fig = _total_figure(tiles)
    # The sum of the values AS DISPLAYED (each tile renders $1.00 via
    # fmtDashUsd's 2dp rounding) is $3.00 -- not the sum-then-round-once
    # figure of $3.01 a naive raw sum would produce.
    assert fig["totalUsd"] == 3.00
    assert fig["totalUsd"] != round(1.004 + 1.004 + 1.004, 2)  # != 3.01


@_node
def test_an_absent_no_findings_analyzer_is_excluded_not_zeroed():
    # Deadweight with no candidates renders 'No candidates', never a $0 tile
    # (root CLAUDE.md anti-pattern 22). Its state carries no usd at all here,
    # mirroring classifyFinding()'s real 'no_findings' shape, and the sum must
    # come out identical to the same row with that tile removed entirely --
    # proof it is excluded structurally (by state), not by a falsy usd check.
    with_deadweight = [
        {"name": "subagent", "state": "actionable", "usd": 100.0, "tokens": 10},
        {"name": "deadweight", "state": "no_findings"},
    ]
    without_deadweight = [
        {"name": "subagent", "state": "actionable", "usd": 100.0, "tokens": 10},
    ]
    with_fig = _total_figure(with_deadweight)
    without_fig = _total_figure(without_deadweight)
    assert with_fig["totalUsd"] == without_fig["totalUsd"] == 100.0
    assert with_fig["contributorCount"] == without_fig["contributorCount"] == 1
    # The excluded tile is still counted as a KNOWN (resolved) tile, just not
    # a contributor -- it answered "nothing here", which is not the same as
    # "not yet known".
    assert with_fig["knownCount"] == 2


@_node
def test_an_at_ceiling_tile_contributes_nothing_to_the_sum():
    # cache's positive "already at the ceiling" state carries a metric string,
    # never a usd figure; it must be excluded the same way no_findings is.
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 50.0, "tokens": 5},
        {"name": "cache", "state": "at_ceiling", "metric": "98% cache efficacy"},
    ]
    fig = _total_figure(tiles)
    assert fig["totalUsd"] == 50.0
    assert fig["contributorCount"] == 1


@_node
def test_all_analyzers_unresolved_is_the_unknown_state_never_zero():
    # Nothing has resolved yet -- the tile must render a skeleton, not a $0.00
    # total (the worst possible placeholder: it reads as "no waste").
    tiles = [
        {"name": "subagent", "state": "not_ready", "hint": "Not run on Overview."},
        {"name": "resend", "state": "not_ready", "hint": "Not run on Overview."},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "unknown"
    assert fig["totalUsd"] == 0
    assert fig["knownCount"] == 0


@_node
def test_every_analyzer_resolved_empty_is_the_empty_state():
    # Every tile answered, none had a recoverable figure: this is the ONE
    # legitimate home for empty-state copy, distinct from 'unknown'.
    tiles = [
        {"name": "subagent", "state": "no_findings"},
        {"name": "deadweight", "state": "no_findings"},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "empty"
    assert fig["totalUsd"] == 0
    assert fig["contributorCount"] == 0
    assert fig["knownCount"] == 2


@_node
def test_a_partially_resolved_row_discloses_its_coverage_not_a_full_claim():
    # Some analyzers answered, some have not: the total must not claim to
    # cover every tile in the row. unresolvedCount is how the renderer knows
    # to disclose the partial population instead of publishing a total that
    # reads as complete.
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 10.0, "tokens": 1},
        {"name": "resend", "state": "not_ready", "hint": "Not run on Overview."},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "populated"
    assert fig["totalUsd"] == 10.0
    assert fig["unresolvedCount"] == 1
    assert fig["totalCount"] == 2
    assert fig["knownCount"] == 1


def test_total_tile_is_first_in_the_row_and_visually_distinct(html: str) -> None:
    # Rendered before tiles.map(), so it is the first child of .tile-grid; a
    # dedicated CSS class carries the weight/border distinction (never the
    # accent colour, which means "typeable/clickable" and this tile links
    # nowhere). The per-tile share bar (lens redesign) wraps the map in an
    # IIFE to compute the rank/tone ramp once, so anchor on render ORDER
    # rather than a single unbroken substring.
    #
    # `fig` (not `tiles`) is what the tile now takes: it is the SAME netted,
    # server-computed rollup figure the Dashboard hero above this row reads,
    # rather than a client-side sum of the six sibling tiles' own figures —
    # see `rollupFigure()` / the comment above `totalOpportunityFigure()`.
    assert "<${TotalOpportunityTile} fig=${rollupFig} framing=${framing} />${" in html
    idx_total = html.index("<${TotalOpportunityTile} fig=${rollupFig} framing=${framing} />${")
    idx_map = html.index("return tiles.map(t => {", idx_total)
    assert idx_total < idx_map, "the total tile must render before tiles.map() in the tile-grid"
    assert ".rec-tile.total-tile" in html
    assert ".rec-tile.total-tile .rec-amount { color: var(--text); }" in html


def test_total_tile_cannot_render_a_dollar_figure_while_unresolved(html: str) -> None:
    # The 'unknown' branch returns before any amount/hint computation touches
    # fmtFramedSavings -- a skeleton, never a number, while nothing has
    # resolved. Anchor on the guard clause and its skeleton markup.
    fn = html[html.index("function TotalOpportunityTile("):]
    fn = fn[: fn.index("\n}\n")]
    assert "if (fig.state === 'unknown') {" in fn
    idx_guard = fn.index("if (fig.state === 'unknown')")
    idx_amount = fn.index("const amount")
    assert idx_guard < idx_amount, "the unresolved guard must return before computing an amount"
    assert 'class="rec-tile total-tile rec-skel" aria-hidden="true"' in fn


def test_total_tile_comment_marks_the_sum_as_deliberate_and_scoped(html: str) -> None:
    # The founder decision (naive sum now, netted rollup once `script` runs
    # for a persona that reaches this row) must be recorded at the summing
    # site, with no internal ticket id per root anti-pattern 11.
    fn_start = html.index("function totalOpportunityFigure")
    comment = html[html.index("// The TOTAL opportunity tile"): fn_start]
    assert "PLAIN SUM" in comment
    assert "netted cross-analyzer" in comment
    assert "persona-disabled" in comment
    import re
    assert not re.search(r"#\d+", comment), "no internal ticket id in a source comment"


def test_total_matches_the_displayed_per_tile_sum(html: str) -> None:
    # roundToCents backs fmtDashUsd's own rounding (a tile's displayed
    # figure) AND totalOpportunityFigure's summing step, from the same
    # helper -- never two independent rounding expressions that could drift
    # apart. Anchors both call sites plus the shared helper's definition.
    assert "function roundToCents(n) {" in html
    assert "return '$' + roundToCents(n).toLocaleString(" in html  # fmtDashUsd
    assert (
        "const totalUsd = roundToCents(contributors.reduce((sum, t) => sum + roundToCents(t.usd), 0));"
        in html
    )


# --------------------------------------------------------------------------- #
# `rollupFigure()` — what the Total opportunity tile AND the Dashboard hero
# now render instead of `totalOpportunityFigure(tiles)`'s plain sum. This is
# the JS-level pin for the netting fix: the figure must come straight off the
# wire's `past_overspend_usd` (GET /relearn/cost-proposals, already netted
# server-side via `_net_cross_analyzer_session_overlap`), never re-derived by
# adding per-analyzer figures client-side — which is exactly what would
# double-count once `reuse` and `script` (both cluster on the identical
# repeated-tool-sequence shape) are enabled together for one persona (`sdk`).
# --------------------------------------------------------------------------- #
def _rollup_figure_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function rollupFigure(read)")
    end = src.index("\n}\n", start) + 2
    return src[start:end]


def _rollup_figure(read: dict):
    script = (
        _rollup_figure_source()
        + "\nconsole.log(JSON.stringify(rollupFigure(" + json.dumps(read) + ")));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


@_node
def test_rollup_figure_is_unknown_before_any_complete_read_lands():
    assert _rollup_figure({"phase": "loading", "data": None})["state"] == "unknown"


@_node
def test_rollup_figure_is_unknown_for_a_store_that_never_computed():
    # A 'never_run' payload answers the HTTP request but never set
    # `computed_at` -- it must not read as a measured empty answer (root
    # CLAUDE.md anti-pattern 22: an un-run scan is not an all-clear).
    data = {"status": "never_run", "computed_at": None, "past_overspend": {}}
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "unknown"
    assert fig["totalUsd"] == 0


@_node
def test_rollup_figure_reads_the_wire_total_verbatim_never_a_client_side_sum():
    # The whole point of the fix: `totalUsd` is `past_overspend_usd` as
    # published, never re-summed from `by_analyzer`. Chosen so a naive
    # client-side sum of the two contributors (50.0 + 73.45 = 123.45) would
    # happen to match here on its own -- the assertion that matters is the
    # LAST one, proving this function never performs that addition at all.
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 123.45,
            "past_overspend_tokens": 999,
            "proposal_count": 3,
            "token_proposal_count": 3,
            "deduplicated_proposal_count": 4,
            "by_analyzer": [
                {"analyzer": "reuse", "usd": 50.0, "tokens": 400, "count": 1},
                {"analyzer": "script", "usd": 73.45, "tokens": 599, "count": 2},
            ],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "populated"
    assert fig["totalUsd"] == 123.45
    assert fig["totalTokens"] == 999
    assert fig["dedupedCount"] == 4
    assert fig["byAnalyzer"] == data["past_overspend"]["by_analyzer"]


@_node
def test_rollup_figure_double_counted_overlap_would_be_visible_if_ever_reintroduced():
    # Guards against a regression back to the old naive-sum shape: if
    # `rollupFigure` ever started re-summing `by_analyzer` instead of trusting
    # the server's own `past_overspend_usd`, this fixture (reuse and script
    # both claiming the SAME 20 sessions before netting, netted total well
    # below their raw sum) would silently start reporting the bigger, wrong
    # number. Mirrors the overlap shape
    # test_dashboard_hero_netted_rollup.py proves on the real wire.
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 60.0,   # the server's netted figure
            "past_overspend_tokens": 500,
            "proposal_count": 2,
            "token_proposal_count": 2,
            "deduplicated_proposal_count": 2,
            "by_analyzer": [
                # Raw, pre-netting figures a naive client-side sum would add
                # to 100.0 -- strictly more than the netted total above.
                {"analyzer": "reuse", "usd": 50.0, "tokens": 300, "count": 1},
                {"analyzer": "script", "usd": 50.0, "tokens": 300, "count": 1},
            ],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    naive_sum = sum(a["usd"] for a in data["past_overspend"]["by_analyzer"])
    assert fig["totalUsd"] == 60.0
    assert fig["totalUsd"] < naive_sum


@_node
def test_rollup_figure_is_empty_only_once_a_completed_pass_found_nothing():
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 0, "past_overspend_tokens": 0,
            "proposal_count": 0, "token_proposal_count": 0,
            "deduplicated_proposal_count": 0, "by_analyzer": [],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "empty"
    assert fig["totalUsd"] == 0


def test_opportunities_row_fits_seven_tiles_without_widening_the_shared_compact_grid(html: str) -> None:
    # The Total tile makes this a 7-tile row (was 6), which orphaned the 7th
    # (Deadweight) onto its own row at the shared .tile-grid.compact minmax.
    # .opp-grid narrows just this row's minmax so all seven fit across at
    # the normal content width; it must NOT touch the shared .compact rule
    # (the health-glance row and this row's own loading skeleton also use
    # it, and don't need narrowing), and both the answered-tiles grid and
    # its loading skeleton must carry the class so neither reflows against
    # the other when the real data lands.
    assert ".tile-grid.compact.opp-grid { grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); }" in html
    assert '.tile-grid.compact { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));' in html
    assert 'class="tile-grid compact opp-grid"' in html
    # Both call sites carry it: the scanning skeleton and the answered tiles.
    assert html.count('class="tile-grid compact opp-grid"') == 2
    # The health-glance row is a separate grid and must NOT be narrowed.
    health = html[html.index('<div class="section-band">Health at a glance</div>'):]
    health = health[: health.index("<!-- The HERO")]
    assert 'class="tile-grid compact"' in health
    assert "opp-grid" not in health


def test_skeleton_tile_count_matches_the_seven_real_tiles(html: str) -> None:
    # REC_SKELETON_TILES stood in for the row before the Total tile existed
    # (6 placeholders for 6 real tiles). Left at 6 it would render one fewer
    # skeleton box than the 7 real tiles that land, a visible reflow.
    assert "const REC_SKELETON_TILES = [0, 1, 2, 3, 4, 5, 6];" in html
    assert "const REC_SKELETON_TILES = [0, 1, 2, 3, 4, 5];" not in html


# --------------------------------------------------------------------------- #
# Persona VIEW gate — which pages each persona is offered.
#
# The defect this section guards against is structural rather than cosmetic: a
# view listed under EVERY persona key is reachable by nobody, and it reads
# exactly like a view that was deliberately shelved. Five built, routed and
# populated screens sat in that state, and a green suite was defending it.
# --------------------------------------------------------------------------- #
def _persona_hidden_views(html: str) -> dict[str, list[str]]:
    """Parse ``PERSONA_HIDDEN_VIEWS`` out of the UI source.

    Parsed rather than re-declared: a test carrying its own copy of the map
    passes while the map says something else, which is the whole failure mode
    here. The literal is plain arrays of single-quoted names by construction
    (there is a separate guard forbidding a Set literal), so a small regex is
    enough and a shape change fails loudly instead of silently matching nothing.
    """
    body = html[html.index("const PERSONA_HIDDEN_VIEWS = {"):]
    body = body[: body.index("};")]
    out: dict[str, list[str]] = {}
    for line in body.splitlines():
        m = re.match(r"\s*'([a-z-]+)'\s*:\s*\[(.*)\]\s*,", line)
        if m:
            out[m.group(1)] = re.findall(r"'([a-z-]+)'", m.group(2))
    assert out, "PERSONA_HIDDEN_VIEWS must parse — its literal shape changed"
    return out


def _persona_hidden_deliberate(html: str) -> set[str]:
    body = html[html.index("const PERSONA_HIDDEN_DELIBERATE = {"):]
    body = body[: body.index("};")]
    return set(re.findall(r"^\s{2}([a-z-]+):", body, re.M))


def test_no_view_is_hidden_for_every_persona_without_a_recorded_reason(
    html: str,
) -> None:
    """THE pin for the hidden-views defect.

    A view hidden under every persona key cannot be opened by anyone. That is a
    legitimate product decision (Spend is one), but it is indistinguishable from
    the bug where a list was copied onto a second persona by mistake — so a
    deliberate one must record WHY in PERSONA_HIDDEN_DELIBERATE. Anything hidden
    everywhere with no entry there fails here.
    """
    hidden = _persona_hidden_views(html)
    assert set(hidden) == {"claude-code", "sdk"}, (
        "a new persona key changes what 'hidden for everyone' means — update "
        "this guard deliberately"
    )
    hidden_everywhere = set.intersection(*(set(v) for v in hidden.values()))
    recorded = _persona_hidden_deliberate(html)
    assert hidden_everywhere <= recorded, (
        f"{sorted(hidden_everywhere - recorded)} is hidden for every persona "
        f"with no reason recorded in PERSONA_HIDDEN_DELIBERATE — either it is "
        f"deliberate and must say so, or the gate is wrong"
    )
    # And the deliberate list may not grow beyond what is actually hidden
    # everywhere: a stale entry would license a future blanket hide.
    assert recorded <= hidden_everywhere, (
        f"{sorted(recorded - hidden_everywhere)} records a deliberate blanket "
        f"hide for a view that is not hidden everywhere"
    )


def test_spend_is_the_deliberately_hidden_one_and_traces_is_sdk_only(
    html: str,
) -> None:
    """The decided target state, named explicitly rather than only set-wise."""
    hidden = _persona_hidden_views(html)
    # Spend: hidden for both, on purpose.
    assert "cost" in hidden["claude-code"] and "cost" in hidden["sdk"]
    assert "cost" in _persona_hidden_deliberate(html)
    # Traces: the standalone cross-session browser is SDK-only. Per-session
    # traces reach every persona through the session detail's Traces tab.
    assert "traces" in hidden["claude-code"]
    assert "traces" not in hidden["sdk"]
    # Alerts / drift / budget are not top-level views any more, so they are in
    # neither list — they are tabs of the Sessions screen instead.
    for view in ("alerts", "drift", "budget"):
        assert view not in hidden["claude-code"], view
        assert view not in hidden["sdk"], view
        assert view in html[html.index("const SESSIONS_SDK_TAB_VIEWS"):][:200], view


def test_alerts_drift_budget_are_sessions_tabs_not_top_level_views(
    html: str,
) -> None:
    """Relocated, not merely re-gated: their routes resolve to the Sessions
    page, they render inside its SDK-services zone, and they are no longer
    mounted as primary views of their own."""
    # primaryKeyFor sends all three to the Sessions screen.
    assert "if (SESSIONS_SDK_TAB_VIEWS.has(v)) return 'sessions';" in html
    # They are no longer keep-alive primary views (that would mount three panes
    # the router can never activate).
    for entry in ("['alerts',    AlertsView]", "['drift',     DriftView]",
                  "['budget',    BudgetView]"):
        assert entry not in html, f"{entry} must leave PRIMARY_VIEWS"
    # StatusView renders them, inside the SDK-only zone.
    assert "${sdkTab === 'alerts' ? html`<${AlertsView}" in html
    assert "${sdkTab === 'drift' ? html`<${DriftView}" in html
    assert "${sdkTab === 'budget' ? html`<${BudgetView}" in html
    # The Sessions nav row stays lit on any of them (comma-list data-view-alt).
    assert 'data-view="sessions" data-view-alt="alerts,drift,budget"' in html
    assert "(el.dataset.viewAlt || '').split(',')" in html


def test_persona_gate_hides_nothing_until_the_persona_is_known(html: str) -> None:
    """A not-yet-known persona must apply NO hiding rules.

    The old fallback resolved an unknown persona to a concrete one, so an
    unresolved read silently applied a real persona's hiding rules to a reader
    who might be the other persona. Both halves of the gate — the JS predicate
    and the CSS attribute it pairs with — now key on `known`.
    """
    assert "function personaHides(persona, view, known) {" in html
    assert "if (!known) return false;" in html
    # The CSS half: syncNavState leaves data-persona EMPTY until settled, so no
    # [data-persona="..."] rule can match.
    assert "const personaAttr = personaKnown ? persona : '';" in html
    assert "sidebar.dataset.persona = personaAttr;" in html
    # The old unconditional write must be gone.
    assert "sidebar.dataset.persona = persona;" not in html
    # An explicit user choice counts as settled even before the fetch lands.
    assert "const personaKnown = personaOverride != null || personaInfo.known;" in html


# --------------------------------------------------------------------------- #
# Persona ANALYZER gate — the selected persona reaches the server.
# --------------------------------------------------------------------------- #
def test_optimize_reads_are_scoped_to_the_selected_persona(html: str) -> None:
    """Every /optimize read names the persona the reader picked.

    The picker used to be pure client-side state that never left the browser, so
    the Optimize submenu, the analyzer cards, the persona-gated chip and the
    Dashboard tiles all keyed off the STORED report's own dominant persona.
    """
    # OptimizeView, the Dashboard band, and App()'s submenu effect.
    assert "api('/optimize', { since, agent_id: agentId || undefined, persona })" in html
    assert "api('/optimize', { since, fast: 'true', persona })" in html
    assert "api('/optimize', { fast: 'true', persona })" in html
    # Persona is a real refetch dependency in both readers.
    assert "}, [since, agentId, compare, persona]);" in html
    assert "}, [since, persona, armOptWait]);" in html


def test_the_blank_the_submenu_on_mismatch_workaround_is_gone(html: str) -> None:
    """The workaround could only BLANK the submenu when the stored report's
    persona differed from the selection — it had no way to compute the right
    entries, so a machine with data for both personas showed nothing at all for
    the non-dominant one. Threading the persona to the server replaces it."""
    assert "if (d.persona && d.persona !== persona) {" not in html
    assert "(thread the selected persona to /optimize) is deferred" not in html
    assert "is deferred." not in html


def test_a_persona_switch_does_not_keep_the_previous_personas_figures(
    html: str,
) -> None:
    """Stale-but-shown is the right call for a refresh and the wrong call for a
    persona switch: the numbers on screen answer a different question, so the
    surface must go back to not-yet-known rather than relabel them."""
    assert "const personaChanged = optPersonaRef.current !== persona;" in html
    assert "data: personaChanged ? null : s.data" in html


def test_an_unanswered_analyzer_is_a_third_state_not_an_empty_result(
    html: str,
) -> None:
    """`persona_unanswered_analyzers` — a lever this persona HAS, that the
    stored pass never ran. It must render as unresolved, never as "No
    candidates", which would be a clean bill of health nothing measured."""
    assert "opt.persona_unanswered_analyzers" in html
    assert "st.opt.persona_unanswered_analyzers" in html
    assert "const PERSONA_UNANSWERED_HINT =" in html
    # It resolves to the not-ready tile state, which is already excluded from
    # every published total (see totalOpportunityFigure).
    assert "{ name: k, state: 'not_ready', hint: PERSONA_UNANSWERED_HINT }" in html
    # And it gets its own detail-page branch, checked before the generic
    # "ran, found nothing" one.
    assert "if (personaUnanswered.has(detailName)) {" in html


def test_sessions_page_filters_both_zones_on_the_selected_persona(
    html: str,
) -> None:
    """The coding-session list was not persona-filtered at all: switching to
    "SDK workflows" left a screenful of Claude Code cards under an SDK heading.
    Both zones now gate, and neither gates on a not-yet-known persona."""
    assert "const showSdkZone = !personaKnown || persona !== 'claude-code';" in html
    assert "const showCodingZone = !personaKnown || persona !== 'sdk';" in html
    assert "function StatusView({ params, persona, personaKnown, routeView })" in html
