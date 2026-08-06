"""Unit tests for the recoverable-total presentation contract in
`api/routes/cost.py` (A1, analyzer-audit #482).

`/cost/components` used to expose a flat `sum()` of every analyzer's
`past_overspend_usd` with nothing signaling that the analyzers price
waste from overlapping angles over the same sessions. These tests pin the
fix: individual analyzer estimates are NEVER touched (the operator does not
want the headline magnitude deflated), but the response now says, in the
data itself, that the sum is a ceiling rather than an achievable total, and
carries a standalone "largest single line" figure that IS honest on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime, timedelta, timezone

from tokenjam.api.routes.cost import (
    _collect_recoverable,
    _recoverable_basis_note,
    _recoverable_overlap_note,
    _recoverable_window_bounds,
)


@dataclass
class _FakeFinding:
    past_overspend_usd: float | None
    past_overspend_tokens: int | None
    estimate_basis: str = ""
    caveat: str = ""


@dataclass
class _FakeReport:
    downgrade: object | None
    findings: dict


def test_collect_recoverable_sorted_biggest_first():
    """Order must be deterministic and USD-descending so 'largest opportunity
    + N more' can render directly off list order, with no client-side sort."""
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(past_overspend_usd=5.0, past_overspend_tokens=100),
            "trim": _FakeFinding(past_overspend_usd=50.0, past_overspend_tokens=200),
            "reuse": _FakeFinding(past_overspend_usd=20.0, past_overspend_tokens=50),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "reuse", "cache"]
    assert [r["past_overspend_usd"] for r in out] == [50.0, 20.0, 5.0]


def test_collect_recoverable_ties_broken_by_tokens():
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(past_overspend_usd=10.0, past_overspend_tokens=100),
            "trim": _FakeFinding(past_overspend_usd=10.0, past_overspend_tokens=500),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "cache"]


def test_overlap_note_empty_for_zero_or_one_finding():
    # Zero findings: nothing to disclose, no false "these overlap" claim.
    assert _recoverable_overlap_note([]) == ""
    # A single finding cannot double-count anything by construction.
    assert _recoverable_overlap_note([{"past_overspend_usd": 10.0}]) == ""


def test_overlap_note_present_for_two_or_more_findings():
    note = _recoverable_overlap_note([
        {"past_overspend_usd": 10.0},
        {"past_overspend_usd": 5.0},
    ])
    assert note != ""
    assert "2" in note  # names how many estimates it's disclaiming
    # House style: never claim the analyzers' figures were changed here.
    assert "reduce" not in note.lower()
    # No em dashes in user-facing copy (house rule).
    assert "—" not in note


def test_overlap_note_scales_the_count_with_the_list():
    note = _recoverable_overlap_note([
        {"past_overspend_usd": 1.0},
        {"past_overspend_usd": 1.0},
        {"past_overspend_usd": 1.0},
    ])
    assert "3" in note


# --------------------------------------------------------------------------- #
# The spend bar's denominator: which window the ceiling was measured over.
#
# `total_recoverable_usd` is read out of the stored analyzer report, so it
# answers the SCAN's window, not the caller's. Drawing it as a shaded share of
# `total_cost_usd` (which does follow `since`) published two windows as one
# ratio. `_recoverable_window_bounds` is what lets the route compute a
# denominator over the ceiling's own window instead. The end-to-end coupling is
# pinned in tests/integration/test_cost_components_recoverable_population.py;
# these cover the bounds resolution itself, including the cases where it must
# REFUSE to answer.
# --------------------------------------------------------------------------- #
_NOW = datetime(2026, 8, 6, 21, 8, 53, tzinfo=timezone.utc)


def test_bounds_prefer_the_cycle_record_s_own_resolved_window():
    since, until = _recoverable_window_bounds({
        "scan_since": (_NOW - timedelta(days=30)).isoformat(),
        "scan_until": _NOW.isoformat(),
        "computed_at": (_NOW + timedelta(hours=4)).isoformat(),
        "window_days": 30,
    })
    # `computed_at` is when the pass FINISHED, which is later than the window it
    # observed. The sealed bounds win so the denominator is not silently shifted
    # by however long the scan took.
    assert until == _NOW
    assert since == _NOW - timedelta(days=30)


def test_bounds_fall_back_to_computed_at_minus_window_days():
    """A pre-record artifact still carries `computed_at` and `window_days`, and
    a denominator derived from those is far better than none."""
    since, until = _recoverable_window_bounds({
        "scan_since": None, "scan_until": None,
        "computed_at": _NOW.isoformat(), "window_days": 7,
    })
    assert until == _NOW
    assert since == _NOW - timedelta(days=7)


def test_bounds_are_never_invented():
    """THE BOUNDS ARE NOT GUESSED. A denominator over a window nobody recorded
    is the defect this helper exists to prevent, wearing a different hat: it
    would look exactly as authoritative as a correct one. `(None, None)` makes
    the route publish no share at all, which the UI renders as unknown."""
    assert _recoverable_window_bounds({}) == (None, None)
    # An anchor with no length, and a length with no anchor: neither is enough.
    assert _recoverable_window_bounds({"computed_at": _NOW.isoformat()}) == (None, None)
    assert _recoverable_window_bounds({"window_days": 30}) == (None, None)
    # A window_days of zero is not a window.
    assert _recoverable_window_bounds(
        {"computed_at": _NOW.isoformat(), "window_days": 0},
    ) == (None, None)
    # Unparseable values are treated as absent, never as "now".
    assert _recoverable_window_bounds(
        {"scan_since": "not-a-date", "scan_until": "not-a-date"},
    ) == (None, None)


def test_bounds_accept_epochs_and_naive_datetimes_as_utc():
    since, until = _recoverable_window_bounds({
        "scan_since": (_NOW - timedelta(days=30)).timestamp(),
        "scan_until": _NOW.replace(tzinfo=None),
    })
    assert until == _NOW
    assert since == _NOW - timedelta(days=30)


def test_basis_note_states_the_window_the_scope_and_what_the_picker_does():
    note = _recoverable_basis_note(_NOW - timedelta(days=30), _NOW, 30)
    # The window, as a length and as dates a reader can check.
    assert "30 days" in note
    assert "2026-07-07" in note and "2026-08-06" in note
    # The scope: every agent, not the one selected.
    assert "every agent" in note
    # That the picker above does not move this bar.
    assert "does not follow the range selected above" in note
    # And the persona sentence, which is the ONLY thing on screen saying the
    # denominator is not persona-scoped. Without it the bar asserts a
    # persona-specific share of spend that it cannot support.
    assert "persona picker changes which analyzers" in note
    assert "not which traffic is measured" in note
    # House rule: no em dashes in user-facing copy.
    assert "—" not in note


def test_basis_note_admits_an_unknown_window_rather_than_implying_one():
    note = _recoverable_basis_note(None, None, None)
    assert "did not record the window" in note
    assert "cannot be shown as a share of spend" in note
    # It must not name a window it does not have.
    assert "days," not in note
    assert "—" not in note
