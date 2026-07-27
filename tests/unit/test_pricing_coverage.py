"""An unpriced model must be VISIBLE in the cost views, not just in a log line.

When `get_rates` finds no row, `calculate_cost` prices the span at a flat
default rate and logs one warning per process. The dollar figure that reaches
the dashboard is indistinguishable from a real one — which is how a benchmark
replay could be wrong by 5-30x for most of its models while every surface
looked fine. The stored `pricing_source` column already records the provenance;
this pins that the cost views actually read it.
"""

from __future__ import annotations

import pytest

from tokenjam.core.pricing_coverage import (
    PricingCoverage,
    coverage_note,
    summarize_pricing_coverage,
)


class _FakeConn:
    """Minimal stand-in returning one canned row for the coverage query."""

    def __init__(self, row):
        self._row = row
        self.executed: list[tuple[str, list]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params or []))
        return self

    def fetchall(self):
        return self._row


def test_no_conn_reports_nothing_rather_than_a_clean_bill():
    cov = summarize_pricing_coverage(None, None, None, None)
    assert cov.measured is False
    assert cov.unpriced_call_count == 0
    assert coverage_note(cov) is None


def test_a_fully_priced_window_produces_no_note():
    conn = _FakeConn([])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.measured is True
    assert cov.unpriced_models == ()
    assert cov.unpriced_call_count == 0
    assert coverage_note(cov) is None


def test_unpriced_models_are_named_with_their_call_share():
    conn = _FakeConn([("anthropic", "claude-mystery-9", 120, 3.5)])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.measured is True
    assert cov.unpriced_call_count == 120
    assert cov.unpriced_models == (("anthropic", "claude-mystery-9", 120),)
    note = coverage_note(cov)
    assert note is not None
    assert "claude-mystery-9" in note
    # The claim the user needs: the number is a default-rate estimate, not a
    # quoted price. It must not read as a $0 or as a confirmed figure.
    assert "default rate" in note
    assert "$0" not in note


def test_the_note_reports_every_unpriced_model_it_was_given():
    conn = _FakeConn([
        ("anthropic", "model-a", 10, 1.0),
        ("openai", "model-b", 5, 0.5),
    ])
    cov = summarize_pricing_coverage(conn, None, None, None)
    assert cov.unpriced_call_count == 15
    note = coverage_note(cov)
    assert note is not None
    assert "model-a" in note and "model-b" in note


def test_filters_are_parameterised_never_interpolated():
    conn = _FakeConn([])
    summarize_pricing_coverage(conn, "agent-1", None, None)
    sql, params = conn.executed[0]
    assert "agent-1" not in sql
    assert "agent-1" in params


def test_a_dataclass_instance_is_immutable():
    cov = PricingCoverage(measured=True, unpriced_models=(), unpriced_call_count=0,
                          unpriced_cost_usd=0.0)
    with pytest.raises(Exception):
        cov.measured = False  # type: ignore[misc]
