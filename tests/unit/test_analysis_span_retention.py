"""One span drives retention, and retention can never undercut it.

There were two settings free to disagree — how much history the store KEPT and
how much the analyzers LOOKED AT — and on a real store they did: roughly eight
weeks of the oldest history was deleted over two days while the analyzers went
on sizing their window against it, and the only way to see it was to measure the
store twice, days apart, and diff. The tests here pin the coupling that makes
that unrepresentable, plus the ledger that makes any deletion observable after
the fact rather than only by that same diff.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.analysis_span import (
    analysis_span_days,
    parse_analysis_span,
    resolved_window_days,
    retention_days_for,
    retention_was_raised_to_span,
    span_label,
)
from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.retention import run_retention_cleanup
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


# --- the derivation ---------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("30d", 30), ("90d", 90), ("30", 30), ("90", 90),
    ("all", None), ("all-available", None), ("ALL", None),
])
def test_a_span_choice_parses_to_days_or_unbounded(raw, expected):
    assert parse_analysis_span(raw) == expected


@pytest.mark.parametrize("raw", ["banana", "", "0d", "-5d", None])
def test_an_unparseable_span_raises_rather_than_defaulting(raw):
    """Silently defaulting would make the product analyze a span nobody wrote."""
    with pytest.raises(ValueError):
        parse_analysis_span(raw)


def test_retention_is_derived_from_the_span():
    assert retention_days_for(StorageConfig(analysis_span="30d")) == 30
    assert retention_days_for(StorageConfig(analysis_span="90d")) == 90


def test_an_all_available_span_disables_retention_entirely():
    storage = StorageConfig(analysis_span="all")
    assert analysis_span_days(storage) is None
    assert retention_days_for(storage) is None
    assert span_label(storage) == "all available"


def test_a_config_with_no_opinion_gets_the_default_span():
    storage = StorageConfig()
    assert analysis_span_days(storage) == parse_analysis_span("90d")
    assert retention_days_for(storage) == 90


# --- backwards compatibility ------------------------------------------------


def test_a_pre_coupling_config_has_its_retention_read_as_the_span():
    """Every config written before this existed sets retention_days and nothing
    else. What such a user kept is the most the product could ever have
    analyzed, so adopting it as the span changes nothing about their setup."""
    storage = StorageConfig(retention_days=45)
    assert analysis_span_days(storage) == 45
    assert retention_days_for(storage) == 45
    assert not retention_was_raised_to_span(storage)


def test_keeping_more_than_you_analyze_is_left_alone():
    """Retention longer than the span costs disk and misleads nobody."""
    storage = StorageConfig(analysis_span="30d", retention_days=365)
    assert retention_days_for(storage) == 365
    assert not retention_was_raised_to_span(storage)


def test_retention_shorter_than_the_span_is_raised_to_it():
    """The invariant, and it is one-directional.

    Lowering storage retention must not be able to silently retract a span the
    product has already promised to analyze over, so the clamp only moves
    retention UP — it never shortens the span to match.
    """
    storage = StorageConfig(analysis_span="90d", retention_days=7)
    assert analysis_span_days(storage) == 90
    assert retention_days_for(storage) == 90
    assert retention_was_raised_to_span(storage)


def test_an_all_available_span_beats_any_explicit_retention():
    storage = StorageConfig(analysis_span="all", retention_days=7)
    assert retention_days_for(storage) is None
    assert retention_was_raised_to_span(storage)


# --- the span an analyzer may actually accumulate over ----------------------


def test_the_window_is_the_choice_bounded_by_what_the_store_holds():
    storage = StorageConfig(analysis_span="90d")
    assert resolved_window_days(storage, 68) == 68     # store is younger
    assert resolved_window_days(storage, 400) == 90    # choice binds


def test_an_unknown_available_span_narrows_nothing():
    """`data_span` returns None for "we do not know", never for zero — an
    unknown span must not be read as an empty one."""
    assert resolved_window_days(StorageConfig(analysis_span="90d"), None) == 90
    assert resolved_window_days(StorageConfig(analysis_span="all"), None) is None


def test_an_unbounded_span_takes_the_whole_available_history():
    assert resolved_window_days(StorageConfig(analysis_span="all"), 68) == 68


# --- the job ----------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    yield db
    db.close()


def _seed(db, *, days_ago: float, session_id: str) -> None:
    at = utcnow() - timedelta(days=days_ago)
    db.upsert_session(make_session(
        session_id=session_id, started_at=at, ended_at=at + timedelta(minutes=1),
    ))
    db.insert_span(make_llm_span(session_id=session_id, start_time=at))


def test_retention_cannot_delete_inside_the_span_it_is_derived_from(store):
    _seed(store, days_ago=60, session_id="inside")
    _seed(store, days_ago=120, session_id="outside")

    run = run_retention_cleanup(store, StorageConfig(analysis_span="90d"))

    assert run.retention_days == 90
    assert (run.spans_deleted, run.sessions_deleted) == (1, 1)
    remaining = [
        r[0] for r in store.conn.execute("SELECT session_id FROM sessions").fetchall()
    ]
    assert remaining == ["inside"]


def test_an_all_available_span_deletes_nothing_and_says_why(store):
    _seed(store, days_ago=3_000, session_id="ancient")

    run = run_retention_cleanup(store, StorageConfig(analysis_span="all"))

    assert (run.spans_deleted, run.sessions_deleted) == (0, 0)
    assert run.cutoff is None
    assert run.skipped_reason and "disabled" in run.skipped_reason
    assert store.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1


def test_a_retention_shorter_than_the_span_does_not_reach_inside_it(store):
    """The clamp is enforced by the job, not only by the config writer.

    A hand-edited config setting a 7-day retention under a 90-day span must not
    be able to delete the history the product is still analyzing.
    """
    _seed(store, days_ago=60, session_id="inside")

    run = run_retention_cleanup(
        store, StorageConfig(analysis_span="90d", retention_days=7),
    )

    assert run.retention_days == 90
    assert (run.spans_deleted, run.sessions_deleted) == (0, 0)
    assert store.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1


# --- the ledger -------------------------------------------------------------


def test_every_run_leaves_a_row_saying_what_it_removed(store):
    _seed(store, days_ago=120, session_id="outside")
    _seed(store, days_ago=2, session_id="inside")

    run_retention_cleanup(store, StorageConfig(analysis_span="90d"))

    rows = store.conn.execute(
        "SELECT retention_days, analysis_span_days, spans_deleted, "
        "sessions_deleted, oldest_kept FROM retention_events"
    ).fetchall()
    assert len(rows) == 1
    retention_days, span_days, spans_deleted, sessions_deleted, oldest_kept = rows[0]
    assert (retention_days, span_days) == (90, 90)
    assert (spans_deleted, sessions_deleted) == (1, 1)
    # Read AFTER the delete, so the row states what survived rather than what
    # the run intended to leave.
    assert oldest_kept is not None
    assert (utcnow() - oldest_kept).days < 90


def test_a_run_that_deletes_nothing_still_leaves_a_row(store):
    """Silence is the state this ledger exists to remove. A run with nothing to
    delete is evidence the job ran; no row at all is evidence of nothing."""
    _seed(store, days_ago=2, session_id="inside")
    run_retention_cleanup(store, StorageConfig(analysis_span="90d"))
    assert store.conn.execute(
        "SELECT COUNT(*) FROM retention_events"
    ).fetchone()[0] == 1


def test_a_disabled_run_writes_no_row_because_it_did_not_run(store):
    run_retention_cleanup(store, StorageConfig(analysis_span="all"))
    assert store.conn.execute(
        "SELECT COUNT(*) FROM retention_events"
    ).fetchone()[0] == 0


def test_a_ledger_write_failure_never_undoes_a_delete_that_happened(store, caplog):
    """A trimmed store plus a run reading as a failure is the worst outcome."""
    store.conn.execute("DROP TABLE retention_events")
    _seed(store, days_ago=120, session_id="outside")

    run = run_retention_cleanup(store, StorageConfig(analysis_span="90d"))

    assert run.spans_deleted == 1
    assert store.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0
    assert "ledger row could not be written" in caplog.text


def test_the_deletion_no_longer_skews_the_span_it_leaves_behind(store):
    """The orphaned-session bug, stated as the measure it corrupted.

    `available_data_span` unions `sessions.started_at` into its day set, so
    leaving the parent rows behind made the deletion inflate the very measure of
    how much history survived it.
    """
    from tokenjam.core.data_span import available_data_span

    _seed(store, days_ago=200, session_id="outside")
    _seed(store, days_ago=1, session_id="inside")

    run_retention_cleanup(store, StorageConfig(analysis_span="90d"))

    span = available_data_span(store.conn)
    assert span.days_with_data == 1
    assert span.ignored_days_before_block == 0


# --- the past-overspend basis ----------------------------------------------


def test_the_cost_window_follows_the_chosen_span_not_a_fixed_month(store):
    """A rolling 30 days was the whole look-back, and it is not history.

    A dead MCP server injected into hundreds of sessions with zero invocations
    did not begin costing money 30 days ago; pricing it on a fixed month
    understates a PAST-tense figure by however long the behaviour actually ran.
    """
    from tokenjam.core.optimize.cost_proposals import (
        FALLBACK_COST_WINDOW_DAYS,
        cost_window_days_for,
    )
    from types import SimpleNamespace

    for days_ago in range(0, 80, 5):        # a contiguous ~80-day block
        _seed(store, days_ago=days_ago, session_id=f"s{days_ago}")

    config = SimpleNamespace(storage=StorageConfig(analysis_span="90d"))
    days = cost_window_days_for(config, store.conn)

    assert days > FALLBACK_COST_WINDOW_DAYS
    assert days == 76                        # 0..75 days ago inclusive


def test_the_cost_window_cannot_exceed_what_the_store_holds(store):
    """A 90-day promise over a week-old store is answerable for a week."""
    from types import SimpleNamespace

    from tokenjam.core.optimize.cost_proposals import cost_window_days_for

    for days_ago in range(0, 5):
        _seed(store, days_ago=days_ago, session_id=f"s{days_ago}")

    config = SimpleNamespace(storage=StorageConfig(analysis_span="90d"))
    assert cost_window_days_for(config, store.conn) == 5


def test_an_all_available_span_takes_the_measured_span(store):
    from types import SimpleNamespace

    from tokenjam.core.optimize.cost_proposals import cost_window_days_for

    for days_ago in range(0, 12):
        _seed(store, days_ago=days_ago, session_id=f"s{days_ago}")

    config = SimpleNamespace(storage=StorageConfig(analysis_span="all"))
    assert cost_window_days_for(config, store.conn) == 12


def test_a_narrower_choice_still_binds(store):
    from types import SimpleNamespace

    from tokenjam.core.optimize.cost_proposals import cost_window_days_for

    for days_ago in range(0, 80, 5):
        _seed(store, days_ago=days_ago, session_id=f"s{days_ago}")

    config = SimpleNamespace(storage=StorageConfig(analysis_span="30d"))
    assert cost_window_days_for(config, store.conn) == 30


def test_the_constant_is_reached_only_when_neither_bound_exists(store):
    """An unreadable store AND an unbounded choice — the one case with no
    answer. A window has to be a number to subtract from now."""
    from types import SimpleNamespace

    from tokenjam.core.optimize.cost_proposals import (
        FALLBACK_COST_WINDOW_DAYS,
        cost_window_days_for,
    )

    config = SimpleNamespace(storage=StorageConfig(analysis_span="all"))
    assert cost_window_days_for(config, None) == FALLBACK_COST_WINDOW_DAYS
    assert cost_window_days_for(SimpleNamespace(), None) == FALLBACK_COST_WINDOW_DAYS


def test_retention_can_never_delete_underneath_the_cost_window(store):
    """The two halves of the coupling, met in the middle.

    The window past-overspend accumulates over and the cutoff retention deletes
    at are derived from the same span, so the figure can never be computed over
    a period whose data the product has already destroyed.
    """
    from types import SimpleNamespace

    from tokenjam.core.optimize.cost_proposals import cost_window_days_for

    for days_ago in range(0, 200, 5):
        _seed(store, days_ago=days_ago, session_id=f"s{days_ago}")

    storage = StorageConfig(analysis_span="90d")
    run = run_retention_cleanup(store, storage)

    window = cost_window_days_for(SimpleNamespace(storage=storage), store.conn)
    assert run.retention_days == 90
    assert window <= run.retention_days
