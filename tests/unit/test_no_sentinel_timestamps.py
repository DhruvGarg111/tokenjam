"""Ingest may not stamp a timestamp it did not observe.

A missing source timestamp used to become `1970-01-01`, and a sentinel is not a
neutral placeholder: it participates in `MIN()`, in `ORDER BY`, and in every day
union, so ONE such row made a corpus with two months of usable history report a
span in the thousands of days. `core/data_span.py` defends against it at READ
time, which protects the surfaces that remember to and no others — a new
aggregate doing a naive `MIN(start_time)` is wrong again on the first try.

NULL is the representation that defends itself: every comparison, range filter
and aggregate excludes it under ordinary SQL semantics, so there is no per-query
guard to forget. The row is still ingested and still counted; it just cannot time
anything.

The paths below are live, not theoretical. Claude Code's own OTel log exporter
sends `timeUnixNano=0` on some records, and Codex's ISO-8601 fallback can fail to
parse — that is where the sentinels on a real store came from.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tokenjam.api.routes.logs import _api_request_to_span, _ts_to_datetime
from tokenjam.core.config import StorageConfig
from tokenjam.core.data_span import MIN_PLAUSIBLE_YEAR, available_data_span
from tokenjam.core.db import DuckDBBackend
from tokenjam.otel.otlp_parsing import parse_otlp_span
from tokenjam.otel.semconv import ClaudeCodeEvents


# --- the logs path (Claude Code + Codex) ------------------------------------


@pytest.mark.parametrize("timestamp_ns", [0, -1])
def test_an_absent_log_timestamp_becomes_none_not_a_zero_epoch(timestamp_ns):
    assert _ts_to_datetime(timestamp_ns) is None


def test_a_real_log_timestamp_is_preserved():
    assert _ts_to_datetime(1_800_000_000_000_000_000) == datetime.fromtimestamp(
        1_800_000_000, tz=timezone.utc,
    )


def test_an_untimed_claude_code_span_carries_no_time_at_either_end():
    """`end_time` is derived from `start_time`, so it has to degrade with it —
    a span starting at NULL and ending at a real instant is worse than either."""
    span = _api_request_to_span(
        {
            ClaudeCodeEvents.SESSION_ID: "sess-1",
            ClaudeCodeEvents.DURATION_MS: 1_200,
            ClaudeCodeEvents.INPUT_TOKENS: 100,
            ClaudeCodeEvents.OUTPUT_TOKENS: 10,
        },
        {},
        0,
    )
    assert span is not None
    assert span.start_time is None
    assert span.end_time is None


# --- the OTLP path ----------------------------------------------------------


def test_an_otlp_span_with_no_start_time_is_untimed_rather_than_now():
    """Substituting `datetime.now()` dates historical work to whenever tj
    happened to receive it, which reads as a real observation."""
    span = parse_otlp_span(
        {"spanId": "s1", "traceId": "t1", "name": "chat", "attributes": []}, {},
    )
    assert span.start_time is None
    assert span.duration_ms is None


def test_an_otlp_span_with_a_start_time_is_unaffected():
    span = parse_otlp_span(
        {
            "spanId": "s1", "traceId": "t1", "name": "chat",
            "startTimeUnixNano": "1800000000000000000",
            "endTimeUnixNano": "1800000001000000000",
            "attributes": [],
        },
        {},
    )
    assert span.start_time == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)
    assert span.duration_ms == pytest.approx(1000.0)


# --- what an untimed row does to the store ----------------------------------


@pytest.fixture
def store(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    yield db
    db.close()


def test_an_untimed_span_is_stored_and_counted_but_times_nothing(store):
    from tests.factories import make_llm_span
    from tokenjam.utils.time_parse import utcnow

    store.insert_span(make_llm_span(session_id="s1", start_time=utcnow()))
    store.insert_span(make_llm_span(session_id="s2", start_time=None))

    # Kept: the row still carries tokens, cost and an agent.
    assert store.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 2
    # But it cannot place anything on a calendar, and needs no guard to say so.
    assert store.conn.execute(
        "SELECT COUNT(*) FROM spans WHERE start_time IS NULL"
    ).fetchone()[0] == 1
    span = available_data_span(store.conn)
    assert span.days_with_data == 1


def test_a_naive_min_over_the_store_can_no_longer_be_dragged_to_1970(store):
    """The generalisation, and the reason the fix is at the WRITE side.

    Read-time defences only protect the queries that remember them. With NULL
    there is nothing for an unguarded aggregate to pick up.
    """
    from tests.factories import make_llm_span
    from tokenjam.utils.time_parse import utcnow

    store.insert_span(make_llm_span(session_id="s1", start_time=utcnow()))
    store.insert_span(make_llm_span(session_id="s2", start_time=None))

    oldest = store.conn.execute("SELECT MIN(start_time) FROM spans").fetchone()[0]
    assert oldest is not None
    assert oldest.year >= MIN_PLAUSIBLE_YEAR


def test_the_migration_deletes_sentinel_rows_rather_than_nulling_them(tmp_path):
    """A row whose only recorded fact was a false timestamp has nothing left to
    attribute once that is removed; keeping it would leave a row counted by
    every COUNT(*) and placed by nothing."""
    import duckdb

    from tokenjam.core.db import MIGRATIONS, run_migrations

    path = str(tmp_path / "legacy.duckdb")
    conn = duckdb.connect(path)
    # Migrate to just before the sentinel purge, then write what an older build
    # would have written.
    for version, sql in MIGRATIONS:
        if version >= 21:
            break
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ)"
        )
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(statement.strip())
        conn.execute(
            "INSERT INTO schema_migrations VALUES (?, now())", [version],
        )
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    conn.execute(
        "INSERT INTO spans (span_id, trace_id, name, kind, status_code, start_time)"
        " VALUES ('sentinel','t','x','internal','ok',?)", [epoch],
    )
    conn.execute(
        "INSERT INTO spans (span_id, trace_id, name, kind, status_code, start_time)"
        " VALUES ('real','t','x','internal','ok',?)", [now],
    )
    conn.execute(
        "INSERT INTO sessions (session_id, agent_id, started_at)"
        " VALUES ('sentinel','a',?)", [epoch],
    )
    conn.execute(
        "INSERT INTO sessions (session_id, agent_id, started_at)"
        " VALUES ('real','a',?)", [now],
    )

    run_migrations(conn)
    try:
        assert [r[0] for r in conn.execute(
            "SELECT span_id FROM spans ORDER BY span_id").fetchall()] == ["real"]
        assert [r[0] for r in conn.execute(
            "SELECT session_id FROM sessions ORDER BY session_id").fetchall()] == ["real"]
        # And the columns are nullable afterwards, so ingest can write NULL.
        nullable = dict(conn.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE (table_name = 'spans' AND column_name = 'start_time') "
            "OR (table_name = 'sessions' AND column_name = 'started_at')"
        ).fetchall())
        assert set(nullable.values()) == {"YES"}
        # The indexes the ALTER had to drop are back — DuckDB refuses to alter a
        # column on a table carrying ART indexes, and nothing else recreates them.
        from tokenjam.core.db import SESSIONS_INDEXES, SPANS_INDEXES

        present = {
            r[0] for r in conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
        }
        for name, _ in SPANS_INDEXES + SESSIONS_INDEXES:
            assert name in present, name
    finally:
        conn.close()
