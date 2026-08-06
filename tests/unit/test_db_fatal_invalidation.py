"""A fatal DuckDB error must stop the process serving, not be logged per-row.

The failure these pin, observed on a real database: a background analyzer pass
persisting one agent-config row raised

    duckdb.FatalException: FATAL Error: Invalid Input Error: Failed to delete
    all rows from index. Only deleted 0 out of 1 rows.

which `DuckDBAgentConfigStore` caught and logged as a soft "this record could
not be persisted" warning before carrying on. But that exception invalidates
the whole DuckDB database INSTANCE, not the connection that raised it — so
every later query in the process, on every connection including ones opened
afterwards, failed with "database has been invalidated because of a previous
fatal error". Every API route 500'd, the dashboard rendered empty, and
`/health` went on returning `{"status": "ok"}` because it never touched the
database.

Three properties, each of which alone would have prevented the outage:

  * the per-record handler must not swallow a fatal (`test_fatal_*`);
  * the health probe must ask the database rather than assert liveness
    (`test_health_*`);
  * recovery must close EVERY connection before reopening, because DuckDB
    hands back the same invalidated instance otherwise (`test_recover_*`).

The root-cause index fault is a persistent property of one database file, so
the fatal itself is simulated here; `test_repair_agent_config_index_*` pins the
repair that clears it.
"""
from __future__ import annotations

import time

import duckdb
import pytest
from fastapi.testclient import TestClient

from tokenjam.core.agent_config import ConfigRecord, DuckDBAgentConfigStore
from tokenjam.core.db import (
    AGENT_CONFIG_INDEXES,
    DuckDBBackend,
    check_agent_config_index_corruption,
    clear_fatal_db_error,
    fatal_db_error,
    is_fatal_db_error,
    note_fatal_db_error,
    recover_invalidated_database,
    repair_agent_config_index,
    run_migrations,
)
from tokenjam.core.config import StorageConfig
from tokenjam.utils.time_parse import utcnow


@pytest.fixture(autouse=True)
def _clean_fatal_state():
    """The fatal record is process-wide (the invalidation is), so reset it."""
    clear_fatal_db_error()
    yield
    clear_fatal_db_error()


@pytest.fixture
def backend(tmp_path):
    b = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield b
    b.close()


class _StubConfig:
    """Stands in for TjConfig: the cycle raises before it reads any of it."""


def _record(name: str = "CLAUDE.md") -> ConfigRecord:
    return ConfigRecord(
        kind="instruction",
        scope="project",
        root="/tmp/p",
        name=name,
        path=f"/tmp/p/{name}",
        size_bytes=10,
        tokens=3,
        content_hash="h",
        last_seen=utcnow(),
    )


# --- classification --------------------------------------------------------

def test_fatal_exception_is_classified_fatal():
    assert is_fatal_db_error(duckdb.FatalException("FATAL Error: boom"))


def test_invalidation_message_is_classified_fatal_even_if_rewrapped():
    # The type is authoritative, but a fatal that reaches a handler wrapped by
    # an intermediate layer must still be recognised — misclassifying it is
    # exactly what turns the outage silent.
    assert is_fatal_db_error(
        RuntimeError("database has been invalidated because of a previous fatal error")
    )


def test_ordinary_write_conflict_is_not_fatal():
    # The conflict-tolerant retry path must keep working: these are the errors
    # the agent-config store is DESIGNED to degrade on, and promoting them to
    # fatal would take down a pass for a recoverable race.
    assert not is_fatal_db_error(duckdb.ConstraintException("Duplicate key"))
    assert not is_fatal_db_error(duckdb.TransactionException("Conflict on tuple deletion!"))


# --- the per-record handler ------------------------------------------------

def test_fatal_during_upsert_is_raised_not_degraded(backend):
    """The regression. A fatal inside the persistence loop must escape it."""
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Exploding:
        def execute(self, *a, **k):
            raise duckdb.FatalException(
                "FATAL Error: Invalid Input Error: Failed to delete all rows "
                "from index. Only deleted 0 out of 1 rows."
            )

    store.conn = Exploding()
    with pytest.raises(duckdb.FatalException):
        store.upsert([_record()])


def test_fatal_during_upsert_is_recorded_process_wide(backend):
    """...and leaves a record, so surfaces stop claiming to be healthy."""
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Exploding:
        def execute(self, *a, **k):
            raise duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")

    store.conn = Exploding()
    assert fatal_db_error() is None
    with pytest.raises(duckdb.FatalException):
        store.upsert([_record()])
    assert "Failed to delete all rows from index" in (fatal_db_error() or "")


def test_recoverable_write_failure_still_degrades_quietly(backend, caplog):
    """The inverse guard: widening 'fatal' must not break ordinary degrading.

    A write-write conflict is the case the store exists to absorb — it must
    still land in the mirror, still be readable, and still not raise.
    """
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Conflicted:
        def execute(self, *a, **k):
            raise duckdb.ConstraintException("Duplicate key")

    store.conn = Conflicted()
    store.upsert([_record()])  # must not raise
    assert store.degraded
    assert fatal_db_error() is None
    assert [r.name for r in store.select()] == ["CLAUDE.md"]


# --- connection health and recovery ----------------------------------------

def test_check_health_true_on_a_live_backend(backend):
    assert backend.check_health() is True


def test_check_health_false_when_the_connection_cannot_answer(backend):
    backend.close()
    assert backend.check_health() is False


def test_recover_reestablishes_a_torn_down_backend(backend):
    """Closing every handle then reopening is the only in-process recovery.

    Verified against the engine: while ANY connection to the path survives,
    `duckdb.connect` returns the same (invalidated) instance from DuckDB's
    per-path cache, so a reconnect that does not close first recovers nothing.
    """
    backend.conn.execute(
        "INSERT INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen) VALUES ('a','instruction','p','/x',now())"
    )
    backend._teardown_connections()
    assert backend.check_health() is False

    note_fatal_db_error(duckdb.FatalException("FATAL Error: simulated"))
    assert recover_invalidated_database() is True

    assert backend.check_health() is True
    assert fatal_db_error() is None
    row = backend.conn.execute("SELECT COUNT(*) FROM agent_config_files").fetchone()
    assert row[0] == 1, "recovery must reopen the database, not replace it"


def test_recover_hands_every_thread_a_fresh_cursor(backend):
    """A thread holding a pre-recovery cursor must not keep using it."""
    stale = backend.conn
    backend._teardown_connections()
    recover_invalidated_database()
    assert backend.conn is not stale
    backend.conn.execute("SELECT 1").fetchone()


def test_recovery_closes_cursors_belonging_to_other_threads(backend):
    """The reason cursors are tracked in a list rather than left to the thread.

    Request-path cursors live in a `threading.local`, which the recovering
    thread cannot enumerate. One surviving handle keeps the invalidated
    instance in DuckDB's per-path cache, so a recovery that closed only its own
    connection would reopen straight back onto the dead database. After
    recovery every thread — including ones that were holding a cursor — must be
    able to query again.
    """
    import threading as _t

    errors: list[str] = []
    started, resume = _t.Event(), _t.Event()

    def worker():
        try:
            backend.conn.execute("SELECT 1").fetchone()  # take a per-thread cursor
            started.set()
            resume.wait(5)
            backend.conn.execute("SELECT 1").fetchone()  # must work post-recovery
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            started.set()

    backend.conn.execute("SELECT 1").fetchone()  # this thread's cursor
    t = _t.Thread(target=worker)
    t.start()
    assert started.wait(5)
    assert len(backend._cursors) == 2, "both threads' cursors must be tracked"

    backend._teardown_connections()
    assert recover_invalidated_database() is True
    resume.set()
    t.join(10)

    assert errors == []
    assert backend._cursors, "recovery must hand out fresh cursors, not reuse closed ones"


def test_in_memory_backend_is_not_torn_down_by_recovery():
    """Its database IS its connection — 'recovering' it would delete the data."""
    from tokenjam.core.db import InMemoryBackend

    mem = InMemoryBackend()
    mem.conn.execute(
        "INSERT INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen) VALUES ('a','instruction','p','/x',now())"
    )
    assert mem.recoverable is False
    recover_invalidated_database()
    row = mem.conn.execute("SELECT COUNT(*) FROM agent_config_files").fetchone()
    assert row[0] == 1


# --- the broad `except Exception` handlers ---------------------------------
#
# Both background jobs are fire-and-forget: they start a daemon thread and
# return, and each thread already swallows every exception ("never crash a
# thread", "errors are logged, never raised"). That is right for a job that
# failed and catastrophic for a fatal, and it is where the outage's exception
# actually died — a guard around the DISPATCH would never have seen it.

def test_handle_if_fatal_recovers_and_reports_it_handled(backend):
    from tokenjam.core.db import handle_if_fatal

    backend._teardown_connections()
    assert handle_if_fatal(
        duckdb.FatalException("FATAL Error: simulated"), what="a job"
    ) is True
    assert backend.check_health() is True
    assert fatal_db_error() is None


def test_handle_if_fatal_leaves_ordinary_failures_to_their_caller(backend):
    """It must return False for anything else, or every job failure would
    trigger a needless full teardown of the process's connections."""
    from tokenjam.core.db import handle_if_fatal

    assert handle_if_fatal(ValueError("a parse failed"), what="a job") is False
    assert handle_if_fatal(
        duckdb.ConstraintException("Duplicate key"), what="a job"
    ) is False
    assert backend.check_health() is True


def test_transcript_catch_up_does_not_swallow_a_fatal(backend, monkeypatch):
    """The catch-up thread's handler must escalate a fatal, not log a warning."""
    from tokenjam.core import transcript_sync

    seen: list[str] = []
    monkeypatch.setattr(
        "tokenjam.core.db.handle_if_fatal",
        lambda exc, what: (seen.append(what), True)[1],
    )
    monkeypatch.setattr(
        transcript_sync, "run_catch_up",
        lambda *a, **k: (_ for _ in ()).throw(
            duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")
        ),
    )
    transcript_sync.start_catch_up(lambda: backend).join(10)
    assert seen == ["transcript catch-up"]


def test_scan_cycle_does_not_swallow_a_fatal(backend, monkeypatch):
    """Same guard on the analyzer cycle's `except Exception` thread handler."""
    from tokenjam.core.optimize import scan_cycle

    seen: list[str] = []
    monkeypatch.setattr(
        "tokenjam.core.db.handle_if_fatal",
        lambda exc, what: (seen.append(what), True)[1],
    )
    # Raise the fatal from the first thing the cycle's thread body does with
    # the database, so the real `except Exception` handler is what catches it.
    monkeypatch.setattr(
        "tokenjam.core.optimize.cycle_provenance.begin_cycle",
        lambda *a, **k: (_ for _ in ()).throw(
            duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")
        ),
    )
    assert scan_cycle.trigger_scan_cycle(lambda: backend, _StubConfig(), force=True)
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)
    assert seen == ["analyzer scan cycle"]


# --- the health surface ----------------------------------------------------

def _app(db):
    from tokenjam.api.app import create_app
    from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig

    config = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))
    return create_app(config, db, ingest_pipeline=object())


def test_health_reports_ok_and_says_storage_was_checked(backend):
    with TestClient(_app(backend)) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["storage"] == "ok"


def test_health_reports_503_when_the_database_cannot_be_recovered(backend, monkeypatch):
    """The core surface regression: liveness must not be reported as health.

    A process that is up but cannot read its database served every route as a
    500 while `/health` returned `{"status": "ok"}`. Nothing may render a
    green status off a probe that never asked the database.
    """
    monkeypatch.setattr(type(backend), "check_health", lambda self: False)
    monkeypatch.setattr(
        "tokenjam.core.db.recover_invalidated_database", lambda **kw: False
    )
    with TestClient(_app(backend)) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["storage"] == "invalidated"


def test_health_recovers_an_invalidated_database_in_place(backend):
    """The polling that notices the outage is allowed to end it."""
    backend._teardown_connections()
    note_fatal_db_error(duckdb.FatalException("FATAL Error: simulated"))
    with TestClient(_app(backend)) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["storage"] == "recovered"
    assert "simulated" in body["recovered_from"]
    assert backend.check_health() is True


def test_health_never_claims_storage_state_it_did_not_check():
    """A backend that cannot be probed reports 'unknown', never 'ok'."""
    class Opaque:
        pass

    with TestClient(_app(Opaque())) as client:
        body = client.get("/health").json()
    assert body["storage"] == "unknown"


# --- the repair ------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = duckdb.connect(str(tmp_path / "r.duckdb"))
    run_migrations(c)
    yield c
    c.close()


def _seed(conn, n: int) -> None:
    for i in range(n):
        conn.execute(
            "INSERT INTO agent_config_files "
            "(config_id, kind, scope, path, last_seen, tokens) "
            "VALUES ($1,'instruction','project',$2, now(), $3)",
            [f"id-{i}", f"/tmp/p{i}/CLAUDE.md", i],
        )


def test_repair_agent_config_index_preserves_every_row(conn):
    _seed(conn, 25)
    assert repair_agent_config_index(conn) == 25
    rows = conn.execute(
        "SELECT config_id, tokens FROM agent_config_files ORDER BY tokens"
    ).fetchall()
    assert rows == [(f"id-{i}", i) for i in range(25)]


def test_repair_agent_config_index_is_idempotent(conn):
    _seed(conn, 5)
    assert repair_agent_config_index(conn) == 5
    assert repair_agent_config_index(conn) == 5
    assert repair_agent_config_index(conn) == 5


def test_repair_agent_config_index_is_safe_on_an_empty_table(conn):
    assert repair_agent_config_index(conn) == 0


def test_repair_agent_config_index_recreates_both_indexes(conn):
    """Dropping without re-issuing would leave the table permanently unindexed:
    the migrations are already recorded applied, so nothing else puts them back."""
    _seed(conn, 3)
    conn.execute("DROP INDEX idx_agent_config_kind")
    repair_agent_config_index(conn)
    names = {
        r[0] for r in conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'agent_config_files'"
        ).fetchall()
    }
    assert {name for name, _ in AGENT_CONFIG_INDEXES} <= names


def test_repair_agent_config_index_leaves_the_primary_key_enforced(conn):
    """The repair touches only the two secondary indexes. The PRIMARY KEY is
    not the damaged one and must come through untouched."""
    _seed(conn, 3)
    repair_agent_config_index(conn)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO agent_config_files "
            "(config_id, kind, scope, path, last_seen) "
            "VALUES ('id-0','instruction','project','/dup', now())"
        )


def test_repair_agent_config_index_leaves_the_table_writable(conn):
    """The point of the repair: DELETE and INSERT OR REPLACE work afterwards.

    These are the two statements a damaged index makes fatal, so a repair that
    did not restore them would clear nothing.
    """
    _seed(conn, 4)
    repair_agent_config_index(conn)
    conn.execute("DELETE FROM agent_config_files WHERE config_id = 'id-1'")
    conn.execute(
        "INSERT OR REPLACE INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen, tokens) "
        "VALUES ('id-2','instruction','project','/moved', now(), 99)"
    )
    rows = conn.execute(
        "SELECT config_id, tokens FROM agent_config_files ORDER BY config_id"
    ).fetchall()
    assert rows == [("id-0", 0), ("id-2", 99), ("id-3", 3)]


# --- the integrity probe ---------------------------------------------------

def test_check_reports_no_faults_on_a_healthy_table(conn):
    _seed(conn, 10)
    assert check_agent_config_index_corruption(conn) == []


def test_check_reports_an_absent_index(conn):
    _seed(conn, 4)
    conn.execute("DROP INDEX idx_agent_config_kind")
    faults = check_agent_config_index_corruption(conn)
    assert ("idx_agent_config_kind", "absent from the catalogue") in faults


def test_check_is_quiet_on_an_empty_table(conn):
    """Nothing to compare is not evidence of damage."""
    assert check_agent_config_index_corruption(conn) == []


def test_probe_uses_a_form_the_index_cannot_serve(conn):
    """The subtle half, and the reason this probe nearly did not work.

    `CAST(col AS VARCHAR)` is a NO-OP on a column that is already VARCHAR, so
    the planner discards it and the index serves both sides of the comparison —
    a probe built on it compares a damaged index against itself and reports
    sound whatever the damage. Concatenating an empty string is a real
    expression for every type. This pins that the scan form still agrees with
    a GROUP BY (which no index can serve) on a healthy table, so the two forms
    are genuinely answering the same question.
    """
    _seed(conn, 12)
    truth = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM agent_config_files GROUP BY 1"
    ).fetchall())
    scanned = conn.execute(
        "SELECT COUNT(*) FROM agent_config_files "
        "WHERE CAST(kind AS VARCHAR) || '' = CAST('instruction' AS VARCHAR)"
    ).fetchone()[0]
    assert scanned == truth["instruction"] == 12
