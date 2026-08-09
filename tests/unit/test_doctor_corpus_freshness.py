"""`tj doctor`'s corpus-freshness check: a stale corpus can never render as
healthy, and must FAIL (not warn) once nothing is positioned to close the
gap automatically -- see the daemon-liveness check this pairs with.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.cli.cmd_doctor import _check_corpus_freshness
from tokenjam.core.config import IngestConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_session


@pytest.fixture()
def db():
    backend = DuckDBBackend(StorageConfig(path=":memory:"))
    yield backend
    backend.close()


def _config() -> TjConfig:
    config = TjConfig(version="1")
    config.ingest = IngestConfig(interval_minutes=30)
    return config


def test_info_when_no_sessions_ever(db: DuckDBBackend) -> None:
    check = _check_corpus_freshness(_config(), db, daemon_alive=True)
    assert check["level"] == "info"


def test_ok_when_recently_ingested(db: DuckDBBackend) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(minutes=5)))
    check = _check_corpus_freshness(_config(), db, daemon_alive=True)
    assert check["level"] == "ok"


def test_warning_when_stale_but_daemon_is_alive(db: DuckDBBackend) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    check = _check_corpus_freshness(_config(), db, daemon_alive=True)
    assert check["level"] == "warning"


def test_error_when_stale_and_daemon_is_dead(db: DuckDBBackend) -> None:
    """This is the real-machine scenario: newest session 2 days stale, and
    nothing is running to catch it up. Must FAIL, never merely warn."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    check = _check_corpus_freshness(_config(), db, daemon_alive=False)
    assert check["level"] == "error"
    assert "not running" in check["message"]


def test_a_stale_corpus_never_renders_as_ok(db: DuckDBBackend) -> None:
    """Never assert more than the data supports: a corpus this far past its
    own configured cadence must never come back level 'ok', regardless of
    daemon state."""
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=5)))
    for daemon_alive in (True, False):
        check = _check_corpus_freshness(_config(), db, daemon_alive=daemon_alive)
        assert check["level"] != "ok"
