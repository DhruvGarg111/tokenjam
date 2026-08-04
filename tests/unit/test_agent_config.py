"""Unit tests for the ingested agent-config surface (core/agent_config.py).

The property under test is not "the walk finds files" — three analyzers already
did that. It is that what the walk found is STORED, that reading it back gives
the same population in the same order, and that a persistent store cannot hand a
later run a root the current one never asked about.
"""
from __future__ import annotations

import json
from datetime import timedelta

import duckdb
import pytest

from tokenjam.core import agent_config as ac
from tokenjam.core.db import run_migrations
from tokenjam.core.summarize.catalog import Catalog
from tokenjam.utils.time_parse import utcnow


@pytest.fixture
def catalog(tmp_path):
    """A controlled catalog, so nothing here reads the developer's real home."""
    gfile = tmp_path / "globalhome" / ".claude" / "CLAUDE.md"
    gfile.parent.mkdir(parents=True)
    gfile.write_text("global instructions " * 50, encoding="utf-8")
    return Catalog(
        project_files=frozenset({"CLAUDE.md", "AGENTS.md"}),
        project_globs=(".claude/agents/*.md", ".claude/commands/*.md"),
        global_paths=(str(gfile),),
        forbidden_roots=(),
    )


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    (root / ".claude" / "agents").mkdir(parents=True)
    (root / ".claude" / "commands").mkdir(parents=True)
    (root / "CLAUDE.md").write_text("project rules " * 40, encoding="utf-8")
    (root / ".claude" / "agents" / "worker.md").write_text("agent " * 40, encoding="utf-8")
    (root / ".claude" / "commands" / "ship.md").write_text("ship " * 40, encoding="utf-8")
    return root


def _duck_store(tmp_path):
    conn = duckdb.connect(str(tmp_path / "t.duckdb"))
    run_migrations(conn)
    return conn, ac.DuckDBAgentConfigStore(conn)


# --- Instruction files ------------------------------------------------------

def test_instruction_files_are_ingested_with_size_hash_and_slot(catalog, project):
    store = ac.InMemoryAgentConfigStore()
    at = ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
    )
    rows = store.select(kind=ac.KIND_INSTRUCTION, seen_at=at)

    by_slot = {r.name: r for r in rows}
    assert "CLAUDE.md" in by_slot
    assert ".claude/agents/worker.md" in by_slot
    assert ".claude/commands/ship.md" in by_slot
    claude_md = by_slot["CLAUDE.md"]
    assert claude_md.size_bytes == (project / "CLAUDE.md").stat().st_size
    assert claude_md.tokens > 0
    assert len(claude_md.content_hash) == 64
    assert claude_md.root == str(project)
    # The SLOT is what makes the same file under two worktrees recognisable as
    # one file seen twice, which is why it is stored alongside the abs path.
    assert by_slot[".claude/agents/worker.md"].subkind == "agent"
    assert by_slot[".claude/commands/ship.md"].subkind == "command"


def test_the_catalog_is_the_only_thing_that_decides_what_counts(catalog, project):
    """A file the catalog does not name is not ingested, however plausible.

    Pinned because "just also pick up X" is a one-line change here and a
    silent widening of every figure downstream.
    """
    (project / "NOTES.md").write_text("notes " * 100, encoding="utf-8")
    store = ac.InMemoryAgentConfigStore()
    at = ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
    )
    assert "NOTES.md" not in {r.name for r in store.select(seen_at=at)}


def test_a_changed_file_keeps_its_id_and_changes_its_hash(catalog, project):
    """Drift is visible as a hash change under a stable id.

    That is the question a consumer actually has — "did this file change since
    we priced it" — and it is why the hash is stored rather than only the size.
    """
    store = ac.InMemoryAgentConfigStore()
    ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
    )
    before = {r.name: r for r in store.select(kind=ac.KIND_INSTRUCTION)}["CLAUDE.md"]

    (project / "CLAUDE.md").write_text("project rules, revised " * 40, encoding="utf-8")
    ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
    )
    after = {r.name: r for r in store.select(kind=ac.KIND_INSTRUCTION)}["CLAUDE.md"]

    assert after.config_id == before.config_id
    assert after.content_hash != before.content_hash


# --- MCP servers and hooks --------------------------------------------------

def test_mcp_servers_are_one_record_per_declaring_file(tmp_path, catalog):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"apollo": {"command": "npx", "args": ["a"]}}}),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    (home).mkdir()
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"exa": {"type": "http", "url": "https://x"}}}),
        encoding="utf-8",
    )
    store = ac.InMemoryAgentConfigStore()
    at = ac.ingest_agent_config(
        store, roots=[root], claude_home=home, kinds=(ac.KIND_MCP_SERVER,),
    )
    rows = store.select(kind=ac.KIND_MCP_SERVER, seen_at=at)

    assert [r.name for r in rows] == ["exa", "apollo"]  # globals first, then roots
    assert rows[0].scope == ac.SCOPE_GLOBAL
    assert rows[1].scope == ac.SCOPE_PROJECT
    assert rows[1].root == str(root)
    # The launch spec travels with the record so a measurement never has to
    # re-open the config file.
    assert rows[1].detail["command"] == "npx"
    assert rows[1].detail["args"] == ["a"]
    # The spec hash IS the content hash: it is what a cached measurement is
    # valid against.
    assert rows[1].content_hash == rows[1].detail["spec_hash"]


def test_hooks_are_ingested_per_command(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [
                    {"type": "command", "command": "echo one"},
                    {"type": "command", "command": "echo two"},
                ]},
            ],
        },
    }), encoding="utf-8")
    store = ac.InMemoryAgentConfigStore()
    at = ac.ingest_agent_config(store, claude_home=home, kinds=(ac.KIND_HOOK,))
    rows = store.select(kind=ac.KIND_HOOK, seen_at=at)

    # Two hooks on ONE event and matcher must not collapse onto one row.
    assert [r.name for r in rows] == ["PreToolUse:Bash:0", "PreToolUse:Bash:1"]
    assert rows[0].detail["command"] == "echo one"
    assert rows[0].subkind == "PreToolUse"


# --- The store contract -----------------------------------------------------

def test_reading_back_preserves_scan_order(catalog, project):
    """A store round-trip must not reorder an analyzer's enumeration.

    Three analyzers had order-dependent behaviour before this table existed
    (which config file wins as a fix target, which candidate sorts first), so a
    silent reorder would change conclusions while every test still passed.
    """
    records = ac.scan_instruction_files(roots=[project], catalog=catalog)
    store = ac.InMemoryAgentConfigStore()
    store.upsert(records)
    assert [r.path for r in store.select(kind=ac.KIND_INSTRUCTION)] == [
        r.path for r in records
    ]


def test_a_persistent_store_does_not_leak_a_previous_runs_root(tmp_path, catalog):
    """THE reason ``seen_at`` exists.

    A persistent store holds every root ever scanned. An analyzer reading the
    whole table would price a repo the current window never touched — the
    stored table has to answer the same question the live walk answered, not a
    broader one.
    """
    old = tmp_path / "old"
    (old).mkdir()
    (old / "CLAUDE.md").write_text("old " * 60, encoding="utf-8")
    new = tmp_path / "new"
    (new).mkdir()
    (new / "CLAUDE.md").write_text("new " * 60, encoding="utf-8")

    conn, store = _duck_store(tmp_path)
    try:
        ac.ingest_agent_config(
            store, roots=[old], kinds=(ac.KIND_INSTRUCTION,),
            include_global=False, catalog=catalog,
        )
        at = ac.ingest_agent_config(
            store, roots=[new], kinds=(ac.KIND_INSTRUCTION,),
            include_global=False, catalog=catalog,
        )
        scoped = store.select(kind=ac.KIND_INSTRUCTION, seen_at=at)
        assert {r.root for r in scoped} == {str(new)}
        # Both are still on the table — the history is kept, it is just not
        # what a scoped read returns.
        assert len(store.select(kind=ac.KIND_INSTRUCTION)) == 2
    finally:
        conn.close()


def test_the_duckdb_store_round_trips_every_field(tmp_path, catalog, project):
    conn, store = _duck_store(tmp_path)
    try:
        at = ac.ingest_agent_config(
            store, roots=[project], kinds=(ac.KIND_INSTRUCTION,),
            include_global=False, catalog=catalog,
        )
        memory = ac.InMemoryAgentConfigStore()
        ac.ingest_agent_config(
            memory, roots=[project], kinds=(ac.KIND_INSTRUCTION,),
            include_global=False, catalog=catalog, seen_at=at,
        )
        stored = store.select(kind=ac.KIND_INSTRUCTION, seen_at=at)
        expected = memory.select(kind=ac.KIND_INSTRUCTION, seen_at=at)
        assert [(r.name, r.size_bytes, r.tokens, r.content_hash, r.subkind, r.root)
                for r in stored] == [
            (r.name, r.size_bytes, r.tokens, r.content_hash, r.subkind, r.root)
            for r in expected
        ]
    finally:
        conn.close()


def test_a_measurement_survives_a_rescan_but_not_a_content_change(tmp_path):
    """Measuring an MCP server means STARTING it, so the answer has to outlive
    the scan that found the server — but never outlive the spec it describes."""
    for store in (ac.InMemoryAgentConfigStore(), _duck_store(tmp_path)[1]):
        record = ac.ConfigRecord(
            kind=ac.KIND_MCP_SERVER, scope=ac.SCOPE_GLOBAL, root="", name="apollo",
            path="/tmp/.claude.json", content_hash="h1",
            detail={"command": "x", "spec_hash": "h1"},
        )
        store.upsert([record])
        store.record_measurement(
            record.config_id, tokens=1234, status=ac.MEASURE_OK, at=utcnow(),
            extra={"deferred_tokens": 40, "tool_count": 3},
        )

        store.upsert([record])  # rescan, unchanged
        kept = store.measurement_for(record.config_id)
        assert kept.tokens == 1234
        assert kept.extra["tool_count"] == 3

        changed = ac.ConfigRecord(
            kind=ac.KIND_MCP_SERVER, scope=ac.SCOPE_GLOBAL, root="", name="apollo",
            path="/tmp/.claude.json", content_hash="h2",
            detail={"command": "y", "spec_hash": "h2"},
        )
        store.upsert([changed])
        assert store.measurement_for(changed.config_id).tokens is None


def test_an_unrecorded_measurement_is_not_a_zero():
    """"Never measured" and "measured and found empty" must stay distinct.

    A store that answered 0 for both would let a consumer bill an unmeasured
    server nothing while believing it had a measurement — the quiet half of the
    defect this whole surface exists to close.
    """
    store = ac.InMemoryAgentConfigStore()
    row = store.measurement_for("nothing-here")
    assert row.tokens is None
    assert row.status == ""
    assert row.at is None


def test_store_for_degrades_to_memory_without_a_connection():
    assert isinstance(ac.store_for(None), ac.InMemoryAgentConfigStore)


def test_the_migration_creates_the_table_and_the_self_heal_recreates_it(tmp_path):
    """Migration 22 plus its ``EXPECTED_TABLES`` entry.

    The second half is what matters: this repo has renumbered migrations
    before, so a version recorded applied under an older definition never
    re-runs and its CREATE TABLE silently never lands. The self-heal is what
    makes that recoverable.
    """
    from tokenjam.core.db import ensure_expected_tables, missing_expected_tables

    conn = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        run_migrations(conn)
        assert "agent_config_files" not in missing_expected_tables(conn)
        conn.execute("DROP TABLE agent_config_files")
        assert "agent_config_files" in missing_expected_tables(conn)
        ensure_expected_tables(conn)
        assert "agent_config_files" not in missing_expected_tables(conn)
    finally:
        conn.close()


def test_last_seen_moves_on_a_rescan(catalog, project):
    store = ac.InMemoryAgentConfigStore()
    first = ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
        seen_at=utcnow() - timedelta(days=1),
    )
    second = ac.ingest_agent_config(
        store, roots=[project], kinds=(ac.KIND_INSTRUCTION,), catalog=catalog,
    )
    assert second > first
    assert store.select(kind=ac.KIND_INSTRUCTION, seen_at=first) == []
    assert store.select(kind=ac.KIND_INSTRUCTION, seen_at=second)
