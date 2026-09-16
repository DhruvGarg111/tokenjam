"""Repo context + developer identity on every session (migration 23; shipped-
value ledger contracts §3 / §4).

Three write paths land on the same eight nullable `sessions` columns and this
module proves each through the path a user actually reaches: the Claude Code
backfill over a fixture transcript carrying `cwd` + `gitBranch`, the Codex
rollout adapter, and the live `IngestPipeline` for a span whose producer
stamped the §3 attributes. Plus the invariant that makes a re-run safe: a
later write fills NULLs and never overwrites a value with NULL.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.core import repo_context
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import EXPECTED_ADDITIVE_COLUMNS, DuckDBBackend, InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.ingest_adapters.codex import ingest_codex
from tokenjam.core.models import SESSION_CONTEXT_FIELDS, SessionContext
from tokenjam.core.repo_context import developer_id_for
from tokenjam.otel.semconv import ResourceAttributes, TjAttributes

from tests.factories import make_llm_span, make_session

EMAIL = "dev@example.com"
_CONTEXT_SELECT = ", ".join(SESSION_CONTEXT_FIELDS)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A real checkout the fixture transcripts point their `cwd` at."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    repo_context.clear_caches()
    path = tmp_path / "widgets"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", EMAIL)
    _git(path, "config", "user.name", "Dev")
    _git(path, "remote", "add", "origin", "https://github.com/Acme/widgets.git")
    (path / "f").write_text("x")
    _git(path, "add", "f")
    _git(path, "commit", "-q", "-m", "init")
    yield path
    repo_context.clear_caches()


def _context_row(db, session_id: str) -> SessionContext:
    row = db.conn.execute(
        f"SELECT {_CONTEXT_SELECT} FROM sessions WHERE session_id = $1", [session_id],
    ).fetchone()
    assert row is not None, f"session {session_id} not written"
    return SessionContext(*row)


# --- Claude Code transcript fixtures ------------------------------------------------

def _assistant(uuid: str, ts: str, session_id: str, cwd: str, branch: str | None) -> dict:
    rec = {
        "type": "assistant", "uuid": uuid, "timestamp": ts, "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    }
    if branch is not None:
        rec["gitBranch"] = branch
    return rec


def _write_transcript(root: Path, session_id: str, records: list[dict]) -> Path:
    project_dir = root / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


# --- Migration ------------------------------------------------------------------------

def test_migration_23_columns_are_declared_for_self_heal():
    declared = {(t, c) for t, c, _ in EXPECTED_ADDITIVE_COLUMNS}
    for column in SESSION_CONTEXT_FIELDS:
        assert ("sessions", column) in declared, column


def test_columns_round_trip_through_a_reopened_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    db_path = tmp_path / "t.duckdb"
    session = replace(
        make_session(session_id="rt-1"),
        repo_remote="https://github.com/Acme/widgets", repo_root="/w",
        branch_start="main", branch_end="feat/a",
        head_sha_start="aaa", head_sha_end="bbb",
        developer_id=developer_id_for(EMAIL), user_email=EMAIL,
    )
    db = DuckDBBackend(StorageConfig(path=str(db_path)))
    try:
        db.upsert_session(session)
    finally:
        db.close()
    db = DuckDBBackend(StorageConfig(path=str(db_path)))
    try:
        got = db.get_session("rt-1")
        assert got is not None
        assert got.context == session.context
    finally:
        db.close()


def test_upsert_fills_nulls_and_never_overwrites_with_null():
    db = InMemoryBackend()
    try:
        base = make_session(session_id="fill-1")
        db.upsert_session(replace(base, repo_remote="https://github.com/Acme/widgets",
                                  branch_start="main", user_email=EMAIL))
        # A later write knowing only the end branch fills that and nothing else.
        db.upsert_session(replace(base, branch_end="feat/b"))
        ctx = _context_row(db, "fill-1")
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        assert ctx.branch_start == "main" and ctx.branch_end == "feat/b"
        assert ctx.user_email == EMAIL
        # A write carrying a DIFFERENT start value does not flip the stored one;
        # a NULL end value does not erase the stored end.
        db.upsert_session(replace(base, repo_remote="https://github.com/Other/x",
                                  branch_start="other", branch_end=None))
        ctx = _context_row(db, "fill-1")
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        assert ctx.branch_start == "main" and ctx.branch_end == "feat/b"
        # A newer end value wins: it describes the latest observed state.
        db.upsert_session(replace(base, branch_end="feat/c"))
        assert _context_row(db, "fill-1").branch_end == "feat/c"
    finally:
        db.close()


# --- Claude Code backfill ------------------------------------------------------------

def test_claude_code_backfill_populates_repo_branch_and_author(repo, tmp_path):
    root = tmp_path / "projects"
    _write_transcript(root, "cc-1", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-1", str(repo), "main"),
        _assistant("m2", "2026-09-01T10:05:00.000Z", "cc-1", str(repo), "feat/ledger"),
    ])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        ctx = _context_row(db, "cc-1")
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        assert ctx.repo_root == str(repo.resolve())
        assert ctx.branch_start == "main"            # first record
        assert ctx.branch_end == "feat/ledger"       # last record
        assert ctx.user_email == EMAIL
        assert ctx.developer_id == developer_id_for(EMAIL)
        # git today cannot name the commit the session started on.
        assert ctx.head_sha_start is None and ctx.head_sha_end is None
    finally:
        db.close()


def test_claude_code_backfill_re_run_is_idempotent_and_fills_a_null_row(repo, tmp_path):
    root = tmp_path / "projects"
    _write_transcript(root, "cc-2", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-2", str(repo), "main"),
    ])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        first = _context_row(db, "cc-2")
        assert first.repo_remote == "https://github.com/Acme/widgets"
        # Simulate a row written by a build that predates the columns.
        db.conn.execute(
            "UPDATE sessions SET repo_remote = NULL, repo_root = NULL, branch_start = NULL, "
            "developer_id = NULL, user_email = NULL WHERE session_id = 'cc-2'"
        )
        assert _context_row(db, "cc-2").repo_remote is None
        result = ingest_claude_code(db, root=root)
        assert result.spans_ingested == 0          # nothing new to insert
        assert _context_row(db, "cc-2") == first   # yet the context is back
    finally:
        db.close()


def test_claude_code_branch_endpoints_follow_timestamps_not_file_order(repo, tmp_path):
    """Claude Code replays and re-snapshots records on resume, so file order is
    not chronology. The branch the session started and ended on is the one at
    the earliest and latest DATED record; an undated record contributes no
    session time and must not name an endpoint either."""
    root = tmp_path / "projects"
    late = _assistant("m2", "2026-09-01T10:05:00.000Z", "cc-6", str(repo), "feat/late")
    early = _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-6", str(repo), "main")
    undated = _assistant("m3", "", "cc-6", str(repo), "feat/replayed")
    del undated["timestamp"]
    # Written out of order: latest first, then earliest, then an undated replay.
    _write_transcript(root, "cc-6", [late, early, undated])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        ctx = _context_row(db, "cc-6")
        assert ctx.branch_start == "main"
        assert ctx.branch_end == "feat/late"
    finally:
        db.close()


def test_subagent_transcripts_never_supply_branch_endpoints(repo, tmp_path):
    """A subagent file carries the PARENT session id and runs inside its
    window; merging its branches by write order would let a subagent that
    happened to be flushed last overwrite the real end branch."""
    root = tmp_path / "projects"
    _write_transcript(root, "cc-7", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-7", str(repo), "main"),
        _assistant("m2", "2026-09-01T10:09:00.000Z", "cc-7", str(repo), "feat/real-end"),
    ])
    sub_dir = root / "proj" / "subagents"
    sub_dir.mkdir(parents=True)
    sub = sub_dir / "agent-abc.jsonl"
    sub.write_text(json.dumps(
        _assistant("s1", "2026-09-01T10:04:00.000Z", "cc-7", str(repo), "feat/subagent")
    ) + "\n")
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        ctx = _context_row(db, "cc-7")
        assert ctx.branch_start == "main"
        assert ctx.branch_end == "feat/real-end"
    finally:
        db.close()


def test_claude_code_backfill_treats_a_detached_head_as_no_branch(repo, tmp_path):
    """Claude Code records `gitBranch: "HEAD"` on a detached checkout. That is
    not a branch, and storing it would let the next wave join on it."""
    root = tmp_path / "projects"
    _write_transcript(root, "cc-5", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-5", str(repo), "HEAD"),
        _assistant("m2", "2026-09-01T10:01:00.000Z", "cc-5", str(repo), "HEAD"),
    ])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        ctx = _context_row(db, "cc-5")
        assert ctx.branch_start is None and ctx.branch_end is None
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
    finally:
        db.close()


def test_claude_code_backfill_without_gitbranch_or_repo_leaves_nulls(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "projects"
    cwd = str(tmp_path / "gone")   # a cwd that no longer exists on disk
    _write_transcript(root, "cc-3", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-3", cwd, None),
    ])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        assert _context_row(db, "cc-3").is_empty()
    finally:
        db.close()


def test_claude_code_backfill_never_shells_out_for_tokenjams_own_invoke_cwd(
    tmp_path, monkeypatch,
):
    """The invoke-cwd transcript is excluded before the resolver runs at all."""
    from tokenjam.core.distill import INVOKE_CWD_DIRNAME
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    calls: list = []
    monkeypatch.setattr(repo_context, "_git", lambda args, cwd: calls.append(args))
    root = tmp_path / "projects"
    cwd = str(tmp_path / INVOKE_CWD_DIRNAME)
    Path(cwd).mkdir()
    _write_transcript(root, "cc-4", [
        _assistant("m1", "2026-09-01T10:00:00.000Z", "cc-4", cwd, "main"),
    ])
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=root)
        assert db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert calls == []
    finally:
        db.close()


# --- Codex rollout ---------------------------------------------------------------------

def _codex_rollout(root: Path, session_id: str, cwd: str, git: dict | None) -> Path:
    day = root / "2026" / "09" / "01"
    day.mkdir(parents=True, exist_ok=True)
    meta: dict = {"session_id": session_id, "timestamp": "2026-09-01T10:00:00.000Z", "cwd": cwd}
    if git is not None:
        meta["git"] = git
    records = [
        {"type": "session_meta", "timestamp": "2026-09-01T10:00:00.000Z", "payload": meta},
        {"type": "turn_context", "timestamp": "2026-09-01T10:00:01.000Z",
         "payload": {"model": "gpt-5"}},
        {"type": "event_msg", "timestamp": "2026-09-01T10:00:02.000Z",
         "payload": {"type": "token_count", "info": {
             "last_token_usage": {"input_tokens": 50, "output_tokens": 10,
                                  "cached_input_tokens": 0}}}},
    ]
    path = day / f"rollout-{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_codex_backfill_takes_branch_and_head_from_session_meta_git(repo, tmp_path):
    root = tmp_path / "codex"
    _codex_rollout(root, "cx-1", str(repo), {
        "commit_hash": "0123abcd", "branch": "feat/codex",
        "repository_url": "git@github.com:Acme/widgets.git",
    })
    db = InMemoryBackend()
    try:
        ingest_codex(db, root=root)
        ctx = _context_row(db, "cx-1")
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        assert ctx.repo_root == str(repo.resolve())
        assert ctx.branch_start == "feat/codex"
        assert ctx.head_sha_start == "0123abcd"
        assert ctx.user_email == EMAIL
        assert ctx.developer_id == developer_id_for(EMAIL)
    finally:
        db.close()


def test_codex_backfill_without_git_meta_falls_back_to_the_checkout_remote(repo, tmp_path):
    root = tmp_path / "codex"
    _codex_rollout(root, "cx-2", str(repo), None)
    db = InMemoryBackend()
    try:
        ingest_codex(db, root=root)
        ctx = _context_row(db, "cx-2")
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        assert ctx.branch_start is None and ctx.head_sha_start is None
    finally:
        db.close()


# --- Live ingest -----------------------------------------------------------------------

def _stamped_context() -> SessionContext:
    return SessionContext(
        repo_remote="https://github.com/Acme/widgets", repo_root="/w/widgets",
        branch_start="main", head_sha_start="abc123",
        developer_id=developer_id_for(EMAIL), user_email=EMAIL,
    )


def test_live_ingest_writes_stamped_context_onto_a_new_session():
    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=TjConfig(version="1"))
    try:
        pipeline.process(make_llm_span(agent_id="claude-code-w", session_id="live-1",
                                       session_context=_stamped_context()))
        assert _context_row(db, "live-1") == _stamped_context()
    finally:
        db.close()


def test_live_ingest_leaves_nulls_when_the_span_carries_nothing():
    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=TjConfig(version="1"))
    try:
        pipeline.process(make_llm_span(agent_id="claude-code-w", session_id="live-2"))
        assert _context_row(db, "live-2").is_empty()
    finally:
        db.close()


def test_live_ingest_fills_an_existing_session_from_a_later_span_and_tracks_the_end():
    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=TjConfig(version="1"))
    try:
        # First span: nothing stamped (e.g. a tool span that raced tj init).
        pipeline.process(make_llm_span(agent_id="claude-code-w", session_id="live-3"))
        assert _context_row(db, "live-3").is_empty()
        # Second span carries the context: start-side fills, end-side too.
        pipeline.process(make_llm_span(
            agent_id="claude-code-w", session_id="live-3",
            session_context=replace(_stamped_context(), branch_end="feat/a"),
        ))
        ctx = _context_row(db, "live-3")
        assert ctx.branch_start == "main" and ctx.branch_end == "feat/a"
        assert ctx.repo_remote == "https://github.com/Acme/widgets"
        # Third span: a moved branch end wins, a different start does not.
        pipeline.process(make_llm_span(
            agent_id="claude-code-w", session_id="live-3",
            session_context=SessionContext(branch_start="other", branch_end="feat/b"),
        ))
        ctx = _context_row(db, "live-3")
        assert ctx.branch_start == "main" and ctx.branch_end == "feat/b"
    finally:
        db.close()


def test_otlp_parser_reads_the_section_3_attributes_off_the_resource():
    from tokenjam.otel.otlp_parsing import extract_resource_attrs, parse_otlp_span

    resource_span = {"resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "claude-code-w"}},
        {"key": ResourceAttributes.USER_EMAIL, "value": {"stringValue": EMAIL}},
        {"key": ResourceAttributes.VCS_REPOSITORY_URL_FULL,
         "value": {"stringValue": "https://github.com/Acme/widgets"}},
        {"key": TjAttributes.REPO_ROOT, "value": {"stringValue": "/w/widgets"}},
        {"key": ResourceAttributes.VCS_REF_HEAD_NAME, "value": {"stringValue": "main"}},
    ]}}
    raw = {
        "spanId": "0011223344556677", "traceId": "00112233445566770011223344556677",
        "name": "gen_ai.llm.call", "kind": 3,
        "startTimeUnixNano": "1700000000000000000", "endTimeUnixNano": "1700000001000000000",
        "attributes": [
            {"key": "session.id", "value": {"stringValue": "otlp-1"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "claude-opus-4-7"}},
            {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "10"}},
        ],
    }
    span = parse_otlp_span(raw, extract_resource_attrs(resource_span))
    assert span.session_context == SessionContext(
        repo_remote="https://github.com/Acme/widgets", repo_root="/w/widgets",
        branch_start="main", developer_id=developer_id_for(EMAIL), user_email=EMAIL,
    )


# --- Surfaces ---------------------------------------------------------------------------

def test_sessions_api_returns_the_new_fields():
    import asyncio

    import httpx

    from tokenjam.api.app import create_app

    db = InMemoryBackend()
    try:
        db.upsert_session(replace(
            # Tokens > 0: /status's archive drops zero-signal sessions.
            make_session(session_id="api-1", agent_id="claude-code-w",
                         input_tokens=100, output_tokens=10, tool_call_count=1),
            repo_remote="https://github.com/Acme/widgets", repo_root="/w",
            branch_start="main", branch_end="feat/a", user_email=EMAIL,
            developer_id=developer_id_for(EMAIL),
        ))
        config = TjConfig(version="1")
        app = create_app(config=config, db=db,
                         ingest_pipeline=IngestPipeline(db=db, config=config))

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                listing = (await c.get("/api/v1/sessions")).json()
                detail = (await c.get("/api/v1/sessions/api-1")).json()
                status = (await c.get("/api/v1/status")).json()
            return listing, detail, status

        listing, detail, status = asyncio.run(go())
        row = listing["sessions"][0]
        for f in SESSION_CONTEXT_FIELDS:
            assert f in row
        assert row["repo"] == "Acme/widgets" and row["branch"] == "feat/a"
        assert row["repo_remote"] == "https://github.com/Acme/widgets"
        s = detail["session"]
        assert s["repo"] == "Acme/widgets" and s["branch"] == "feat/a"
        assert s["branch_start"] == "main" and s["user_email"] == EMAIL
        archived = [a for a in status["archived"] if a["session_id"] == "api-1"]
        assert archived and archived[0]["repo"] == "Acme/widgets"
        assert archived[0]["branch"] == "feat/a"
    finally:
        db.close()


def test_tj_status_shows_repo_and_branch(tmp_path, monkeypatch):
    from unittest.mock import patch

    from tokenjam.cli.main import cli

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    db = InMemoryBackend()
    try:
        db.upsert_session(replace(
            make_session(session_id="st-1", agent_id="claude-code-w", status="completed"),
            repo_remote="https://github.com/Acme/widgets", branch_start="main",
            branch_end="feat/a",
        ))
        config = TjConfig(version="1")
        with patch("tokenjam.cli.main.load_config", return_value=config), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            runner = CliRunner()
            human = runner.invoke(cli, ["status", "--agent", "claude-code-w"])
            assert human.exit_code == 0, human.output
            assert "Acme/widgets · feat/a" in human.output
            as_json = runner.invoke(cli, ["--json", "status", "--agent", "claude-code-w"])
            assert as_json.exit_code == 0, as_json.output
            payload = json.loads(as_json.output)
            agent = payload["agents"][0]
            assert agent["repo"] == "Acme/widgets" and agent["branch"] == "feat/a"
    finally:
        db.close()


def test_tj_init_is_the_onboard_command():
    """`tj init` is invocable and is literally the same Command as `tj
    onboard` (Critical Rule 44: an advertised name must resolve)."""
    import click

    from tokenjam.cli.main import cli

    ctx = click.Context(cli)
    assert cli.get_command(ctx, "init") is cli.get_command(ctx, "onboard")
    result = CliRunner().invoke(cli, ["init", "--help"])
    assert result.exit_code == 0, result.output
    assert "Set up tj" in result.output
    assert "--claude-code" in result.output


def test_lens_repo_branch_label_renders_only_what_is_known():
    """The Sessions view's `repo · branch` cell, executed under node (see
    test_lens_select_all_behaviour.py for why UI logic is run, not grepped)."""
    if shutil.which("node") is None:
        pytest.skip("node not available for JS evaluation")
    ui = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"
    html = ui.read_text(encoding="utf-8")
    start = html.index("function repoBranchLabel")
    end = html.index("\n}\n", start) + 3
    cases = [
        {"repo": "Acme/widgets", "branch": "main"},
        {"repo": "Acme/widgets"},
        {"branch": "feat/x"},
        {},
        None,
    ]
    script = (
        html[start:end]
        + "\nconsole.log(JSON.stringify(" + json.dumps(cases) + ".map(repoBranchLabel)));"
    )
    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout) == ["Acme/widgets · main", "Acme/widgets", "feat/x", "", ""]
