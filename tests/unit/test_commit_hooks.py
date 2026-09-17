"""The commit trailer hook, the notes hook, `tj init --hooks / --notes /
--enforce`, the doctor check and the uninstall round trip (ledger W3;
contracts §3, §5, §9).

The hook is a POSIX sh script git runs, so it is proved the way production
reaches it: installed into a real repository in `tmp_path` and exercised by
real `git commit` invocations, with `HOME` pointed at a fixture home
(Critical Rule 47 in `tokenjam/CLAUDE.md`: the hook reads `$HOME`, and so
does `Path.home()`).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import tokenjam.core.config as cfg_mod
from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_doctor import _check_commit_hook
from tokenjam.cli.cmd_onboard import cmd_onboard
from tokenjam.cli.cmd_uninstall import cmd_uninstall
from tokenjam.cli.ledger_hooks import (
    SUBSCRIPTION_TRAFFIC_SENTENCE,
    hook_summary_line,
    install_repo_hooks,
    remove_all_repo_hooks,
)
from tokenjam.cli.main import cli
from tokenjam.core import commit_hooks
from tokenjam.core.commit_hooks import (
    HOOK_BLOCK_END,
    HOOK_BLOCK_START,
    STATE_ABSENT,
    STATE_CURRENT,
    STATE_DAMAGED,
    STATE_STALE,
    active_session_for,
    active_sessions_path,
    build_commit_note,
    hook_state,
    install_hook_block,
    read_active_sessions,
    record_active_session,
    remove_hook_block,
    render_hook_block,
    resolve_hooks_location,
)
from tokenjam.core.config import ProviderBudget, TjConfig, write_config

from tests.unit.test_advertised_commands_are_invocable import advertised_commands, assert_invocable

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


# --- Fixtures ---------------------------------------------------------------------

def _git(repo: Path, *args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    e.update({"GIT_AUTHOR_EMAIL": "dev@example.com", "GIT_COMMITTER_EMAIL": "dev@example.com",
              "GIT_AUTHOR_NAME": "Dev", "GIT_COMMITTER_NAME": "Dev", "GIT_EDITOR": "true"})
    if env:
        e.update(env)
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check, env=e)


def _body(repo: Path, ref: str = "HEAD") -> str:
    return _git(repo, "log", "-1", "--format=%B", ref).stdout.rstrip("\n") + "\n"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    return h


@pytest.fixture
def repo(tmp_path, home) -> Path:
    path = tmp_path / "widgets"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "dev@example.com")
    _git(path, "config", "user.name", "Dev")
    (path / "README").write_text("hello\n")
    _git(path, "add", "README")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _commit(repo: Path, message: str | None, *, env: dict | None = None, extra: list[str] | None = None) -> str:
    f = repo / "f.txt"
    f.write_text(f.read_text() + "x\n" if f.exists() else "x\n")
    _git(repo, "add", "f.txt")
    args = ["commit", "-q", *(extra or [])]
    if message is not None:
        args += ["-m", message]
    _git(repo, *args, env=env)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _record(repo: Path, sid: str = "sess-abc", *, age_s: int = 0) -> None:
    at = datetime.now(tz=timezone.utc) - timedelta(seconds=age_s)
    assert record_active_session(str(repo), sid, now=at)


# --- The hook, run by git --------------------------------------------------------

def test_hook_appends_the_trailer_for_a_plain_shell_commit(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-abc")
    _commit(repo, "feat: thing")
    assert _body(repo) == "feat: thing\n\nTokenJam-Session: sess-abc\n"


def test_hook_prefers_the_env_var_claude_code_exports(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "from-file")
    _commit(repo, "feat", env={"CLAUDE_CODE_SESSION_ID": "from-env"})
    assert "TokenJam-Session: from-env" in _body(repo)
    assert "from-file" not in _body(repo)


def test_hook_keeps_every_existing_trailer_and_is_idempotent(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-abc")
    msg = "feat\n\nBody.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01X"
    sha = _commit(repo, msg)
    body = _body(repo, sha)
    assert body == msg + "\nTokenJam-Session: sess-abc\n"
    # Amend, still inside the session: the trailer stays exactly once.
    _git(repo, "commit", "-q", "--amend", "--no-edit")
    assert _body(repo).count("TokenJam-Session:") == 1
    assert _body(repo) == body


def test_hook_leaves_a_trailer_naming_another_session_alone(repo):
    """Merge / squash / amend messages carry the trailer of the commit they
    came from; that one wins, never the current session."""
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-now")
    _commit(repo, "feat\n\nTokenJam-Session: sess-earlier")
    assert _body(repo) == "feat\n\nTokenJam-Session: sess-earlier\n"


def test_hook_leaves_a_merge_message_with_a_trailer_untouched(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-now")
    _git(repo, "checkout", "-q", "-b", "topic")
    _commit(repo, "topic work\n\nTokenJam-Session: sess-topic")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "main work")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge topic\n\nTokenJam-Session: sess-topic", "topic")
    assert _body(repo) == "Merge topic\n\nTokenJam-Session: sess-topic\n"


def test_hook_ignores_a_stale_active_session(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-old", age_s=31 * 60)
    _commit(repo, "feat")
    assert "TokenJam-Session" not in _body(repo)


def test_hook_is_a_no_op_with_no_record_and_never_fails_the_commit(repo):
    install_repo_hooks(str(repo), notes=False)
    assert not active_sessions_path().exists()
    _commit(repo, "feat")
    assert _body(repo) == "feat\n"
    # A garbage record is the same: exit 0, message untouched.
    active_sessions_path().parent.mkdir(parents=True, exist_ok=True)
    active_sessions_path().write_text("not json {")
    _commit(repo, "feat2")
    assert _body(repo) == "feat2\n"


def test_hook_matches_the_worktree_root_for_a_session_in_a_subdirectory(repo):
    install_repo_hooks(str(repo), notes=False)
    (repo / "pkg").mkdir()
    assert record_active_session(str(repo / "pkg"), "sess-sub")
    _commit(repo, "feat")
    assert "TokenJam-Session: sess-sub" in _body(repo)


def test_hook_lays_an_editor_message_out_like_git_commit_s(repo, tmp_path):
    """No `-m`: the message file is comments only when the hook runs. The
    trailer goes two lines down so the subject the user types stays its own
    paragraph, exactly where `git commit -s` puts a sign-off."""
    install_repo_hooks(str(repo), notes=False)
    _record(repo, "sess-abc")
    capture = tmp_path / "editor.sh"
    capture.write_text("#!/bin/sh\ncp \"$1\" \"$1.seen\"\nprintf 'Typed subject\\n' | cat - \"$1\" > \"$1.new\" && mv \"$1.new\" \"$1\"\n")
    capture.chmod(0o755)
    _commit(repo, None, env={"GIT_EDITOR": str(capture)})
    seen = (repo / ".git" / "COMMIT_EDITMSG.seen").read_text()
    assert seen.startswith("\n\nTokenJam-Session: sess-abc\n\n")
    assert _body(repo) == "Typed subject\n\nTokenJam-Session: sess-abc\n"


def test_hook_rejects_a_session_id_it_would_not_trust(repo):
    install_repo_hooks(str(repo), notes=False)
    _commit(repo, "feat", env={"CLAUDE_CODE_SESSION_ID": "x; rm -rf /"})
    assert "TokenJam-Session" not in _body(repo)


def test_hook_runs_in_under_a_second(repo):
    install_repo_hooks(str(repo), notes=False)
    _record(repo)
    started = datetime.now()
    _commit(repo, "feat")
    assert (datetime.now() - started).total_seconds() < 1.0


# --- The managed block (Critical Rule 21) -----------------------------------------

def test_a_users_own_hook_is_preserved_around_the_block_and_still_runs(repo):
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/bash\n# mine\necho 'Mine: yes' >> \"$1\"\n")
    hook.chmod(0o755)
    assert install_hook_block("prepare-commit-msg", hook) == "updated"
    text = hook.read_text()
    assert text.startswith("#!/bin/bash\n" + HOOK_BLOCK_START["prepare-commit-msg"])
    assert text.endswith("# mine\necho 'Mine: yes' >> \"$1\"\n")
    assert text.count(HOOK_BLOCK_START["prepare-commit-msg"]) == 1
    _record(repo, "sess-abc")
    _commit(repo, "feat")
    assert "TokenJam-Session: sess-abc" in _body(repo)
    assert "Mine: yes" in _body(repo)
    # Remove: the user's hook is byte-for-byte what it was.
    assert remove_hook_block("prepare-commit-msg", hook) == "removed"
    assert hook.read_text() == "#!/bin/bash\n# mine\necho 'Mine: yes' >> \"$1\"\n"
    assert hook.stat().st_mode & stat.S_IXUSR


def test_reinstall_replaces_a_stale_block_and_never_appends_a_second(repo):
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.parent.mkdir(parents=True, exist_ok=True)
    stale = (f"#!/bin/sh\n{HOOK_BLOCK_START['prepare-commit-msg']}\n# an older build's body\n"
             f"{HOOK_BLOCK_END['prepare-commit-msg']}\necho user >/dev/null\n")
    hook.write_text(stale)
    assert hook_state("prepare-commit-msg", hook) == STATE_STALE
    assert install_hook_block("prepare-commit-msg", hook) == "updated"
    text = hook.read_text()
    assert text.count(HOOK_BLOCK_START["prepare-commit-msg"]) == 1
    assert "an older build's body" not in text
    assert text.endswith("echo user >/dev/null\n")
    assert install_hook_block("prepare-commit-msg", hook) == "kept"
    assert hook_state("prepare-commit-msg", hook) == STATE_CURRENT


def test_a_block_at_eof_without_a_trailing_newline_is_still_stripped(tmp_path):
    hook = tmp_path / "prepare-commit-msg"
    hook.write_text("#!/bin/sh\n" + render_hook_block("prepare-commit-msg").rstrip("\n"))
    assert remove_hook_block("prepare-commit-msg", hook) == "deleted"
    assert not hook.exists()


def test_a_damaged_block_is_reported_not_silently_kept(tmp_path):
    hook = tmp_path / "prepare-commit-msg"
    hook.write_text(f"#!/bin/sh\n{HOOK_BLOCK_START['prepare-commit-msg']}\nhalf\n")
    assert hook_state("prepare-commit-msg", hook) == STATE_DAMAGED


def test_a_non_shell_hook_is_never_edited(repo):
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/usr/bin/env python3\nprint('mine')\n")
    hook.chmod(0o755)
    report = install_repo_hooks(str(repo), notes=False)
    assert report.outcomes == {"prepare-commit-msg": "skipped"}
    assert hook.read_text() == "#!/usr/bin/env python3\nprint('mine')\n"


def test_install_and_uninstall_leave_zero_residue(repo, home):
    report = install_repo_hooks(str(repo), notes=True)
    assert report.outcomes == {"prepare-commit-msg": "written", "post-commit": "written"}
    hooks = repo / ".git" / "hooks"
    assert (hooks / "prepare-commit-msg").stat().st_mode & stat.S_IXUSR
    assert json.loads((home / ".tj" / "commit_hooks.json").read_text()) == {"hooks_dirs": [str(hooks)]}
    removed = remove_all_repo_hooks(str(home))  # from OUTSIDE the repo: the registry finds it
    assert removed == {str(hooks): {"prepare-commit-msg": "deleted", "post-commit": "deleted"}}
    assert not (hooks / "prepare-commit-msg").exists()
    assert not (hooks / "post-commit").exists()
    assert remove_all_repo_hooks(str(repo)) == {}


def test_a_tracked_hooks_path_is_refused(repo):
    (repo / ".husky").mkdir()
    _git(repo, "config", "core.hooksPath", ".husky")
    loc = resolve_hooks_location(str(repo))
    assert loc is not None and loc.tracked
    report = install_repo_hooks(str(repo), notes=False)
    assert report.refusal is not None and "tracked" in report.refusal
    assert not list((repo / ".husky").iterdir())


def test_a_global_hooks_path_outside_any_worktree_is_an_opt_in(repo, home):
    global_hooks = home / ".git-hooks"
    global_hooks.mkdir()
    _git(repo, "config", "core.hooksPath", str(global_hooks))
    report = install_repo_hooks(str(repo), notes=False)
    assert report.location is not None and not report.location.tracked
    assert report.location.hooks_dir == global_hooks.resolve()
    assert report.outcomes == {"prepare-commit-msg": "written"}
    _record(repo, "sess-abc")
    _commit(repo, "feat")
    assert "TokenJam-Session: sess-abc" in _body(repo)


def test_outside_a_repo_nothing_is_written(tmp_path, home):
    report = install_repo_hooks(str(tmp_path), notes=False)
    assert report.location is None and report.refusal == "not a git repository"


# --- The active-session record ------------------------------------------------------

def test_record_is_atomic_pruned_and_keyed_by_worktree_root(repo, home):
    path = active_sessions_path()
    assert record_active_session(str(repo), "old", now=NOW - timedelta(hours=7))
    assert record_active_session(str(repo / "sub"), "new", now=NOW) is True
    data = read_active_sessions(path)
    assert list(data) == [os.path.realpath(repo)]
    assert data[os.path.realpath(repo)]["session_id"] == "new"
    assert data[os.path.realpath(repo)]["updated_epoch"] == int(NOW.timestamp())
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".active_sessions.")]
    # One entry per line, key first: the sh hook depends on this layout.
    lines = path.read_text().splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    assert lines[1].startswith(f'  "{os.path.realpath(repo)}": {{"session_id": "new", ')


def test_record_refuses_bad_input_and_never_raises(home, tmp_path):
    assert record_active_session(None, "x") is False
    assert record_active_session(str(tmp_path), "") is False
    assert record_active_session(str(tmp_path), "bad id") is False
    assert record_active_session(str(tmp_path), "ok", path=tmp_path / "nope" / "dir" / "f.json") is True
    unwritable = tmp_path / "file-not-dir"
    unwritable.write_text("")
    assert record_active_session(str(tmp_path), "ok", path=unwritable / "f.json") is False


def test_active_session_for_applies_the_hooks_freshness_window(repo):
    assert record_active_session(str(repo), "s", now=NOW - timedelta(minutes=29))
    assert active_session_for(str(repo), now=NOW)["session_id"] == "s"
    assert active_session_for(str(repo), now=NOW + timedelta(minutes=2)) is None


def test_statusline_writes_the_record(repo, home):
    from tokenjam.cli.cmd_statusline import cmd_statusline

    payload = {"session_id": "abc-123", "cwd": str(repo / "src"), "model": "Opus"}
    result = CliRunner().invoke(cmd_statusline, input=json.dumps(payload), obj={})
    assert result.exit_code == 0
    assert read_active_sessions()[os.path.realpath(repo)]["session_id"] == "abc-123"


# --- The §5 note --------------------------------------------------------------------

def test_build_commit_note_is_the_contract_object():
    from tests.factories import make_session

    s = make_session(session_id="s1", agent_id="claude-code-widgets", plan_tier="max_5x",
                     total_cost_usd=1.25)
    s.dominant_model = "claude-opus-4-1"
    assert build_commit_note(s) == {
        "v": 1, "session_id": "s1", "tool": "claude-code", "model": "claude-opus-4-1",
        "cost_usd": 1.25, "pricing_mode": "subscription",
        "confidence": "deterministic", "source": "trailer_session",
    }


def test_matcher_reads_our_note_at_deterministic(repo, home, monkeypatch):
    from dataclasses import replace

    from tokenjam.core import repo_context, shipped
    from tokenjam.core.db import InMemoryBackend
    from tests.factories import make_session

    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    repo_context.clear_caches()
    sha = _commit(repo, "feat")
    committed = datetime.fromtimestamp(int(_git(repo, "log", "-1", "--format=%ct").stdout), tz=timezone.utc)
    db = InMemoryBackend()
    try:
        db.upsert_session(replace(
            make_session(session_id="s1", agent_id="claude-code-widgets", started_at=committed - timedelta(minutes=5),
                         status="completed", total_cost_usd=1.0),
            ended_at=committed + timedelta(minutes=5), repo_root=str(repo), branch_start="main",
        ))
        note = build_commit_note(db.get_session("s1"))
        assert commit_hooks.write_commit_note(str(repo), sha, note)
        shipped.match_sessions_to_commits(db, now=committed + timedelta(minutes=10))
        rows = db.conn.execute(
            "SELECT commit_sha, confidence, source FROM session_commits WHERE session_id = 's1'").fetchall()
        assert rows == [(sha, "deterministic", "git_note")]
    finally:
        db.close()
        repo_context.clear_caches()


def test_tj_commit_note_writes_the_note_for_a_trailered_commit(repo, home, monkeypatch):
    from dataclasses import replace

    from tokenjam.core.db import InMemoryBackend
    from tests.factories import make_session

    sha = _commit(repo, "feat\n\nTokenJam-Session: s1")
    db = InMemoryBackend()
    try:
        db.upsert_session(replace(
            make_session(session_id="s1", agent_id="claude-code-widgets", plan_tier="api", total_cost_usd=0.5),
            dominant_model="claude-sonnet-4"))
        monkeypatch.chdir(repo)
        with patch("tokenjam.cli.main.load_config", return_value=TjConfig(version="1")), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            result = CliRunner().invoke(cli, ["--json", "commit-note"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["outcome"] == "written"
        stored = json.loads(_git(repo, "notes", "--ref=tokenjam", "show", sha).stdout)
        assert stored == payload["note"]
        assert stored["cost_usd"] == 0.5 and stored["pricing_mode"] == "api"
        assert stored["model"] == "claude-sonnet-4"
        # A commit without the trailer, or an unknown session: exit 0, no note.
        _commit(repo, "plain")
        with patch("tokenjam.cli.main.load_config", return_value=TjConfig(version="1")), \
             patch("tokenjam.cli.main.open_db", return_value=db):
            result = CliRunner().invoke(cli, ["--json", "commit-note"])
        assert result.exit_code == 0 and json.loads(result.output)["outcome"] == "no_trailer"
    finally:
        db.close()


def test_api_backend_get_session_maps_the_detail_payload():
    from tokenjam.core.api_backend import ApiBackend

    backend = ApiBackend("http://127.0.0.1:1")
    with patch.object(backend, "_get", return_value={
        "session": {"session_id": "s1", "agent_id": "claude-code-x", "plan_tier": "max_20x",
                    "total_cost_usd": 2.5, "started_at": "2026-09-16T10:00:00+00:00"},
        "model_mix": [{"model": "claude-opus-4-1", "calls": 3}, {"model": "claude-haiku", "calls": 1}],
    }):
        s = backend.get_session("s1")
    assert s is not None and s.dominant_model == "claude-opus-4-1" and s.pricing_mode == "subscription"
    assert build_commit_note(s)["cost_usd"] == 2.5
    with patch.object(backend, "_get", side_effect=ValueError("404")):
        assert backend.get_session("nope") is None


# --- `tj init` (Critical Rule 44) ----------------------------------------------------

def _global_config(home: Path, monkeypatch) -> Path:
    path = home / ".config" / "tj" / "config.toml"
    path.parent.mkdir(parents=True)
    write_config(TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="api")}), path)
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path(".tj/config.toml"), path])
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    return path


def test_tj_init_hooks_notes_enforce_parse():
    for args in (["init", "--hooks"], ["init", "--notes"], ["init", "--enforce"],
                 ["init", "--hooks", "--enforce"], ["onboard", "--hooks"], ["commit-note"]):
        result = CliRunner().invoke(cli, [*args, "--help"])
        assert result.exit_code == 0, (args, result.output)


def test_tj_init_hooks_with_an_existing_config_installs_without_the_wizard(repo, home, monkeypatch):
    _global_config(home, monkeypatch)
    monkeypatch.chdir(repo)
    wizard = MagicMock()
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._run_onboard_wizard", wizard)
    result = CliRunner().invoke(cmd_onboard, ["--notes"], obj={})
    assert result.exit_code == 0, result.output
    wizard.assert_not_called()
    assert hook_state("prepare-commit-msg", repo / ".git" / "hooks" / "prepare-commit-msg") == STATE_CURRENT
    assert hook_state("post-commit", repo / ".git" / "hooks" / "post-commit") == STATE_CURRENT
    out = " ".join(result.output.split())
    assert "prepare-commit-msg hook installed" in out
    assert "refs/notes/tokenjam" in out
    for command in advertised_commands(result.output):
        assert_invocable(command)
    # Again: idempotent, and says so.
    result = CliRunner().invoke(cmd_onboard, ["--hooks"], obj={})
    assert "already current" in result.output


def test_tj_init_hooks_without_a_config_runs_the_wizard_first(repo, home, monkeypatch):
    monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [Path(".tj/config.toml"), home / "nope.toml"])
    monkeypatch.delenv("TJ_CONFIG", raising=False)
    monkeypatch.chdir(repo)
    wizard = MagicMock()
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._run_onboard_wizard", wizard)
    result = CliRunner().invoke(cmd_onboard, ["--hooks"], obj={})
    assert result.exit_code == 0, result.output
    wizard.assert_called_once()
    assert (repo / ".git" / "hooks" / "prepare-commit-msg").exists()


def test_tj_init_enforce_enables_the_proxy_and_prints_the_privacy_sentence(repo, home, monkeypatch):
    from tokenjam.core.config import load_config

    config_path = _global_config(home, monkeypatch)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(cmd_onboard, ["--hooks", "--enforce"], obj={})
    assert result.exit_code == 0, result.output
    assert load_config(str(config_path)).proxy.enabled is True
    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert settings["env"]["ANTHROPIC_BASE_URL"].startswith("http://")
    out = " ".join(result.output.split())
    assert "Enforcement" in out and "suggest mode" in out
    assert " ".join(SUBSCRIPTION_TRAFFIC_SENTENCE.split()) in out
    assert "prepare-commit-msg hook installed" in out
    for command in advertised_commands(result.output):
        assert_invocable(command)


def test_hook_summary_line_names_every_state_and_its_commands(repo, tmp_path):
    assert "not installed" in hook_summary_line(str(repo))
    assert "not a git repository" in hook_summary_line(str(tmp_path))
    install_repo_hooks(str(repo), notes=True)
    assert hook_summary_line(str(repo)) == "Commit hook: installed, git notes on"
    for state_line in (hook_summary_line(str(repo)), hook_summary_line(str(tmp_path))):
        for command in advertised_commands(state_line):
            assert_invocable(command)


# --- `tj doctor` -----------------------------------------------------------------------

def test_doctor_commit_hook_check_reports_each_state(repo, tmp_path, home, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    assert _check_commit_hook(str(tmp_path))["level"] == "info"
    absent = _check_commit_hook(str(repo))
    assert absent["level"] == "info" and "tj init --hooks" in absent["message"]
    install_repo_hooks(str(repo), notes=False)
    no_record = _check_commit_hook(str(repo))
    assert no_record["level"] == "ok" and "no active-session record yet" in no_record["message"]
    _record(repo, "sess-abc", age_s=40 * 60)
    stale = _check_commit_hook(str(repo))
    assert stale["level"] == "ok" and "would carry no trailer" in stale["message"]
    _record(repo, "sess-abc")
    fresh = _check_commit_hook(str(repo))
    assert fresh["level"] == "ok" and "sess-abc" in fresh["message"]
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.write_text(hook.read_text().replace(HOOK_BLOCK_END["prepare-commit-msg"], ""))
    assert _check_commit_hook(str(repo))["level"] == "warning"
    assert hook_state("prepare-commit-msg", hook) == STATE_DAMAGED
    install_repo_hooks(str(repo), notes=False)
    assert _check_commit_hook(str(repo))["level"] == "ok"
    remove_all_repo_hooks(str(repo))
    assert hook_state("prepare-commit-msg", hook) == STATE_ABSENT


# --- `tj uninstall` ---------------------------------------------------------------------

def test_uninstall_removes_the_hook_blocks_and_keeps_the_notes_ref(repo, home, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(home)  # NOT the repo: the registry has to find it
    monkeypatch.setattr("tokenjam.cli.cmd_stop.cmd_stop", MagicMock())
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    install_repo_hooks(str(repo), notes=True)
    hook = repo / ".git" / "hooks" / "prepare-commit-msg"
    hook.write_text(hook.read_text() + "echo mine >/dev/null\n")
    sha = _commit(repo, "feat")
    assert commit_hooks.write_commit_note(str(repo), sha, {"v": 1, "session_id": "s1"})

    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        result = CliRunner().invoke(cmd_uninstall, ["--yes"])
    assert result.exit_code == 0, result.output
    assert "Removed tj hook block" in result.output
    assert hook.read_text() == "#!/bin/sh\necho mine >/dev/null\n"
    assert not (repo / ".git" / "hooks" / "post-commit").exists()
    assert json.loads(_git(repo, "notes", "--ref=tokenjam", "show", sha).stdout) == {"v": 1, "session_id": "s1"}
