"""`core.repo_context`: the resolver behind every session's repo + identity
columns (shipped-value ledger, contracts §3).

Everything here runs against a throwaway git repo built in `tmp_path`. The
resolver refuses temp directories by design (a temp checkout is not a project
the user worked in), so the fixture switches that one guard off for the tests
that need a repo, and one test keeps it on to prove it exists.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core import repo_context
from tokenjam.core.distill import INVOKE_CWD_DIRNAME
from tokenjam.core.models import SessionContext
from tokenjam.core.repo_context import (
    EMPTY_CONTEXT,
    developer_id_for,
    ensure_install_id,
    github_login,
    normalise_remote_url,
    repo_name_from_url,
    resolve_repo_context,
    session_context_for_cwd,
    session_context_from_attrs,
)
from tokenjam.otel.semconv import ResourceAttributes, TjAttributes

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not available"
)

EMAIL = "Dev.Person@Example.com"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A real repo with one commit, an `origin` remote and a local author.

    HOME is redirected so the developer's global git config never leaks in
    (Critical Rule 47: `expanduser` reads the env var, not `Path.home`).
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(repo_context, "_is_temp_cwd", lambda _cwd: False)
    repo_context.clear_caches()
    path = tmp_path / "proj"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", EMAIL)
    _git(path, "config", "user.name", "Dev Person")
    _git(path, "remote", "add", "origin", "git@github.com:Acme/widgets.git")
    (path / "README.md").write_text("hi\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "init")
    yield path
    repo_context.clear_caches()


# --- Pure helpers ------------------------------------------------------------

def test_developer_id_is_the_lowercased_email_hash_prefix():
    import hashlib
    expected = hashlib.sha256(EMAIL.lower().encode()).hexdigest()[:16]
    assert developer_id_for(EMAIL) == expected
    assert developer_id_for("  " + EMAIL.upper() + " ") == expected
    assert developer_id_for(None) is None
    assert developer_id_for("   ") is None


@pytest.mark.parametrize("raw,expected", [
    ("git@github.com:Acme/widgets.git", "https://github.com/Acme/widgets"),
    ("https://github.com/Acme/widgets.git", "https://github.com/Acme/widgets"),
    ("https://user:t0ken@github.com/Acme/widgets.git", "https://github.com/Acme/widgets"),
    ("ssh://git@github.com:22/Acme/widgets/", "https://github.com/Acme/widgets"),
    ("git://GitHub.com/Acme/widgets.git", "https://github.com/Acme/widgets"),
    ("https://gitlab.example.com/group/sub/repo.git", "https://gitlab.example.com/group/sub/repo"),
    ("/local/path/repo.git", None),
    ("", None),
    (None, None),
])
def test_remote_url_normalisation(raw, expected):
    assert normalise_remote_url(raw) == expected


def test_normalised_urls_never_carry_credentials():
    assert "t0ken" not in (normalise_remote_url("https://u:t0ken@host.com/a/b.git") or "")


def test_repo_name_is_the_last_two_segments():
    assert repo_name_from_url("https://github.com/Acme/widgets") == "Acme/widgets"
    assert repo_name_from_url("https://gitlab.example.com/group/sub/repo") == "sub/repo"
    assert repo_name_from_url("https://github.com/only") is None
    assert repo_name_from_url(None) is None


# --- The resolver --------------------------------------------------------------

def test_resolves_root_remote_branch_head_and_author(repo):
    ctx = resolve_repo_context(str(repo))
    assert ctx.repo_root == str(repo.resolve())
    assert ctx.remote_url == "https://github.com/Acme/widgets"
    assert ctx.repo_name == "Acme/widgets"
    assert ctx.branch == "main"
    assert ctx.head_sha == _git(repo, "rev-parse", "HEAD")
    assert ctx.user_email == EMAIL
    assert ctx.developer_id == developer_id_for(EMAIL)


def test_resolves_from_a_subdirectory_of_the_repo(repo):
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    ctx = resolve_repo_context(str(sub))
    assert ctx.repo_root == str(repo.resolve())
    assert ctx.repo_name == "Acme/widgets"


def test_a_repo_without_a_remote_still_yields_root_and_branch(repo):
    _git(repo, "remote", "remove", "origin")
    repo_context.clear_caches()
    ctx = resolve_repo_context(str(repo))
    assert ctx.repo_root == str(repo.resolve())
    assert ctx.remote_url is None
    assert ctx.repo_name is None
    assert ctx.branch == "main"


def test_detached_head_has_no_branch(repo):
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", sha)
    repo_context.clear_caches()
    ctx = resolve_repo_context(str(repo))
    assert ctx.branch is None
    assert ctx.head_sha == sha


def test_a_non_repo_directory_yields_no_repo_but_may_know_the_author(repo, tmp_path):
    plain = tmp_path / "notarepo"
    plain.mkdir()
    ctx = resolve_repo_context(str(plain))
    assert not ctx.is_repo
    assert ctx.repo_root is None and ctx.remote_url is None and ctx.branch is None


def test_a_missing_directory_is_absence(repo):
    assert resolve_repo_context(str(repo / "gone")) == EMPTY_CONTEXT
    assert resolve_repo_context(None) == EMPTY_CONTEXT
    assert resolve_repo_context("") == EMPTY_CONTEXT


def test_tokenjams_own_invoke_cwd_is_skipped(repo):
    invoke = repo / INVOKE_CWD_DIRNAME
    invoke.mkdir()
    assert resolve_repo_context(str(invoke)) == EMPTY_CONTEXT


def test_a_temp_directory_is_skipped(tmp_path, monkeypatch):
    """The real guard, with the fixture's override absent: `tmp_path` lives
    under the platform temp dir, so a repo built there resolves to nothing."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo_context.clear_caches()
    path = tmp_path / "tmpproj"
    path.mkdir()
    _git(path, "init", "-q")
    assert repo_context._is_temp_cwd(str(path))
    assert resolve_repo_context(str(path)) == EMPTY_CONTEXT


def test_git_failures_never_raise(repo, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("simulated git explosion")
    monkeypatch.setattr(repo_context, "_read_head", boom)
    repo_context.clear_caches()
    assert resolve_repo_context(str(repo)) == EMPTY_CONTEXT


def test_every_shell_out_is_read_only_bounded_and_unchecked(repo, monkeypatch):
    calls: list[tuple] = []
    real_run = subprocess.run

    def spy(args, **kwargs):
        calls.append((args, kwargs))
        return real_run(args, **kwargs)
    monkeypatch.setattr(repo_context.subprocess, "run", spy)
    repo_context.clear_caches()
    resolve_repo_context(str(repo))
    assert calls, "expected git to be invoked"
    for args, kwargs in calls:
        assert kwargs["timeout"] == repo_context.GIT_TIMEOUT_S == 2.0
        assert kwargs["check"] is False
        verb = args[1]
        assert verb in {"rev-parse", "remote", "config"}, args
        if verb == "remote":
            assert args[2] == "get-url"
        if verb == "config":
            assert "--global" in args or args[-1] == "user.email"


def test_repo_identity_is_cached_per_root_and_head(repo, monkeypatch):
    resolve_repo_context(str(repo))
    calls: list[list[str]] = []
    monkeypatch.setattr(repo_context, "_git", lambda args, cwd: calls.append(args))
    # Second resolution inside the TTL: no shell-out at all.
    ctx = resolve_repo_context(str(repo))
    assert ctx.repo_name == "Acme/widgets"
    assert calls == []


def test_missing_git_binary_is_absence(repo, monkeypatch):
    monkeypatch.setattr(repo_context.shutil, "which", lambda _name: None)
    repo_context.clear_caches()
    assert resolve_repo_context(str(repo)) == EMPTY_CONTEXT


# --- Session-shaped views --------------------------------------------------------

def test_session_context_for_cwd_takes_branches_from_the_caller_and_no_sha(repo):
    ctx = session_context_for_cwd(str(repo), branch_start="feat/x", branch_end="feat/y")
    assert ctx == SessionContext(
        repo_remote="https://github.com/Acme/widgets",
        repo_root=str(repo.resolve()),
        branch_start="feat/x",
        branch_end="feat/y",
        head_sha_start=None,   # git today cannot name the session's start commit
        head_sha_end=None,
        developer_id=developer_id_for(EMAIL),
        user_email=EMAIL,
    )


def test_session_context_from_attrs_reads_every_section_3_key():
    attrs = {
        ResourceAttributes.USER_EMAIL: EMAIL,
        ResourceAttributes.VCS_REPOSITORY_URL_FULL: "git@github.com:Acme/widgets.git",
        ResourceAttributes.VCS_REF_HEAD_NAME: "main",
        ResourceAttributes.VCS_REF_HEAD_REVISION: "abc123",
        TjAttributes.REPO_ROOT: "/work/widgets",
        TjAttributes.SESSION_BRANCH_END: "feat/z",
        TjAttributes.SESSION_HEAD_END: "def456",
    }
    ctx = session_context_from_attrs(attrs)
    assert ctx == SessionContext(
        repo_remote="https://github.com/Acme/widgets",
        repo_root="/work/widgets",
        branch_start="main",
        branch_end="feat/z",
        head_sha_start="abc123",
        head_sha_end="def456",
        developer_id=developer_id_for(EMAIL),   # derived when not sent
        user_email=EMAIL,
    )


def test_session_context_from_attrs_prefers_a_sent_developer_id():
    ctx = session_context_from_attrs({
        ResourceAttributes.USER_EMAIL: EMAIL,
        TjAttributes.DEVELOPER_ID: "sentbyproducer00",
    })
    assert ctx is not None and ctx.developer_id == "sentbyproducer00"


def test_session_context_from_attrs_is_none_when_nothing_is_stamped():
    assert session_context_from_attrs({}) is None
    assert session_context_from_attrs(None) is None
    assert session_context_from_attrs({"gen_ai.request.model": "x", "session.id": "s"}) is None


# --- Identity files under ~/.tj ---------------------------------------------------

def test_install_id_is_generated_once_under_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    first = ensure_install_id()
    assert first and len(first) == 36
    assert (home / ".tj" / "install_id").read_text().strip() == first
    assert ensure_install_id() == first


def test_github_login_uses_gh_only_when_present_and_caches_for_a_day(tmp_path, monkeypatch):
    identity = tmp_path / "identity.json"
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    runs: list[list[str]] = []

    monkeypatch.setattr(repo_context.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None)
    monkeypatch.setattr(repo_context, "_run", lambda args, cwd: runs.append(args) or "octocat")
    assert github_login(identity_path=identity, now=now) == "octocat"
    assert runs == [["/usr/bin/gh", "api", "user", "-q", ".login"]]
    cached = json.loads(identity.read_text())
    assert cached["github_login"] == "octocat"

    # Inside the TTL: served from the file, no shell-out.
    assert github_login(identity_path=identity, now=now + timedelta(hours=23)) == "octocat"
    assert len(runs) == 1
    # Past it: re-asked.
    assert github_login(identity_path=identity, now=now + timedelta(hours=25)) == "octocat"
    assert len(runs) == 2


def test_github_login_is_absent_without_gh_and_the_absence_is_cached(tmp_path, monkeypatch):
    identity = tmp_path / "identity.json"
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(repo_context.shutil, "which", lambda _name: None)
    assert github_login(identity_path=identity, now=now) is None
    assert json.loads(identity.read_text())["github_login"] is None


def test_github_login_is_absent_when_gh_is_not_authenticated(tmp_path, monkeypatch):
    identity = tmp_path / "identity.json"
    monkeypatch.setattr(repo_context.shutil, "which", lambda _name: "/usr/bin/gh")
    monkeypatch.setattr(repo_context, "_run", lambda args, cwd: None)  # non-zero exit
    assert github_login(identity_path=identity) is None
