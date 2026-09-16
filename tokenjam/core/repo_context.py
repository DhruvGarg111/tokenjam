"""
Repo context + developer identity for a session (shipped-value ledger, contracts §3).

Given the directory a session ran in, answer: which repo (normalised remote +
root), which branch and HEAD, and who (git author email, hashed into a
developer id). The next wave joins sessions to commits on exactly these
columns, so this module is the one place they are derived.

Discipline, all of it load-bearing:

* **Read-only git shell-outs, `timeout=2`, `check=False`.** Nothing here
  mutates a repo, and nothing here may hang an ingest.
* **A failure yields absence, never a placeholder.** No git on PATH, no repo,
  an unborn HEAD, a cwd that no longer exists (backfilled transcripts routinely
  point at deleted worktrees): every one of these is `None` for the fields it
  affects, so a downstream join can never match on a made-up value.
* **Never on the request path.** Callers are the backfill, the daemon's
  transcript catch-up and `tj onboard`; a REST route must not call this.
* **Skipped for tokenjam's own invoke cwd and for anything under a temp dir**
  (`is_tokenjam_invoke_cwd`, `_is_temp_cwd`): those are not projects a user
  worked in, and a temp checkout's remote would attribute work to the wrong
  place.
* **Never raises.** Every public function is wrapped so a bug here degrades to
  "no context", not a failed backfill.

Caching: the cheap `cwd -> (repo_root, branch, head_sha)` read is memoised per
cwd for `_HEAD_TTL_S` so a backfill over thousands of sessions in a handful of
directories does not shell out per session, while a long-lived daemon still
notices a branch switch. The remote and author email are cached per
`(repo_root, head_sha)` for the life of the process, as the brief specifies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePath
from typing import Any, Mapping

from tokenjam.core.distill import is_tokenjam_invoke_cwd
from tokenjam.core.models import SessionContext
from tokenjam.otel.semconv import ResourceAttributes, TjAttributes
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger(__name__)

#: Hard ceiling on every git / gh shell-out. A hung subprocess must never stall
#: an ingest; 2s is generous for `rev-parse` and `config` on a local checkout.
GIT_TIMEOUT_S = 2.0

#: How long the per-cwd HEAD read is trusted before it is re-taken.
_HEAD_TTL_S = 30.0

#: How long a resolved (or absent) GitHub login is trusted.
GITHUB_LOGIN_TTL = timedelta(hours=24)

# Process-wide caches. Keyed as documented in the module docstring; cleared by
# `clear_caches()` (tests) and never persisted.
_head_cache: dict[str, tuple[float, tuple[str | None, str | None, str | None]]] = {}
_repo_cache: dict[tuple[str, str | None], tuple[str | None, str | None]] = {}
_global_email_cache: list[str | None] = []


@dataclass(frozen=True)
class RepoContext:
    """What `resolve_repo_context` learned about a cwd. Every field optional."""
    repo_root:    str | None = None
    remote_url:   str | None = None   # normalised: https://github.com/org/repo
    repo_name:    str | None = None   # org/repo
    branch:       str | None = None
    head_sha:     str | None = None
    user_email:   str | None = None
    developer_id: str | None = None

    @property
    def is_repo(self) -> bool:
        return self.repo_root is not None


EMPTY_CONTEXT = RepoContext()


# --- Pure helpers ------------------------------------------------------------

def developer_id_for(email: str | None) -> str | None:
    """`sha256(lower(email))[:16]`, or None for an empty email (contracts §3)."""
    if not email:
        return None
    cleaned = email.strip().lower()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


_SCP_LIKE = re.compile(r"^(?:[A-Za-z0-9._-]+@)?([A-Za-z0-9._-]+):(?!//)(.+)$")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")


def normalise_remote_url(url: str | None) -> str | None:
    """Canonical `https://host/org/repo` for any git remote spelling.

    Handles `git@github.com:org/repo.git`, `ssh://git@github.com/org/repo`,
    `https://user:token@github.com/org/repo.git`, `git://`, trailing slashes
    and `.git`. Credentials are always dropped: this value travels to a
    shared ledger. Anything unparseable (a local path, an empty string) is
    None, never echoed back.
    """
    if not url:
        return None
    raw = url.strip()
    if not raw:
        return None
    host: str | None = None
    path: str | None = None
    m = _SCHEME.match(raw)
    if m:
        rest = raw[m.end():]
        # Drop user[:password]@ before the host.
        if "@" in rest.split("/", 1)[0]:
            rest = rest.split("@", 1)[1]
        host, _, path = rest.partition("/")
        # ssh://host:port/path -> keep host only; a port is not identity.
        host = host.split(":", 1)[0]
    else:
        m = _SCP_LIKE.match(raw)
        if m:
            host, path = m.group(1), m.group(2)
    if not host or not path:
        return None
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/")
    if not path or "/" not in path:
        return None
    return f"https://{host.lower()}/{path}"


def repo_name_from_url(remote_url: str | None) -> str | None:
    """`org/repo` from a normalised remote URL (the last two path segments)."""
    if not remote_url:
        return None
    m = _SCHEME.match(remote_url)
    path = remote_url[m.end():] if m else remote_url
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:
        return None
    return "/".join(parts[-2:])


def _is_temp_cwd(cwd: str) -> bool:
    """True when `cwd` sits under the platform temp dir (or the usual macOS /
    Linux spellings of it). Compared on path parts, never by substring."""
    candidates = {tempfile.gettempdir(), "/tmp", "/private/tmp", "/var/folders",
                  "/private/var/folders"}
    try:
        candidates.add(os.path.realpath(tempfile.gettempdir()))
    except OSError:
        pass
    target = PurePath(cwd)
    for root in candidates:
        root_parts = PurePath(root).parts
        if root_parts and target.parts[:len(root_parts)] == root_parts:
            return True
    return False


def _skip_cwd(cwd: str | None) -> bool:
    if not cwd or not isinstance(cwd, str):
        return True
    if is_tokenjam_invoke_cwd(cwd) or _is_temp_cwd(cwd):
        return True
    try:
        return not os.path.isdir(cwd)
    except (OSError, ValueError):
        return True


# --- Shell-outs ----------------------------------------------------------------

def _run(args: list[str], cwd: str | None) -> str | None:
    """Run a read-only command; stdout stripped on exit 0, else None.

    Every failure mode (missing binary, missing cwd, non-zero exit, timeout,
    undecodable output) is absence. Never raises.
    """
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True,
            timeout=GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def _git(args: list[str], cwd: str | None) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    return _run([git, *args], cwd)


def _read_head(cwd: str) -> tuple[str | None, str | None, str | None]:
    """`(repo_root, branch, head_sha)` for cwd, memoised for `_HEAD_TTL_S`."""
    now = time.monotonic()
    hit = _head_cache.get(cwd)
    if hit is not None and now - hit[0] < _HEAD_TTL_S:
        return hit[1]
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    branch: str | None = None
    sha: str | None = None
    if root:
        # One call for both; an unborn HEAD fails the pair, which is correct:
        # a repo with no commits has no revision to name. `--abbrev-ref`
        # applies to every argument AFTER it, so the sha must come first.
        both = _git(["rev-parse", "HEAD", "--abbrev-ref", "HEAD"], cwd)
        if both:
            lines = both.splitlines()
            if len(lines) == 2:
                sha = lines[0].strip() or None
                branch = lines[1].strip() or None
                if branch == "HEAD":
                    # Detached: the abbrev is the literal word, not a branch.
                    branch = None
    result = (root, branch, sha)
    _head_cache[cwd] = (now, result)
    return result


def _read_repo_identity(root: str, sha: str | None, cwd: str) -> tuple[str | None, str | None]:
    """`(normalised remote, user_email)` cached per `(repo_root, head_sha)`."""
    key = (root, sha)
    hit = _repo_cache.get(key)
    if hit is not None:
        return hit
    remote = normalise_remote_url(_git(["remote", "get-url", "origin"], cwd))
    # `git config user.email` inside a repo already falls back to global.
    email = _git(["config", "user.email"], cwd)
    result = (remote, email)
    _repo_cache[key] = result
    return result


def _global_email() -> str | None:
    if not _global_email_cache:
        _global_email_cache.append(_git(["config", "--global", "user.email"], None))
    return _global_email_cache[0]


def clear_caches() -> None:
    """Forget every memoised read. For tests and for `tj onboard`'s re-runs."""
    _head_cache.clear()
    _repo_cache.clear()
    _global_email_cache.clear()


# --- Public resolvers ----------------------------------------------------------

def resolve_repo_context(cwd: str | None) -> RepoContext:
    """Everything §3 needs about `cwd`, or `EMPTY_CONTEXT` when it cannot be
    known. Never raises; see the module docstring for the rules."""
    try:
        return _resolve(cwd)
    except Exception:  # pragma: no cover - the whole point is to never raise
        logger.debug("repo context resolution failed for %r", cwd, exc_info=True)
        return EMPTY_CONTEXT


def _resolve(cwd: str | None) -> RepoContext:
    if _skip_cwd(cwd):
        return EMPTY_CONTEXT
    assert cwd is not None
    root, branch, sha = _read_head(cwd)
    if root is None:
        # Not a repo: only the author identity can still be known (global git
        # config), and only that is returned.
        email = _global_email()
        return RepoContext(user_email=email, developer_id=developer_id_for(email))
    remote, email = _read_repo_identity(root, sha, cwd)
    return RepoContext(
        repo_root=root,
        remote_url=remote,
        repo_name=repo_name_from_url(remote),
        branch=branch,
        head_sha=sha,
        user_email=email,
        developer_id=developer_id_for(email),
    )


def session_context_for_cwd(
    cwd: str | None,
    *,
    branch_start: str | None = None,
    branch_end: str | None = None,
) -> SessionContext:
    """The session columns a backfill can truthfully derive for a transcript.

    Branch values come from the transcript itself (the caller passes them);
    repo identity and author come from git in `cwd`. The HEAD sha is
    deliberately NOT taken from git here: `rev-parse HEAD` today names the
    current commit, not the one the session started on, and a wrong sha is
    worse than none for a join. `head_sha_*` stay None on this path.
    """
    ctx = resolve_repo_context(cwd)
    return SessionContext(
        repo_remote=ctx.remote_url,
        repo_root=ctx.repo_root,
        branch_start=branch_start or None,
        branch_end=branch_end or None,
        developer_id=ctx.developer_id,
        user_email=ctx.user_email,
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def session_context_from_attrs(attrs: Mapping[str, Any] | None) -> SessionContext | None:
    """Read the §3 attributes off a merged resource+span attribute dict.

    Returns None when none of them are present, so a span from a producer
    that never stamps them costs nothing downstream. `developer_id` is
    derived from `user.email` when the producer sent the email but not the
    hash, so the two can never disagree.
    """
    if not attrs:
        return None
    email = _str_or_none(attrs.get(ResourceAttributes.USER_EMAIL))
    dev_id = _str_or_none(attrs.get(TjAttributes.DEVELOPER_ID)) or developer_id_for(email)
    ctx = SessionContext(
        repo_remote=normalise_remote_url(
            _str_or_none(attrs.get(ResourceAttributes.VCS_REPOSITORY_URL_FULL))
        ),
        repo_root=_str_or_none(attrs.get(TjAttributes.REPO_ROOT)),
        branch_start=_str_or_none(attrs.get(ResourceAttributes.VCS_REF_HEAD_NAME)),
        branch_end=_str_or_none(attrs.get(TjAttributes.SESSION_BRANCH_END)),
        head_sha_start=_str_or_none(attrs.get(ResourceAttributes.VCS_REF_HEAD_REVISION)),
        head_sha_end=_str_or_none(attrs.get(TjAttributes.SESSION_HEAD_END)),
        developer_id=dev_id,
        user_email=email,
    )
    return None if ctx.is_empty() else ctx


# --- Identity files under ~/.tj ------------------------------------------------

def tj_home() -> Path:
    return Path.home() / ".tj"


def ensure_install_id(path: Path | None = None) -> str | None:
    """The once-generated uuid4 in `~/.tj/install_id` (contracts §3), creating
    it on first call. None (never a fresh id per call) when the file cannot be
    read or written: an install id that changes is worse than none."""
    target = path or (tj_home() / "install_id")
    try:
        if target.exists():
            existing = target.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        target.parent.mkdir(parents=True, exist_ok=True)
        new_id = str(uuid.uuid4())
        tmp = target.with_suffix(".tmp")
        tmp.write_text(new_id + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return new_id
    except OSError:
        logger.debug("could not persist install id at %s", target, exc_info=True)
        return None


def github_login(*, identity_path: Path | None = None, now: datetime | None = None) -> str | None:
    """`gh api user -q .login`, only when `gh` is on PATH and authenticated.

    The answer (including "none") is cached in `~/.tj/identity.json` for
    `GITHUB_LOGIN_TTL` so a backfill never pays a network round-trip per
    session, and a machine without `gh` never shells out again for a day.
    Never raises; absent on any failure.
    """
    try:
        return _github_login(identity_path or (tj_home() / "identity.json"), now or utcnow())
    except Exception:  # pragma: no cover
        logger.debug("github login lookup failed", exc_info=True)
        return None


def _github_login(identity_path: Path, now: datetime) -> str | None:
    cached = _read_identity(identity_path)
    checked = cached.get("github_login_checked_at")
    if isinstance(checked, str):
        try:
            checked_at = datetime.fromisoformat(checked)
        except ValueError:
            checked_at = None
        if checked_at is not None and checked_at.tzinfo is not None:
            if now - checked_at < GITHUB_LOGIN_TTL:
                stored = cached.get("github_login")
                return stored if isinstance(stored, str) and stored else None
    gh = shutil.which("gh")
    login: str | None = None
    if gh is not None:
        login = _run([gh, "api", "user", "-q", ".login"], None)
        if login is not None and ("\n" in login or " " in login):
            login = None
    cached["github_login"] = login
    cached["github_login_checked_at"] = now.isoformat()
    _write_identity(identity_path, cached)
    return login


def _read_identity(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_identity(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.debug("could not persist identity cache at %s", path, exc_info=True)


__all__ = [
    "GIT_TIMEOUT_S",
    "GITHUB_LOGIN_TTL",
    "RepoContext",
    "EMPTY_CONTEXT",
    "clear_caches",
    "developer_id_for",
    "ensure_install_id",
    "github_login",
    "normalise_remote_url",
    "repo_name_from_url",
    "resolve_repo_context",
    "session_context_for_cwd",
    "session_context_from_attrs",
]
