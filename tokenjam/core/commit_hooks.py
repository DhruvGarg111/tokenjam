"""
The commit trailer hook, the git-notes hook and the active-session record
(shipped-value ledger W3; contracts §3, §5).

Why: a commit the developer makes by hand mid-session, from a plain shell
rather than the agent's Bash tool, leaves no tool span for W2's matcher to
key on, so it joins the session at best as `inferred` and usually not at
all. The `prepare-commit-msg` hook appends `TokenJam-Session: <id>` to the
message, which `core/shipped.py` already resolves at `deterministic`
confidence (`trailer_session`). The optional `post-commit` hook writes the
session's measured cost to `refs/notes/tokenjam` (contracts §5 write side).

Three pieces, all here so the CLI (`tj init --hooks / --notes`, `tj doctor`,
`tj uninstall`) and the statusline cannot disagree about the file formats:

* **The active-session record** (`~/.tj/active_sessions.json`). Claude Code
  hands `tj statusline` `{session_id, cwd}` on every render; the statusline
  writes `{repo_root: {session_id, cwd, updated_at, updated_epoch}}` here,
  atomically, pruned of entries older than `ACTIVE_SESSION_TTL_S`. The hook
  reads it with `grep`/`sed`, never Python, so the layout is a CONTRACT: one
  entry per line, the key JSON-encoded without ASCII escaping, the fields in
  that order. Keyed by the git worktree root rather than the raw cwd because
  git runs hooks from the root, while a session may sit in a subdirectory.
  `CLAUDE_CODE_SESSION_ID`, which Claude Code exports to its own Bash
  children, takes precedence in the hook; the file is the plain-shell path.

* **Managed hook blocks.** A hook file is a script the user may own, so the
  block is delimited by a stable sentinel pair (Critical Rule 21 in
  `tokenjam/CLAUDE.md`): install strips every existing block and writes one
  fresh block after the shebang, remove strips it and deletes the file only
  when nothing else is left. A hook whose shebang is not a POSIX shell is
  never edited (a Python hook cannot host a sh block); the caller reports
  that and moves on.

* **The note payload** (`build_commit_note`), the exact §5 object.

Git shell-outs follow W1's discipline: read-only, `timeout=2`, `check=False`,
absent on failure.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenjam.core.agent_kind import classify_agent_kind
from tokenjam.core.repo_context import GIT_TIMEOUT_S, tj_home
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger(__name__)

#: An active-session entry older than this is pruned on the next write.
ACTIVE_SESSION_TTL_S = 6 * 3600
#: The hook trusts an entry only this fresh (mirrors the sh constant).
ACTIVE_SESSION_FRESH_S = 30 * 60

#: The notes ref this build writes (contracts §5) and the trailer token.
NOTES_REF = "tokenjam"
TRAILER_TOKEN = "TokenJam-Session"

#: Sentinels, one pair per hook. Stable strings: a rename is a migration.
HOOK_BLOCK_START = {
    "prepare-commit-msg": "# >>> tokenjam commit trailer (managed) >>>",
    "post-commit": "# >>> tokenjam commit note (managed) >>>",
}
HOOK_BLOCK_END = {
    "prepare-commit-msg": "# <<< tokenjam commit trailer <<<",
    "post-commit": "# <<< tokenjam commit note <<<",
}
HOOK_NAMES: tuple[str, ...] = ("prepare-commit-msg", "post-commit")

_SHEBANG_SHELLS = ("sh", "bash", "dash", "zsh", "ksh", "ash")
_DEFAULT_SHEBANG = "#!/bin/sh"


# --- Active-session record ------------------------------------------------------

def active_sessions_path() -> Path:
    return tj_home() / "active_sessions.json"


def _worktree_root(cwd: str) -> str:
    """The nearest ancestor of `cwd` holding `.git` (a directory, or the file
    a linked worktree carries), else `cwd` itself. Pure filesystem: this runs
    on every statusline render and must not shell out."""
    p = Path(cwd)
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            # realpath, as `git rev-parse --show-toplevel` reports it: the
            # hook's key must match this one byte for byte.
            return os.path.realpath(candidate)
    return os.path.realpath(p)


def _render_active_sessions(entries: dict[str, dict[str, Any]]) -> str:
    """One entry per line, key first, fields in a fixed order: the sh hook
    greps the key and seds the fields, so this layout is the contract."""
    lines = ["{"]
    items = sorted(entries.items())
    for i, (root, e) in enumerate(items):
        row = (
            f"  {json.dumps(root, ensure_ascii=False)}: "
            f"{{\"session_id\": {json.dumps(str(e['session_id']))}, "
            f"\"cwd\": {json.dumps(str(e.get('cwd') or root), ensure_ascii=False)}, "
            f"\"updated_at\": {json.dumps(str(e['updated_at']))}, "
            f"\"updated_epoch\": {int(e['updated_epoch'])}}}"
        )
        lines.append(row + ("," if i < len(items) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def read_active_sessions(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The record as written, `{}` when missing or unreadable. Never raises."""
    target = path or active_sessions_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and v.get("session_id")}


def record_active_session(
    cwd: str | None, session_id: str | None, *, path: Path | None = None,
    now: datetime | None = None,
) -> bool:
    """Write `cwd`'s session into the record; True when written.

    Atomic (temp file + `os.replace`), pruned of entries past the TTL, and
    silent on every failure: this runs from the statusline, which must never
    surface an error.
    """
    if not cwd or not session_id:
        return False
    sid = str(session_id).strip()
    if not sid or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
        return False
    target = path or active_sessions_path()
    at = now or utcnow()
    epoch = int(at.timestamp())
    try:
        entries = read_active_sessions(target)
        entries = {
            k: v for k, v in entries.items()
            if isinstance(v.get("updated_epoch"), int)
            and epoch - int(v["updated_epoch"]) <= ACTIVE_SESSION_TTL_S
        }
        entries[_worktree_root(cwd)] = {
            "session_id": sid, "cwd": cwd,
            "updated_at": at.isoformat(), "updated_epoch": epoch,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".active_sessions.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(_render_active_sessions(entries))
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return True
    except (OSError, ValueError, TypeError):
        logger.debug("could not record active session at %s", target, exc_info=True)
        return False


def active_session_for(
    cwd: str, *, path: Path | None = None, now: datetime | None = None,
    max_age_s: int = ACTIVE_SESSION_FRESH_S,
) -> dict[str, Any] | None:
    """The fresh entry for `cwd`'s worktree, resolved the way the hook does,
    or None. `tj doctor` uses it to say whether a hand commit right now
    would join a session."""
    entries = read_active_sessions(path)
    entry = entries.get(_worktree_root(cwd))
    if not entry or not isinstance(entry.get("updated_epoch"), int):
        return None
    epoch = int((now or utcnow()).timestamp())
    if epoch - int(entry["updated_epoch"]) > max_age_s:
        return None
    return entry


# --- Where the blocks went (so `tj uninstall` can find them) -------------------

def hooks_registry_path() -> Path:
    return tj_home() / "commit_hooks.json"


def registered_hooks_dirs(path: Path | None = None) -> list[str]:
    target = path or hooks_registry_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    dirs = data.get("hooks_dirs") if isinstance(data, dict) else None
    return [d for d in dirs if isinstance(d, str)] if isinstance(dirs, list) else []


def register_hooks_dir(hooks_dir: Path, *, path: Path | None = None) -> None:
    """Remember a hooks directory tj wrote a block into. Hooks are per repo
    and `tj uninstall` runs from anywhere, so without this it could only
    clean the repo it happened to be run from. Best effort, never raises."""
    target = path or hooks_registry_path()
    dirs = registered_hooks_dirs(target)
    entry = str(hooks_dir)
    if entry in dirs:
        return
    dirs.append(entry)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({"hooks_dirs": dirs}, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        logger.debug("could not record hooks dir at %s", target, exc_info=True)


# --- Git helpers ----------------------------------------------------------------

def _git(args: list[str], cwd: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(
            [git, *args], cwd=cwd, capture_output=True, text=True,
            timeout=GIT_TIMEOUT_S, check=False, errors="replace",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return proc.stdout if proc.returncode == 0 else None


@dataclass(frozen=True)
class HooksLocation:
    """Where this repo's hooks live and whether tj may write there."""
    repo_root: str
    hooks_dir: Path
    #: `core.hooksPath` as configured, None when git's default applies.
    hooks_path_setting: str | None
    #: A `core.hooksPath` inside the worktree is a tracked directory (husky
    #: and friends); tj never edits tracked files, so this is a refusal.
    tracked: bool


def resolve_hooks_location(cwd: str) -> HooksLocation | None:
    """Resolve the hooks directory git will consult for `cwd`, or None when
    `cwd` is not inside a git worktree."""
    root = (_git(["rev-parse", "--show-toplevel"], cwd) or "").strip()
    if not root:
        return None
    hooks_out = (_git(["rev-parse", "--git-path", "hooks"], cwd) or "").strip()
    if not hooks_out:
        return None
    hooks_dir = Path(cwd, hooks_out).resolve()
    setting = (_git(["config", "--get", "core.hooksPath"], cwd) or "").strip() or None
    tracked = False
    if setting is not None:
        root_r = Path(root).resolve()
        git_dir = Path(cwd, (_git(["rev-parse", "--git-dir"], cwd) or ".git").strip()).resolve()
        inside_worktree = hooks_dir == root_r or root_r in hooks_dir.parents
        inside_git_dir = hooks_dir == git_dir or git_dir in hooks_dir.parents
        tracked = inside_worktree and not inside_git_dir
    return HooksLocation(repo_root=root, hooks_dir=hooks_dir,
                         hooks_path_setting=setting, tracked=tracked)


# --- Managed blocks -------------------------------------------------------------

def _templates_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "hooks"


def hook_block_body(name: str) -> str:
    """The packaged block body for `name` (no sentinels, no shebang)."""
    if name not in HOOK_NAMES:
        raise ValueError(f"unknown hook {name!r}")
    return (_templates_dir() / name).read_text(encoding="utf-8").rstrip("\n") + "\n"


def render_hook_block(name: str) -> str:
    return f"{HOOK_BLOCK_START[name]}\n{hook_block_body(name)}{HOOK_BLOCK_END[name]}\n"


def _block_pattern(name: str) -> re.Pattern[str]:
    # Both line endings optional (`(?:\n|$)`): a block at EOF with no trailing
    # newline is still a block, the trap Critical Rule 21 records.
    return re.compile(
        rf"{re.escape(HOOK_BLOCK_START[name])}\n.*?{re.escape(HOOK_BLOCK_END[name])}(?:\n|$)",
        re.DOTALL,
    )


def _damaged_pattern(name: str) -> re.Pattern[str]:
    # A start sentinel whose end sentinel was lost: the body still ends with
    # its `unset -f <fn>` line (both templates do), so cut through that and
    # leave whatever the user has after it alone.
    return re.compile(
        rf"{re.escape(HOOK_BLOCK_START[name])}\n.*?unset -f \S+(?:\n|$)", re.DOTALL,
    )


def strip_hook_block(name: str, text: str) -> str:
    """`text` with every tj-managed block for `name` removed, a damaged one
    (end sentinel gone) included when its body is still recognisable.
    Idempotent."""
    out = _block_pattern(name).sub("", text)
    if HOOK_BLOCK_START[name] in out:
        out = _damaged_pattern(name).sub("", out)
        out = re.sub(rf"^{re.escape(HOOK_BLOCK_END[name])}\n?", "", out, flags=re.MULTILINE)
    return out


def installed_block_body(name: str, text: str) -> str | None:
    """The body of the first managed block in `text`, None when absent or
    unterminated (a start sentinel with no end is damage, not a block)."""
    m = _block_pattern(name).search(text)
    if m is None:
        return None
    inner = m.group(0)
    inner = inner[len(HOOK_BLOCK_START[name]) + 1:]
    inner = inner[: inner.rfind(HOOK_BLOCK_END[name])]
    return inner


def _shebang_is_shell(first_line: str) -> bool:
    if not first_line.startswith("#!"):
        return True  # git runs a shebang-less hook through sh
    interp = first_line[2:].strip().split()
    if not interp:
        return False
    prog = Path(interp[0]).name
    if prog == "env" and len(interp) > 1:
        prog = Path(interp[1]).name
    return prog in _SHEBANG_SHELLS


def install_hook_block(name: str, hook_path: Path) -> str:
    """Write exactly one fresh managed block into `hook_path`.

    Returns `written` (new file), `updated` (block replaced or added to an
    existing hook), `kept` (already current), `skipped` (the existing hook
    is not a shell script, or is a symlink; left untouched) or `damaged` (a
    start sentinel with no recognisable body after it; left for the user).
    """
    block = render_hook_block(name)
    if hook_path.is_symlink():
        return "skipped"
    if not hook_path.exists():
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        _write_executable(hook_path, f"{_DEFAULT_SHEBANG}\n{block}")
        return "written"
    text = hook_path.read_text(encoding="utf-8", errors="replace")
    first = text.split("\n", 1)[0]
    if not _shebang_is_shell(first):
        return "skipped"
    if installed_block_body(name, text) == hook_block_body(name):
        return "kept"
    rest = strip_hook_block(name, text)
    if HOOK_BLOCK_START[name] in rest:
        return "damaged"  # unrecognisable residue; not ours to guess at
    if rest.startswith("#!"):
        shebang, _, tail = rest.partition("\n")
        new_text = f"{shebang}\n{block}{tail}"
    else:
        new_text = f"{_DEFAULT_SHEBANG}\n{block}{rest}"
    _write_executable(hook_path, new_text)
    return "updated"


def remove_hook_block(name: str, hook_path: Path) -> str:
    """Strip the managed block; delete the file when only our shebang is
    left. Returns `removed`, `deleted` or `absent`."""
    if hook_path.is_symlink() or not hook_path.exists():
        return "absent"
    text = hook_path.read_text(encoding="utf-8", errors="replace")
    if installed_block_body(name, text) is None and HOOK_BLOCK_START[name] not in text:
        return "absent"
    rest = strip_hook_block(name, text)
    if rest == text:
        return "absent"
    if rest.strip() in ("", _DEFAULT_SHEBANG):
        hook_path.unlink()
        return "deleted"
    _write_executable(hook_path, rest)
    return "removed"


def _write_executable(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o755
    mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tj-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


#: `hook_state` values. `current` is the packaged block; `stale` a block from
#: another build (re-run `tj init --hooks`); `damaged` a start sentinel with
#: no end; `absent` no block; `foreign` a non-shell hook we would not edit.
STATE_CURRENT = "current"
STATE_STALE = "stale"
STATE_DAMAGED = "damaged"
STATE_ABSENT = "absent"
STATE_FOREIGN = "foreign"


def hook_state(name: str, hook_path: Path) -> str:
    if not hook_path.exists():
        return STATE_ABSENT
    text = hook_path.read_text(encoding="utf-8", errors="replace")
    body = installed_block_body(name, text)
    if body is not None:
        return STATE_CURRENT if body == hook_block_body(name) else STATE_STALE
    if HOOK_BLOCK_START[name] in text:
        return STATE_DAMAGED
    if not _shebang_is_shell(text.split("\n", 1)[0]):
        return STATE_FOREIGN
    return STATE_ABSENT


def hook_states(hooks_dir: Path) -> dict[str, str]:
    return {name: hook_state(name, hooks_dir / name) for name in HOOK_NAMES}


# --- The §5 note ----------------------------------------------------------------

def build_commit_note(session: Any) -> dict[str, Any]:
    """The `refs/notes/tokenjam` object for a commit carrying a
    `TokenJam-Session:` trailer that resolved to `session` (contracts §5).

    `confidence` / `source` are what the trailer earns in `core/shipped.py`,
    not a claim of their own. `cost_usd` is the session's MEASURED cost and
    `pricing_mode` rides with it so a reader never renders a subscription
    session's figure as spend (contracts §1 copy rules).
    """
    tool = getattr(session, "source", None) or classify_agent_kind(
        getattr(session, "agent_id", None)).group or "sdk"
    cost = getattr(session, "total_cost_usd", None)
    return {
        "v": 1,
        "session_id": session.session_id,
        "tool": tool,
        "model": getattr(session, "dominant_model", None),
        "cost_usd": float(cost) if cost is not None else None,
        "pricing_mode": getattr(session, "pricing_mode", None) or "unknown",
        "confidence": "deterministic",
        "source": "trailer_session",
    }


def commit_trailer_session(repo_root: str, sha: str = "HEAD") -> str | None:
    """The `TokenJam-Session:` value on `sha`, or None."""
    body = _git(["log", "-1", "--format=%B", sha], repo_root)
    if not body:
        return None
    m = re.search(rf"^{TRAILER_TOKEN}:\s*(\S+)\s*$", body, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def write_commit_note(repo_root: str, sha: str, note: dict[str, Any]) -> bool:
    """`git notes --ref=tokenjam add -f` the note. The one git WRITE in the
    ledger, and only ever to our own ref."""
    git = shutil.which("git")
    if git is None:
        return False
    try:
        proc = subprocess.run(
            [git, "notes", f"--ref={NOTES_REF}", "add", "-f", "-m",
             json.dumps(note, separators=(",", ":")), sha],
            cwd=repo_root, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return proc.returncode == 0
