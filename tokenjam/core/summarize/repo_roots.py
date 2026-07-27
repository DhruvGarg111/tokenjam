"""Which project roots the analysed window actually worked in.

`core/summarize/candidates` scans project scope from ONE root. For `tj summarize
list` that root is the cwd, which is right: the user is asking about the repo
they are standing in. For the optimize analyzer it is wrong — it prices a whole
telemetry window that spans every repo the user touched, so a single-cwd scan
leaves every other repo's always-resident `CLAUDE.md` contributing exactly $0.00
while the same file is demonstrably re-sent at the head of every one of that
repo's sessions.

The roots are OBSERVED, not guessed: Claude Code records the working directory
in each transcript, and `core/summarize/invocations` already reads every
transcript in the window (with the shared persistent parse cache), so the cwds
come back from a walk that was happening anyway.

Two honesty rules govern what survives that derivation:

* **A recorded directory that no longer exists on disk contributes nothing.**
  There is no file to read, so there is no reduction to offer and no figure to
  quote (Critical Rule 22 — never show a figure the user cannot act on). It is
  counted, so the basis string can say how much of the corpus was skipped, but
  it is never reconstructed.
* **A root must pass the same boundary gate the cwd scan passes**
  (:func:`candidates.is_safe_scan_root`): never ``/``, never ``$HOME``, never a
  bare top-level directory. A derivation path that can be pointed at ``$HOME``
  by a stray recorded cwd is a derivation path that will eventually scan the
  user's whole home directory.

Nothing here reads telemetry or prices anything; it turns recorded paths into
scannable roots and reports what it dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenjam.core.summarize.candidates import find_repo_root, is_safe_scan_root


@dataclass(frozen=True)
class ResolvedRoots:
    """Scannable project roots plus what the derivation had to drop."""

    #: Deduplicated, sorted, existing, boundary-safe roots to scan.
    roots: tuple[Path, ...] = ()
    #: Distinct working directories the window recorded, before filtering.
    recorded: int = 0
    #: Distinct recorded directories that no longer exist on disk. Scanned as
    #: nothing, never reconstructed — the blind spot is reported, not filled.
    vanished: int = 0
    #: Distinct recorded directories that exist but are refused as scan roots by
    #: the boundary gate (``$HOME``, ``/``, a bare top-level dir).
    refused: int = 0


def resolve_roots(cwds: Iterable[str]) -> ResolvedRoots:
    """Turn recorded working directories into scannable project roots.

    For each recorded directory that still exists, BOTH the directory itself and
    its enclosing git repo root (when they differ) become scan roots: Claude Code
    loads `CLAUDE.md` from the working directory and from its ancestors, so a
    session launched in a sub-package of a monorepo really does carry both files.
    Duplicate content across those roots is the caller's problem to collapse —
    the same file reached two ways resolves to one path here, but two byte-equal
    COPIES (a git worktree) are two real files and are returned as such.
    """
    roots: set[Path] = set()
    recorded: set[str] = set()
    vanished: set[str] = set()
    refused: set[str] = set()

    for raw in cwds:
        if not raw:
            continue
        recorded.add(raw)
        try:
            directory = Path(raw).expanduser()
            if not directory.is_dir():
                vanished.add(raw)
                continue
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            vanished.add(raw)
            continue
        found = False
        for candidate_root in (resolved, find_repo_root(resolved)):
            if candidate_root is None or not is_safe_scan_root(candidate_root):
                continue
            roots.add(candidate_root)
            found = True
        if not found:
            refused.add(raw)

    return ResolvedRoots(
        roots=tuple(sorted(roots)),
        recorded=len(recorded),
        vanished=len(vanished),
        refused=len(refused),
    )
