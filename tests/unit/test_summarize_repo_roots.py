"""Deriving scannable project roots from the window's recorded working dirs.

The summarize analyzer prices a telemetry window that spans every repo the user
touched. These tests pin the two honesty rules that govern the derivation: a
recorded directory that is gone contributes nothing (and is counted, so the
basis can say so), and no derived root may escape the boundary gate the cwd
scan already passes.
"""
from __future__ import annotations

from pathlib import Path

from tokenjam.core.summarize.repo_roots import resolve_roots


def _repo(root: Path) -> Path:
    (root / ".git").mkdir(parents=True)
    return root


def test_existing_directories_become_roots(tmp_path):
    a, b = _repo(tmp_path / "code" / "a"), _repo(tmp_path / "code" / "b")
    resolved = resolve_roots([str(a), str(b)])

    assert set(resolved.roots) == {a.resolve(), b.resolve()}
    assert resolved.recorded == 2
    assert resolved.vanished == 0


def test_vanished_directory_is_counted_not_reconstructed(tmp_path):
    """A deleted repo has no file to read, so no figure may be quoted for it —
    but the size of that blind spot must still be reportable."""
    live = _repo(tmp_path / "code" / "live")
    resolved = resolve_roots([str(live), str(tmp_path / "code" / "deleted")])

    assert resolved.roots == (live.resolve(),)
    assert resolved.recorded == 2
    assert resolved.vanished == 1


def test_subdirectory_contributes_its_repo_root_too(tmp_path):
    """Claude Code loads `CLAUDE.md` from the working directory AND its
    ancestors, so a session launched inside a sub-package really does carry
    both files."""
    repo = _repo(tmp_path / "code" / "repo")
    inner = repo / "packages" / "api"
    inner.mkdir(parents=True)

    resolved = resolve_roots([str(inner)])

    assert set(resolved.roots) == {inner.resolve(), repo.resolve()}


def test_home_is_never_a_scan_root(tmp_path, monkeypatch):
    """A stray recorded cwd must not point the scan at the user's whole home."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = resolve_roots([str(tmp_path)])

    assert resolved.roots == ()
    assert resolved.refused == 1
    assert resolved.vanished == 0      # it exists; it is refused, not missing


def test_duplicate_and_empty_recordings_collapse(tmp_path):
    repo = _repo(tmp_path / "code" / "repo")

    resolved = resolve_roots([str(repo), str(repo), "", str(repo)])

    assert resolved.roots == (repo.resolve(),)
    assert resolved.recorded == 1
