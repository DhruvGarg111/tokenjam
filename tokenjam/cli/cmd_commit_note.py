"""`tj commit-note`: write a commit's session cost to `refs/notes/tokenjam`
(shipped-value ledger W3; contracts §5 write side).

Called by the `post-commit` hook `tj init --notes` installs, for a commit
whose message carries `TokenJam-Session: <id>`. Resolves that session in
the local store (through the running daemon when it holds the DuckDB lock,
the standard root-callback fallback) and attaches the §5 object as a note.
The hook already skips when the daemon is down, so the direct-DB path here
is the by-hand one.

Never fails the commit it is called from: every outcome exits 0 and says
what it did or why it did nothing (`-v` for the reason). The note is the
one git write in the ledger, and only ever to our own ref; the commit
message is never touched here.
"""
from __future__ import annotations

import json as json_mod
import os

import click

from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.commit_hooks import (
    NOTES_REF,
    build_commit_note,
    commit_trailer_session,
    resolve_hooks_location,
    write_commit_note,
)
from tokenjam.utils.formatting import console


#: Structural opt-out: runs from a git hook on every commit and must stay
#: quiet there (its stdout is the hook's stdout).
@click.command("commit-note", cls=TjCommand, no_status=True,
               short_help="Write a trailered commit's session cost as a git note")
@click.argument("sha", default="HEAD")
@click.pass_context
def cmd_commit_note(ctx: click.Context, sha: str) -> None:
    """Attach the session's measured cost to a trailered commit as a git note."""
    output_json = ctx.obj.get("output_json", False)
    verbose = ctx.obj.get("verbose", False)

    def done(outcome: str, **extra: object) -> None:
        payload = {"outcome": outcome, "sha": sha, **extra}
        if output_json:
            click.echo(json_mod.dumps(payload))
        elif outcome == "written":
            console.print(f"✓ note written to refs/notes/{NOTES_REF} for {sha}")
        elif verbose:
            console.print(f"commit-note: {outcome} ({sha})")

    loc = resolve_hooks_location(os.getcwd())
    if loc is None:
        return done("not_a_repo")
    session_id = commit_trailer_session(loc.repo_root, sha)
    if session_id is None:
        return done("no_trailer")
    db = ctx.obj.get("db")
    session = db.get_session(session_id) if db is not None else None
    if session is None:
        return done("session_not_ingested", session_id=session_id)
    note = build_commit_note(session)
    if not write_commit_note(loc.repo_root, sha, note):
        return done("git_notes_failed", session_id=session_id)
    return done("written", session_id=session_id, note=note)
