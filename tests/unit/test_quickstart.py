"""Unit tests for the zero-install / zero-config first-run (`tj quickstart`, #6).

The contract under test: a user with NO prior setup runs one command and sees
quota composition + a session timeline straight from on-disk Claude Code JSONL —
with no daemon, no onboarding, and crucially **no on-disk DB** (the command uses
a transient in-memory backend and must never call `open_db`).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from click.testing import CliRunner

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    compute_session_timeline,
    timeline_to_dict,
)

_NOW = datetime.now(timezone.utc)


def _date(month: int, day: int) -> str:
    """A `YYYY-MM-DD` fixture date anchored to test-execution time, not a
    hardcoded absolute literal.

    Every fixture below represents "recent" Claude Code history filtered
    through `--since 90d` (computed from real wall-clock `now()` inside the
    CLI). A fixed literal is a time bomb: once wall time passes
    `literal + 90d`, every assertion here starts failing with no code change
    involved -- the same class of bug fixed in
    `test_onboard_backfill_scope.py` and `test_transcript_sync.py`. The
    `(month, day)` pair only encodes the ORIGINAL relative spacing between
    fixture dates (e.g. `_date(6, 10)` is always ~20 days older than
    `_date(6, 30)`, `_date(6, 28)` ~2 days older); the actual calendar date
    floats with "now" so the gap to the `--since` cutoff never closes.
    """
    offset_days = (date(2026, 6, 30) - date(2026, month, day)).days
    return (_NOW - timedelta(days=offset_days)).strftime("%Y-%m-%d")


def _ts(month: int, day: int, time_of_day: str) -> str:
    """Full `...Z` timestamp for `_date(month, day)` at a given time-of-day
    (e.g. `"10:05:00.000"`)."""
    return f"{_date(month, day)}T{time_of_day}Z"


def _make_session_file(root: Path, session_id: str, cwd: str,
                       records: list[dict]) -> Path:
    project_dir = root / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant(uuid: str, session_id: str, cwd: str, ts: str, *,
               input_tokens: int = 500, output_tokens: int = 200,
               cache_read: int = 8000, cache_creation: int = 0) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
    }


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    # Two sessions across two projects, recent timestamps.
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", _ts(6, 20, "10:00:00.000")),
        _assistant("a2", "sess-a", "/Users/me/projA", _ts(6, 20, "10:05:00.000")),
    ])
    _make_session_file(root, "sess-b", "/Users/me/projB", [
        _assistant("b1", "sess-b", "/Users/me/projB", _ts(6, 21, "11:00:00.000"),
                   cache_read=50000),
    ])
    return root


# ── Session-timeline core (pure logic over an in-memory DB) ──────────────────

def test_timeline_summarizes_backfilled_sessions(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)

    assert timeline.has_data
    assert timeline.total_sessions == 2
    assert timeline.project_count == 2
    # Most-recent first.
    assert timeline.sessions[0].started_at >= timeline.sessions[-1].started_at
    # Project label is derived from the claude-code-<name> agent_id.
    projects = {s.project for s in timeline.sessions}
    assert "proja" in projects and "projb" in projects


def test_timeline_reread_share_reflects_cache_reads(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)
    for s in timeline.sessions:
        # Every fixture turn has cache reads, so re-read share is > 0.
        assert s.reread_share > 0
        assert s.total_tokens >= s.cache_tokens


def test_timeline_to_dict_is_json_serialisable(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    payload = timeline_to_dict(compute_session_timeline(db.conn))
    # Round-trips through json without error.
    round_tripped = json.loads(json.dumps(payload, default=str))
    assert round_tripped["total_sessions"] == 2
    assert len(round_tripped["sessions"]) == 2


def test_timeline_empty_db_has_no_data():
    db = InMemoryBackend()
    timeline = compute_session_timeline(db.conn)
    assert not timeline.has_data
    assert timeline.total_sessions == 0


# ── CLI: the zero-setup first run, with NO on-disk DB ────────────────────────

def _invoke_quickstart(args):
    """Run the zero-install report command directly.

    It has no public/typeable name on the `cli` group — `cli/main.py`'s
    no-subcommand branch invokes it via `ctx.invoke` only when the npm
    wrapper's `TJ_NPX_ZERO_INSTALL_REPORT` env var is set — so tests invoke
    the underlying `click.Command` object directly rather than through
    `cli`'s subcommand dispatch. The whole point of the command is that it
    never opens the on-disk DB or contacts the daemon — it manages its own
    transient in-memory backend.
    """
    from tokenjam.cli.cmd_quickstart import cmd_quickstart

    return CliRunner().invoke(cmd_quickstart, args)


def test_quickstart_renders_without_daemon_or_ondisk_db(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # Leads with the reads-your-local-logs framing. Compared against the
    # flattened output: Rich wraps the lead to console width, so the phrase
    # can straddle a line break.
    flat = _flat(result.output)
    assert "where your quota actually goes" in flat
    assert "~/.claude/projects" in flat
    # Both halves of the first-run value are present.
    assert "quota" in result.output.lower()
    # The per-session timeline table was replaced by the persona line + the
    # past-overspend callout; the window totals line survives it.
    assert "Session timeline" not in result.output
    assert "Totals:" in result.output
    # The opt-in "go deeper" pointer prints a CTA — see the two footer tests
    # below for the ephemeral (`npx tokenjam onboard`) vs installed (`tj
    # onboard`) forms (#507). Here we just assert an onboard CTA is present.
    assert "onboard" in result.output
    # The outro sells the local dashboard (#120) exactly once — the most
    # product-looking asset shouldn't be invisible at the conversion moment,
    # but a second, redundant mention right under the CTA was dropped (#436
    # review) to keep the outro tight and consistent with the npx-form CTA.
    assert result.output.count("dashboard") == 1
    assert "Lens" in result.output


# ── "Go deeper" footer CTA is context-aware (#507) ─────────────────────────
#
# Bare `npx tokenjam` / `uvx --from tokenjam tj` runs quickstart from a
# throwaway uvx/pipx-run cache → the CTA is the zero-install `npx tokenjam
# onboard`. But when quickstart runs from an already-installed `tj` binary the
# user obviously has it installed → the CTA drops the `npx tokenjam` prefix and
# points straight at `tj onboard`. `_go_deeper_command()` picks based on
# `cmd_onboard._is_ephemeral_runner()`.


def test_go_deeper_footer_ephemeral_runner_shows_npx_cta():
    import unittest.mock as mock

    from tokenjam.cli.cmd_quickstart import _go_deeper_command

    with mock.patch("tokenjam.cli.cmd_onboard._is_ephemeral_runner", return_value=True):
        assert _go_deeper_command() == "npx tokenjam onboard"


def test_go_deeper_footer_installed_binary_shows_tj_cta():
    import unittest.mock as mock

    from tokenjam.cli.cmd_quickstart import _go_deeper_command

    with mock.patch("tokenjam.cli.cmd_onboard._is_ephemeral_runner", return_value=False):
        assert _go_deeper_command() == "tj onboard"


# ── Quota-weighted headline + no named-session reclaim list (#119) ──────────
#
# The headline used to report a RAW token share as "quota" (mixing the two
# framings) and named individual ended sessions as "actionable" — but a user
# never returns to a session closed days ago, so a per-session retrospective
# callout is unactionable noise. The headline must now read as quota-weighted,
# and the default output must never name a past session.

def test_quickstart_headline_reads_as_quota_not_raw_tokens(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "of your quota went to" in result.output
    # The old raw-token wording is gone.
    assert "of your tokens went to" not in result.output


def _heavy_reread_fixture_root(tmp_path: Path) -> Path:
    """A session with one huge-cache-read turn — clears the compact-candidate
    thresholds (>= 200k re-read tokens, >= 80% re-read share) so the aggregate
    reclaim line renders."""
    root = tmp_path / "projects"
    _make_session_file(root, "sess-heavy", "/Users/me/projHeavy", [
        _assistant("h1", "sess-heavy", "/Users/me/projHeavy",
                   _ts(6, 20, "10:00:00.000"),
                   input_tokens=500, output_tokens=200, cache_read=300_000),
    ])
    return root


def test_quickstart_reclaim_section_is_aggregate_not_named_sessions(tmp_path):
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    # Rich wraps panel text to console width and interleaves the panel's own
    # box-drawing border characters, so compare against normalized output
    # (whitespace-collapsed, borders stripped) rather than a raw substring.
    stripped = result.output.translate({ord(c): " " for c in "│╭╮╰╯─"})
    normalized = " ".join(stripped.split())

    assert result.exit_code == 0, result.output
    # The old per-session list (and its heading) is gone entirely.
    assert "Biggest reclaim opportunities" not in result.output
    # The aggregate reclaim line inside the quota-composition panel never
    # names a session (#119). Scope this to the panel itself: the SEPARATE
    # statusline live preview section (#438) legitimately names one session
    # as a concrete "what you'd have seen live" example — a whole-output
    # check would false-positive against that unrelated, later section.
    panel_text = result.output.split("With the statusline installed")[0]
    assert "sess-heavy" not in panel_text
    # An aggregate line takes its place — no named session, live-signal framing.
    assert "ran context-heavy enough to warrant a mid-session" in normalized
    assert "/compact" in normalized
    assert "a closed session can't be reclaimed" in normalized


def test_quickstart_no_compact_candidates_omits_reclaim_section(tmp_path):
    """A history with no context-heavy sessions renders no reclaim section at
    all (same gating as before — this isn't about forcing the line to show)."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "ran context-heavy enough to warrant a mid-session" not in result.output
    assert "Biggest reclaim opportunities" not in result.output


def test_quickstart_json_emits_both_views(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    # The JSON line is the last line (Rich logging may precede it on stderr).
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "quota_composition" in payload
    assert "session_timeline" in payload
    assert payload["session_timeline"]["total_sessions"] == 2
    assert payload["backfill"]["sessions_ingested"] == 2


def test_quickstart_no_logs_is_graceful(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _invoke_quickstart(["--root", str(missing)])
    assert result.exit_code == 0, result.output
    assert "No Claude Code logs" in result.output


# ── Pre-ingest progress: ingest was previously the ONE silent stretch in the
# whole command (~40s dead cursor on a large history, nothing printed until
# after it returned). An honest status line now lands before ingest starts,
# and the shared streaming counter (`backfill_progress`) advances per session
# through to render. `--json` must stay byte-for-byte clean on stdout.

def test_quickstart_prints_pre_ingest_status_before_render(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "Reading your last 90 days of Claude Code history" in result.output
    assert "(most-recent 300 sessions)" in result.output
    # It's the FIRST thing printed -- ahead of the quota-composition panel,
    # not tacked on after ingest already finished.
    assert result.output.index("Reading your last 90 days") < result.output.index(
        "Where your quota goes"
    )


def test_quickstart_pre_ingest_status_omits_cap_when_full(tmp_path):
    """`--full` lifts the session cap (#13) -- the status line must not claim
    a "most-recent N sessions" scope that no longer applies."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--full"])

    assert result.exit_code == 0, result.output
    assert "Reading your last 90 days of Claude Code history…" in result.output
    assert "most-recent" not in result.output.split("Where your quota goes")[0]


def test_quickstart_json_stdout_stays_pure(tmp_path):
    """`--json` must be pipeable straight into a JSON parser: stdout carries
    ONLY the JSON payload, never the pre-ingest status line or the streaming
    progress counter -- those route to stderr instead."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    # stdout parses as JSON on its own -- no leading/trailing progress noise.
    payload = json.loads(result.stdout.strip())
    assert "quota_composition" in payload
    assert payload["backfill"]["sessions_ingested"] == 2
    # The status line still printed -- just on stderr, never stdout.
    assert "Reading your last 90 days of Claude Code history" in result.stderr
    assert "Reading your last 90 days" not in result.stdout


def test_quickstart_advancing_counter_on_large_history(tmp_path):
    """On a large history the shared streaming counter keeps advancing
    through ingest (not just a single static pre-ingest line) -- non-TTY
    output (as under CliRunner) degrades to periodic plain prints every 100
    sessions, mirroring `tj onboard --claude-code`'s backfill counter."""
    root = _large_fixture_root(tmp_path, n_sessions=250)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "Backfilling 100/250 sessions" in result.output
    assert "Backfilling 200/250 sessions" in result.output


def _large_fixture_root(tmp_path: Path, n_sessions: int) -> Path:
    """A synthetic history with `n_sessions` sessions, two turns each, recent.

    Mtimes are staggered so the most-recent-first cap is deterministic: higher
    session index = newer file. This lets the cap tests assert *which* sessions
    survive without depending on filesystem write ordering.
    """
    import os

    root = tmp_path / "projects"
    base_ts = 1_900_000_000  # arbitrary recent epoch
    for i in range(n_sessions):
        sid = f"sess-{i:05d}"
        cwd = f"/Users/me/proj{i % 5}"
        path = _make_session_file(root, sid, cwd, [
            _assistant(f"{sid}-a", sid, cwd, _ts(6, 20, "10:00:00.000")),
            _assistant(f"{sid}-b", sid, cwd, _ts(6, 20, "10:05:00.000")),
        ])
        # Newer index => newer mtime, so the cap keeps the highest indices.
        os.utime(path, (base_ts + i, base_ts + i))
    return root


# ── First-run cap on a large history (#13) ───────────────────────────────────

def test_quickstart_caps_sessions_on_large_history(tmp_path):
    """The first-run path bounds its work: only `max_sessions` are ingested even
    when far more exist on disk, and the cap is flagged."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=120)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=25)

    # Bounded work: exactly the cap was ingested, not the full 120.
    assert result.sessions_ingested == 25
    assert result.sessions_seen == 25
    assert result.limit_reached is True
    # The transient DB holds only the capped sessions' rows.
    (session_rows,) = db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert session_rows == 25


def test_quickstart_cap_keeps_most_recent_sessions(tmp_path):
    """The cap retains the freshest sessions (by mtime), not arbitrary ones."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=50)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root, max_sessions=10)

    kept = {
        r[0] for r in db.conn.execute("SELECT session_id FROM sessions").fetchall()
    }
    # The 10 highest indices (newest mtimes) survive; older ones are dropped.
    assert kept == {f"sess-{i:05d}" for i in range(40, 50)}


def test_quickstart_no_cap_ingests_everything(tmp_path):
    """`max_sessions=None` (the full `tj backfill claude-code` path) is unbounded
    and never sets the limit flag — the cap is opt-in, not a regression."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=40)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=None)

    assert result.sessions_ingested == 40
    assert result.limit_reached is False


def test_quickstart_below_cap_does_not_flag_limit(tmp_path):
    """A small history under the cap is not falsely reported as truncated."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=5)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=300)

    assert result.sessions_ingested == 5
    assert result.limit_reached is False


def test_quickstart_cli_discloses_truncation(tmp_path, monkeypatch):
    """When the cap truncates, the CLI says so honestly and points at the full
    picture — no silent truncation that reads as 'this is everything'."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 8)
    root = _large_fixture_root(tmp_path, n_sessions=30)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # The disclosure names the cap and points at the full-history escape hatch.
    # Assert on stable, non-wrapping tokens only: Rich word-wraps the inline
    # "npx tokenjam onboard" CTA across a line break at narrow widths (it lands
    # as "npx tokenjam \nonboard"), so asserting that literal is flaky. The
    # footer-CTA form is covered by the dedicated footer tests above.
    assert "most-recent" in result.output
    assert "tj context" in result.output
    assert "full history" in result.output


def test_quickstart_cli_full_flag_lifts_cap(tmp_path, monkeypatch):
    """`--full` processes the whole history and emits no truncation note."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 3)
    root = _large_fixture_root(tmp_path, n_sessions=12)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d",
                                 "--full", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backfill"]["sessions_ingested"] == 12
    assert payload["backfill"]["limit_reached"] is False
    assert payload["backfill"]["max_sessions"] is None


def test_quickstart_json_reports_cap_metadata(tmp_path, monkeypatch):
    """JSON output exposes the cap state so machine consumers see the scoping."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 6)
    root = _large_fixture_root(tmp_path, n_sessions=20)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backfill"]["sessions_ingested"] == 6
    assert payload["backfill"]["limit_reached"] is True
    assert payload["backfill"]["max_sessions"] == 6


# ── Statusline live preview ("what you'd see live") ──────────────────────────
#
# `_display_model_name` reconstructs "Sonnet 4.5" from the raw transcript
# `claude-sonnet-4-5-20250929` id every fixture session below uses.

def _flat(output: str) -> str:
    """Collapse Rich's line-wrapping so long-sentence substring checks aren't
    sensitive to where the terminal happened to wrap a word."""
    return " ".join(output.split())


def _session_with_crossing(root: Path, session_id: str, cwd: str, base_date: str,
                            *, n_turns: int, crossing_turn: int) -> Path:
    """A synthetic session whose cumulative re-read %% stays low through
    `crossing_turn - 1`, then jumps hard enough that it crosses REREAD_WARN
    (70%%) starting exactly at `crossing_turn` (1-indexed) and stays crossed.
    """
    records = []
    for i in range(1, n_turns + 1):
        ts = f"{base_date}T10:{i:02d}:00.000Z"
        cache = 20 if i < crossing_turn else 100_000
        records.append(_assistant(
            f"{session_id}-{i}", session_id, cwd, ts,
            input_tokens=100, output_tokens=50, cache_read=cache,
        ))
    return _make_session_file(root, session_id, cwd, records)


def test_quickstart_preview_shows_most_recent_substantial_crossing_session(
    tmp_path, monkeypatch,
):
    from tokenjam.cli import cmd_quickstart as q
    from tokenjam.cli.cmd_statusline import format_status_line

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    # Older, more turns, but NOT more recent -> must lose to the recent one.
    _session_with_crossing(
        root, "sess-old", "/Users/me/projA", _date(6, 10),
        n_turns=10, crossing_turn=2,
    )
    # Recent AND substantial (5 >= PREVIEW_MIN_TURNS=3) -> wins on recency.
    recent_path = _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", _date(6, 25),
        n_turns=5, crossing_turn=3,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "With the statusline installed" in flat
    assert "session sess-recent would have shown this at turn 3" in flat
    assert "tj onboard" in flat.split("With the statusline installed")[1]

    # Formatter reuse: the rendered line is byte-identical to what
    # format_status_line produces for the FIRST turn whose cumulative re-read
    # crosses the nudge threshold — never a hand-rolled duplicate of the live
    # statusline's text.
    from tokenjam.cli.cmd_statusline import REREAD_WARN
    from tokenjam.core.usage import iter_cumulative_usage
    with open(recent_path, encoding="utf-8") as fh:
        crossing = next(
            (turn_index, usage) for turn_index, _model, usage in iter_cumulative_usage(fh)
            if usage.total and 100.0 * usage.cache_read_tokens / usage.total >= REREAD_WARN
        )
    turn_index, usage = crossing
    assert turn_index == 3
    total = usage.total
    reread_pct = 100.0 * usage.cache_read_tokens / total
    expected_line = format_status_line("Sonnet 4.5", total, reread_pct)
    # Flatten Rich's line-wrapping (as the assertions above do) — the preview
    # reuses the same formatter, but the longer driver-conditional nudge can wrap
    # at the console width; the point is byte-identical *content*, not layout.
    assert _flat(expected_line) in _flat(result.output)


def test_quickstart_preview_stops_walking_after_first_substantial_candidate(
    tmp_path, monkeypatch,
):
    """Sessions are walked most-recent-first; once the most-recent SUBSTANTIAL
    crossing candidate is found, selection must stop rather than re-reading
    every remaining session's transcript — a large `~/.claude` history would
    otherwise blow past quickstart's fast-first-run budget."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    # Most-recent session is ALREADY substantial + crossing -> must win
    # without inspecting any of the five older sessions below it.
    _session_with_crossing(
        root, "sess-newest", "/Users/me/projZ", _date(6, 28),
        n_turns=5, crossing_turn=2,
    )
    for i in range(5):
        _session_with_crossing(
            root, f"sess-old-{i}", f"/Users/me/proj{i}", _date(6, 10 + i),
            n_turns=5, crossing_turn=2,
        )

    calls: list[str] = []
    original_walk = q._walk_for_preview

    def _counting_walk(path):
        calls.append(path)
        return original_walk(path)

    monkeypatch.setattr(q, "_walk_for_preview", _counting_walk)

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "session sess-newest" in _flat(result.output)
    # Only the winning (most-recent, already-substantial) candidate's
    # transcript was walked -- the 5 older sessions were never opened.
    assert len(calls) == 1


def test_quickstart_preview_falls_back_to_largest_when_none_substantial(
    tmp_path, monkeypatch,
):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 50)  # neither session qualifies
    root = tmp_path / "projects"
    _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", _date(6, 25),
        n_turns=5, crossing_turn=3,
    )
    _session_with_crossing(
        root, "sess-old", "/Users/me/projA", _date(6, 10),
        n_turns=8, crossing_turn=5,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    # Neither is "substantial" -> falls back to the largest (by turns), not
    # simply the most recent.
    flat = _flat(result.output)
    assert "session sess-old would have shown this at turn 5" in flat


def test_quickstart_preview_omitted_when_no_session_crosses_threshold(tmp_path):
    root = tmp_path / "projects"
    # Healthy sessions: tiny cache reads relative to input/output, never near
    # the 70% nudge threshold.
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", _ts(6, 20, "10:00:00.000"),
                   input_tokens=1000, output_tokens=200, cache_read=10),
    ])

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "With the statusline installed" not in result.output


def test_quickstart_preview_omitted_when_no_sessions(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _invoke_quickstart(["--root", str(missing)])
    assert result.exit_code == 0, result.output
    assert "With the statusline installed" not in result.output


# ── Both-personas line + past-overspend callout ────────────────────────────
#
# What replaced the Session Story teaser and the per-session timeline table.
# Two jobs: say out loud that tokenjam serves Claude Code users AND Anthropic
# SDK / API traffic (they arrive by different ingest paths), and close on the
# single largest thing this history already overspent on, with the fix.
#
# Honesty rules under test, all from the repo CLAUDE.md field contract:
#   * `past_overspend_usd` is the canonical figure, observed over the window,
#     never paced and never summed with `observed_cost_usd`;
#   * the SINGLE LARGEST finding is shown, never a sum across analyzers;
#   * `None` means "not measured", never `$0` — a corpus with nothing to
#     report renders no callout at all rather than a fabricated figure.


def _cheapest_model_assistant(uuid: str, session_id: str, cwd: str,
                              ts: str) -> dict:
    """An assistant turn on the cheapest model tokenjam prices, so no analyzer
    has a cheaper alternative to propose."""
    record = _assistant(uuid, session_id, cwd, ts,
                        input_tokens=200, output_tokens=80, cache_read=400)
    record["message"]["model"] = "claude-haiku-4-5"
    return record


def _proposal(*, analyzer="deadweight", title="Unused MCP server: posthog",
              evidence="0 tool calls across 12 sessions.",
              advise_text="Remove the `posthog` MCP server.",
              usd=None, tokens=None, caveat="Estimated, correlational figure."):
    from tokenjam.core.optimize.cost_proposals import CostProposal

    return CostProposal(
        kind="cost",
        analyzer=analyzer,
        signature=f"cost:{analyzer}",
        title=title,
        target_key={},
        evidence=evidence,
        baseline={},
        advise_text=advise_text,
        past_overspend_usd=usd,
        past_overspend_tokens=tokens,
        caveat=caveat,
    )


class _StubWindow:
    def __init__(self, total_cost_usd: float) -> None:
        self.total_cost_usd = total_cost_usd


class _StubReport:
    def __init__(self, total_cost_usd: float) -> None:
        self.window = _StubWindow(total_cost_usd)


def _patch_optimize(monkeypatch, proposals, window_cost=1000.0):
    """Stub the two optimize entry points `_compute_past_overspend` imports.

    Both imports happen inside the function body, so patching the modules they
    resolve from is enough — no import-order games. Everything under test here
    is the SELECTION (largest, defensible, priced-or-token) and the render, not
    the analyzers themselves; those have their own suites.
    """
    import tokenjam.core.optimize as opt
    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(opt, "build_report",
                        lambda **kwargs: _StubReport(window_cost))
    monkeypatch.setattr(cp, "cost_proposals_from_report",
                        lambda *args, **kwargs: list(proposals))


def _compute(monkeypatch, proposals, window_cost=1000.0):
    from datetime import timedelta

    from tokenjam.cli.cmd_quickstart import _compute_past_overspend

    _patch_optimize(monkeypatch, proposals, window_cost=window_cost)
    return _compute_past_overspend(object(), _NOW - timedelta(days=30), _NOW)


def test_past_overspend_picks_the_single_largest_priced_finding(monkeypatch):
    callout = _compute(monkeypatch, [
        _proposal(analyzer="subagent", title="Right-size a subagent", usd=4.0,
                  tokens=100),
        _proposal(analyzer="deadweight", title="Unused MCP server: posthog",
                  usd=42.5, tokens=900,
                  advise_text="Remove the `posthog` MCP server."),
        _proposal(analyzer="summarize", title="Review 3 oversized files",
                  usd=9.0, tokens=300),
    ])

    assert callout is not None
    # The largest, NOT the 55.5 sum of the three (analyzers price overlapping
    # angles on the same spans, so summing double-counts).
    assert callout.usd == 42.5
    assert callout.title == "Unused MCP server: posthog"
    assert callout.advise_text == "Remove the `posthog` MCP server."


def test_past_overspend_drops_a_figure_larger_than_the_window_cost(monkeypatch):
    """A figure bigger than what the whole window cost is self-refuting: this
    render prints the window's implied API value a few lines below it. The
    over-ceiling finding is DROPPED, never rescaled into a paced number."""
    callout = _compute(monkeypatch, [
        _proposal(analyzer="deadweight", title="Unused MCP server", usd=357.0),
        _proposal(analyzer="summarize", title="Review 3 oversized files", usd=88.0),
    ], window_cost=245.0)

    assert callout is not None
    assert callout.usd == 88.0


def test_past_overspend_falls_back_to_tokens_below_the_dollar_floor(monkeypatch):
    callout = _compute(monkeypatch, [
        _proposal(usd=0.12, tokens=250_000),
    ])

    assert callout is not None
    assert callout.usd is None            # never a fabricated / near-zero figure
    assert callout.tokens == 250_000
    assert "under $1" in callout.tokens_only_reason


def test_past_overspend_falls_back_to_tokens_when_never_priced(monkeypatch):
    """`None` means "not measured", never `$0` — so an unpriced finding shows
    its token figure and says why there is no dollar figure."""
    callout = _compute(monkeypatch, [
        _proposal(usd=None, tokens=800_000),
    ])

    assert callout is not None
    assert callout.usd is None
    assert callout.tokens == 800_000
    assert "no priced figure" in callout.tokens_only_reason


def test_past_overspend_is_none_when_nothing_qualifies(monkeypatch):
    assert _compute(monkeypatch, []) is None
    assert _compute(monkeypatch, [_proposal(usd=None, tokens=None)]) is None
    assert _compute(monkeypatch, [_proposal(usd=None, tokens=0)]) is None


def test_past_overspend_never_dies_on_the_analyzers(monkeypatch):
    """A first run must degrade, not crash, if the analyzers raise."""
    import tokenjam.core.optimize as opt

    def _boom(**kwargs):
        raise RuntimeError("analyzer exploded")

    monkeypatch.setattr(opt, "build_report", _boom)

    from datetime import timedelta

    from tokenjam.cli.cmd_quickstart import _compute_past_overspend

    assert _compute_past_overspend(object(), _NOW - timedelta(days=30), _NOW) is None


def test_past_overspend_strips_a_duplicate_money_clause_from_the_title(monkeypatch):
    """Several proposal titles carry their own money suffix (the Review inbox
    renders them beside no figure). This callout LEADS with the figure, so the
    suffix would print two differently-rounded amounts in one sentence."""
    callout = _compute(monkeypatch, [
        _proposal(title="Review 17 oversized files, $187.55", usd=187.55),
    ])

    assert callout is not None
    assert callout.title == "Review 17 oversized files"


def test_past_overspend_falls_back_to_optimize_when_advice_is_unusable(monkeypatch):
    """The persona gate substitutes a "no fix is shown for it" string for
    findings this persona can't act on. Surfacing that as "the fix" would be
    worse than pointing at the full report."""
    from tokenjam.core.optimize.cost_proposals import CACHE_NO_LEVER_TEXT

    blank = _compute(monkeypatch, [_proposal(advise_text="   ", usd=5.0)])
    gated = _compute(monkeypatch, [_proposal(advise_text=CACHE_NO_LEVER_TEXT, usd=5.0)])

    assert blank is not None and gated is not None
    assert blank.advise_text == "Run `tj optimize` for this finding and its fix."
    assert gated.advise_text == blank.advise_text


# ── The rendered block ─────────────────────────────────────────────────────


def test_quickstart_names_both_ingest_paths_in_one_line(tmp_path):
    """tokenjam is not a Claude-Code-only tool: the report says so once, in
    terms of the ingest path each persona actually arrives by. MCP is the
    SDK/API path — Claude Code arrives out of band (these local logs)."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "TokenJam ingests both sides of your spend" in flat
    assert "Claude Code from these local session logs" in flat
    assert "Anthropic SDK / API traffic from OTel spans or the tokenjam MCP server" in flat


def test_quickstart_renders_the_past_overspend_callout(tmp_path, monkeypatch):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "_compute_past_overspend", lambda *a, **k: q.PastOverspendCallout(
        title="Unused MCP server: posthog",
        evidence="`posthog` made 0 tool calls across 12 session(s).",
        advise_text="Remove or project-scope the `posthog` MCP server.",
        caveat="Estimated, correlational figure.",
        usd=42.50,
    ))
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "What that already cost you" in flat
    assert "$42.50 of that was avoidable: Unused MCP server: posthog" in flat
    assert "0 tool calls across 12 session(s)" in flat
    assert "Fix: Remove or project-scope the `posthog` MCP server." in flat
    assert "Estimated, correlational figure." in flat


def test_quickstart_past_overspend_token_form_states_why_no_dollars(
    tmp_path, monkeypatch,
):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "_compute_past_overspend", lambda *a, **k: q.PastOverspendCallout(
        title="Right-size a subagent",
        evidence="3 subagents ran on a premium model.",
        advise_text="Set `model:` in the agent file.",
        tokens=250_000,
        tokens_only_reason="under $1 at API list rates over this window",
    ))
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "250.0k tokens of that was avoidable" in flat
    assert "No dollar figure: under $1 at API list rates over this window." in flat
    # Never a fabricated / zero dollar figure alongside the token form.
    assert "$0.00" not in result.output


def test_quickstart_degrades_cleanly_when_nothing_is_recoverable(tmp_path):
    """The REAL analyzer path on a corpus with nothing to report: no panel, no
    `$0.00`, no fabricated number, and the rest of the report still renders.

    The fixture already runs on the cheapest model in the pricing table, so
    `downsize` has nothing to route it down to, and an isolated HOME (see
    `tests/conftest.py`) leaves every config-reading analyzer with nothing to
    find. No stubbing: this is the path a first-time user with a clean, small
    history actually takes.
    """
    root = tmp_path / "projects"
    _make_session_file(root, "sess-cheap", "/Users/me/projCheap", [
        _cheapest_model_assistant("c1", "sess-cheap", "/Users/me/projCheap",
                                  _ts(6, 20, "10:00:00.000")),
    ])
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "What that already cost you" not in result.output
    assert "of that was avoidable" not in result.output
    # No fabricated / zero figure anywhere above the totals line (the totals
    # line's own "implied API value" is a different, pre-existing figure).
    assert "$0.00" not in result.output.split("Totals:")[0]
    # The surrounding report is unaffected.
    assert "where your quota actually goes" in _flat(result.output)
    assert "Totals:" in result.output


def test_quickstart_no_longer_teases_session_story(tmp_path, monkeypatch):
    """The Session Story teaser is gone from the first run (`tj session-story`
    itself is untouched and still a real command)."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", _date(6, 25),
        n_turns=5, crossing_turn=3,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # The statusline preview it used to hang off of still renders.
    assert "With the statusline installed" in result.output
    assert "Session Story" not in result.output
    assert "tj session-story" not in result.output


def test_quickstart_user_facing_copy_has_no_em_dashes(tmp_path):
    """Standing copy rule for tokenjam: periods, semicolons or colons, never an
    em dash.

    Scope: quickstart's OWN copy, which is what this fixture exercises (no
    analyzer finding qualifies on it). The past-overspend callout renders a
    winning proposal's `advise_text` and `caveat` VERBATIM, and those strings
    are authored in `core/optimize/cost_proposals.py` for every surface
    (CLI, Review inbox, web UI); several still contain em dashes. They are
    deliberately not rewritten at render time: the honesty caveats are
    contractually surfaced verbatim, so the fix belongs at the source, in a
    change that moves every surface at once.
    """
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "—" not in result.output


def test_overspend_analyzer_set_drops_only_relearn():
    """Runtime bound, not a second persona filter: `relearn` reports on
    `cost_of_waste_*` and leaves `past_overspend_*` at None, so it can never
    win this selection, while costing the large majority of the analyzer time
    on a real corpus. Everything else in `COST_ANALYZERS` is passed straight
    through, and the persona gate stays inside `build_report`."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS)

    assert "relearn" not in selected
    assert selected == [n for n in COST_ANALYZERS if n != "relearn"]


def test_overspend_analyzer_set_keeps_deadweight_when_not_capped():
    """`deadweight` scans every matching transcript on disk regardless of
    `--max-sessions`, but when the ingest was NOT truncated its population
    is identical to what got ingested — no reason to drop it."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS, population_capped=False)

    assert "deadweight" in selected


def test_overspend_analyzer_set_drops_deadweight_when_population_capped():
    """When quickstart's session ingest actually truncated at the cap,
    `deadweight`'s own disk scan reasons over strictly MORE sessions than the
    ones ingested and rendered — a population mismatch the magnitude ceiling
    alone can't catch (a smaller out-of-population figure still clears it).
    Excluding the analyzer, not rescaling its figure, is the only honest
    move here."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS, population_capped=True)

    assert "deadweight" not in selected
    assert "relearn" not in selected
    assert selected == [n for n in COST_ANALYZERS if n not in ("relearn", "deadweight")]


def test_past_overspend_excludes_deadweight_from_the_report_when_population_capped(monkeypatch):
    """`_compute_past_overspend(population_capped=True)` must not even ASK
    `build_report` to run `deadweight` — filtering its proposal out after
    the fact would still let evidence/cost from out-of-window sessions leak
    into the report object."""
    import tokenjam.core.optimize as opt
    from tokenjam.cli.cmd_quickstart import _compute_past_overspend

    captured: dict = {}

    def _fake_build_report(**kwargs):
        captured["findings"] = kwargs.get("findings")
        return _StubReport(1000.0)

    monkeypatch.setattr(opt, "build_report", _fake_build_report)
    import tokenjam.core.optimize.cost_proposals as cp
    monkeypatch.setattr(cp, "cost_proposals_from_report", lambda *a, **k: [])

    _compute_past_overspend(
        object(), _NOW - timedelta(days=30), _NOW, population_capped=True,
    )

    assert "deadweight" not in captured["findings"]

    _compute_past_overspend(
        object(), _NOW - timedelta(days=30), _NOW, population_capped=False,
    )

    assert "deadweight" in captured["findings"]


def test_quickstart_reads_no_config_and_opens_no_ondisk_db(tmp_path, monkeypatch):
    """The zero-setup promise, now that the run also builds an optimize report:
    the analyzers get an in-memory `TjConfig()` default, so no config file is
    read or written and no on-disk DB is opened."""
    import tokenjam.cli.main as main
    import tokenjam.core.config as cfg

    def _boom(*args, **kwargs):
        raise AssertionError("quickstart must not touch config / the on-disk DB")

    monkeypatch.setattr(cfg, "load_config", _boom)
    monkeypatch.setattr(cfg, "find_config_file", _boom)
    monkeypatch.setattr(cfg, "write_config", _boom, raising=False)
    monkeypatch.setattr(main, "open_db", _boom, raising=False)

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
