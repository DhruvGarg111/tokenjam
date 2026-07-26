"""The zero-install, zero-config first run (issue #6).

The 15-second time-to-first-value path that lets a brand-new user — reached via
``npx tokenjam`` / ``uvx tj`` with **no** pip env, **no** daemon, **no** onboarding —
see where their Claude Code quota actually goes, straight from the JSONL files
ccusage already reads (``~/.claude/projects/*.jsonl``).

Design (what makes it "zero-setup"):

  * It opens a **transient in-memory** DuckDB (``InMemoryBackend``) — nothing is
    written to ``~/.tj``, no config is read or written, no daemon is started or
    contacted. Each run re-reads the JSONL fresh.
  * It backfills the on-disk Claude Code sessions into that transient DB via the
    existing :func:`tokenjam.core.backfill.ingest_claude_code` parser, then runs
    the same two read-only views the paid-deeper path exposes:
      - quota composition (re-reading vs. net-new work) from
        :mod:`tokenjam.core.context_diagnostic` (issue #4's engine, reused);
      - a session timeline from :mod:`tokenjam.core.session_timeline`
        (the `--json` payload and the statusline preview's session pick;
        the human render shows totals, not a per-session table);
      - the largest single past-overspend finding plus its fix, from the
        same ``build_report`` / ``COST_ANALYZERS`` path the Review inbox
        and ``tj status`` use.
  * The output **leads with reads-your-local-logs + added-value framing** —
    "reads your ~/.claude session logs; here's where your quota actually goes" —
    then ends on the opt-in "go deeper" pointer to ``tj onboard`` (daemon /
    statusline / live capture).

This has no public/typeable command name — ``cli/main.py``'s no-subcommand
branch invokes ``cmd_quickstart`` directly (via ``ctx.invoke``) when the npm
wrapper's ``TJ_NPX_ZERO_INSTALL_REPORT`` env var is set, so it never opens the
on-disk DB or trips the daemon's write lock either way.

Honesty discipline (CLAUDE.md Rule 14): every figure here is a *measured* token
share re-derived from the JSONL, never a projected saving.
"""
from __future__ import annotations

import glob as _glob
import json as _json
import re as _re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

from tokenjam.cli.backfill_progress import backfill_progress
from tokenjam.cli.cmd_statusline import REREAD_WARN, format_status_line
from tokenjam.core.backfill import (
    CLAUDE_CODE_PROJECTS_ROOT,
    count_claude_code_sessions_in_scope,
    ingest_claude_code,
)
from tokenjam.core.context_diagnostic import compute_context_diagnostic
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    SessionTimeline,
    TimelineSession,
    compute_session_timeline,
    timeline_to_dict,
)
from tokenjam.core.usage import AssistantUsage, iter_cumulative_usage
from tokenjam.utils.formatting import console, err_console, format_cost, format_tokens
from tokenjam.utils.time_parse import parse_since, utcnow

# First-run cap (#13): on a large ~/.claude history a full backfill into the
# transient DB blows past the <30s time-to-first-value goal. We cap the headline
# to the most-recent N sessions (bounded work, well under 30s even on thousands
# of sessions) and disclose the cap; `--full` lifts it for the complete picture.
# ~300 sessions keeps the slowest plausible session shapes comfortably in budget.
DEFAULT_MAX_SESSIONS = 300

# "Substantial" floor for the statusline live-preview (#120-adjacent): the
# most-recent session is only worth previewing the nudge on if it actually ran
# long enough to feel like a real session, not a two-turn smoke test. Below
# this we fall back to the largest recent session that crossed the threshold.
PREVIEW_MIN_TURNS = 20

# Floor for the past-overspend callout's dollar figure. Mirrors
# `cmd_status._TEASER_MIN_USD`: below a dollar the figure reads as noise and
# invites the reader to dismiss the whole report, so we degrade to the token
# figure (with the reason stated) rather than print a near-zero headline.
# A `$0.00` is never printed: `None` means "not measured", never zero.
OVERSPEND_MIN_USD = 1.0


@click.command("quickstart")
@click.option("--since", default="30d",
              help="Window for analysis (e.g. 7d, 30d, 2026-03-01). Default 30d.")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--full", is_flag=True,
              help=f"Process the full history (default caps at the most-recent "
                   f"{DEFAULT_MAX_SESSIONS} sessions for a fast first run).")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_quickstart(ctx: click.Context, since: str, root_path: str | None,
                   full: bool, output_json: bool) -> None:
    """Zero-setup first run: where your Claude Code quota actually goes.

    Reads the same ~/.claude/projects/*.jsonl files ccusage does: no pip env,
    no daemon, no onboarding. On a large history the first run caps at the
    most-recent sessions for speed (use `--full` for everything). Run
    `tj onboard` afterwards to go deeper (live capture, the dashboard, and the
    zero-token statusline).
    """
    from pathlib import Path

    root = Path(root_path).expanduser() if root_path else CLAUDE_CODE_PROJECTS_ROOT
    if not root.exists():
        _render_no_logs(root, output_json)
        return

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    until_dt = utcnow()

    # Transient in-memory DB — nothing persisted, no config, no daemon.
    max_sessions = None if full else DEFAULT_MAX_SESSIONS
    db = InMemoryBackend()

    # Ingest is the only silent stretch in the whole command — on a large
    # history it can run tens of seconds with zero output otherwise. An
    # honest status line lands within ~1s of launch, then the shared
    # streaming counter (#443/#444's `backfill_progress`) advances per
    # session through to render. `--json` must keep stdout byte-for-byte
    # clean, so both route to the stderr console when JSON is requested —
    # never suppressed outright, so a human watching a scripted run still
    # sees it's alive.
    status_console = err_console if output_json else console
    # Best-effort pre-scan: a stat()-only count taken before ingest starts, so
    # the progress counter's "of N" denominator can drift if files under
    # `root` change mid-run (a session file appears/disappears between this
    # count and the actual walk). Cosmetic only — never affects what's
    # ingested, since `ingest_claude_code` re-walks `root` itself.
    total_in_scope = count_claude_code_sessions_in_scope(
        root=root, since=since_dt, max_sessions=max_sessions,
    )
    status_console.print(f"[dim]{_pre_ingest_status(since, max_sessions)}[/dim]")
    with backfill_progress(total_in_scope, console=status_console) as progress_cb:
        result = ingest_claude_code(db, root=root, since=since_dt,
                                    max_sessions=max_sessions, progress=progress_cb)

    if result.sessions_ingested == 0:
        _render_no_sessions(result, since, output_json)
        return

    diag = compute_context_diagnostic(db.conn, since_dt, until_dt)
    timeline = compute_session_timeline(db.conn)

    if output_json:
        from tokenjam.core.context_diagnostic import diagnostic_to_dict
        payload = {
            "quota_composition": diagnostic_to_dict(diag),
            "session_timeline": timeline_to_dict(timeline),
            "backfill": {
                "sessions_ingested": result.sessions_ingested,
                "spans_ingested": result.spans_ingested,
                "project_count": result.project_count,
                "total_cost_usd": round(result.total_cost_usd, 6),
                "limit_reached": result.limit_reached,
                "max_sessions": max_sessions,
            },
        }
        click.echo(_json.dumps(payload, default=str))
        return

    # Computed only on the human path: `--json` must stay byte-clean AND fast,
    # and the analyzers are the one materially expensive step after ingest.
    overspend = _compute_past_overspend(
        db, since_dt, until_dt,
        population_capped=max_sessions is not None and result.limit_reached,
    )

    _render(diag, timeline, since=since,
            limit_reached=result.limit_reached, max_sessions=max_sessions,
            root=root, overspend=overspend)


# ────────────────────────── past overspend ────────────────────────────────
#
# The single largest ALREADY-INCURRED, avoidable finding over the ingested
# window, with the concrete fix it ends in. Quickstart is a report, so this is
# the one place it tells a first-run user what their history already cost them
# in a unit they reason in.
#
# Contract (repo CLAUDE.md, "THE per-analyzer dollar-field contract"):
#   * `past_overspend_usd` is the canonical figure, observed over the analyzed
#     window, past tense. It is NEVER paced, projected, or multiplied by a
#     30-day ratio, and `observed_cost_usd` is never summed into it.
#   * `None` means "not measured", never `$0`.
#   * The SINGLE LARGEST proposal is shown, never a sum: the analyzers price
#     overlapping angles on the same spans, so summing double-counts (same
#     rationale as `cmd_status._recoverable_teaser` and
#     `api/routes/cost._recoverable_overlap_note`).
#
# Analyzer selection is NOT re-implemented here, and there is no second
# persona filter. `build_report` is handed `COST_ANALYZERS` and does the
# persona gating itself (`PERSONA_DISABLED_ANALYZERS`, a TRUE skip before
# dispatch), so a Claude-Code-dominant window never spends query time on a
# finding that persona could not act on. The one deviation is a runtime bound
# for this zero-install path, documented at `_OVERSPEND_SKIP_ANALYZERS`.
#
# Pricing: quickstart reads no config, so a plan tier is never declared and
# `pricing_mode_for(dominant_plan(...))` would be `"unknown"`. `cmd_status`'s
# stricter api-only gate is deliberately NOT reused: the product stance is to
# assume API pricing, and this same render already prints an "implied API
# value" total from the very same window.


@dataclass(frozen=True)
class PastOverspendCallout:
    """One finding's already-incurred avoidable spend, plus its fix.

    Exactly one of `usd` / `tokens` drives the headline: `usd` when the
    finding priced a figure at or above `OVERSPEND_MIN_USD`, otherwise
    `tokens` with `tokens_only_reason` stating why no dollar figure is shown.
    A callout with neither is never constructed (the caller omits the block).
    """
    title:      str
    evidence:   str
    advise_text: str
    caveat:     str = ""
    usd:        float | None = None
    tokens:     int | None = None
    tokens_only_reason: str = ""


# Shown instead of `advise_text` when the proposal carries no usable fix text
# (blank, or the persona-gated "there is no lever here" substitution).
_OVERSPEND_FALLBACK_ADVICE = "Run `tj optimize` for this finding and its fix."


# Several cost-proposal titles end in their own money clause ("Review 17
# oversized files, $187.55") because the Review inbox renders the title on a
# row with no figure beside it. This callout LEADS with the figure, so the
# suffix would print two differently-rounded dollar amounts in one sentence.
_TITLE_MONEY_SUFFIX = _re.compile(
    r",\s*(?:~?\$[\d,]+(?:\.\d+)?[kKmMbB]?|~?[\d,]+\s*tok)\s*$"
)


def _strip_money_suffix(title: str) -> str:
    """`title` without a trailing ", $12.34" / ", ~1,234 tok" clause."""
    return _TITLE_MONEY_SUFFIX.sub("", title).strip()


# The ONE analyzer dropped from `COST_ANALYZERS` for this callout, and the
# only one: `relearn`.
#
# This is a runtime bound, not a second persona filter — persona gating stays
# entirely inside `build_report` (`PERSONA_DISABLED_ANALYZERS`, a true skip
# before dispatch), which is why the full tuple is passed through this helper
# rather than rebuilt.
#
# Why it is safe to drop: `_relearn_to_proposals` deliberately leaves
# `past_overspend_usd`/`_tokens` at None and reports its figure on
# `cost_of_waste_*`, which `_with_past_overspend` routes onto
# `observed_cost_*` (relearn scans unbounded history and has no fixed window
# to observe an avoidable figure over). This callout selects strictly on
# `past_overspend_*`, so relearn can never be the winner. Measured on a real
# 300-session corpus it cost 27.9s of the 35.0s the whole set took, for a
# finding structurally ineligible here. If relearn ever moves onto the
# canonical field (the retirement sequence in the repo CLAUDE.md), delete this
# helper and pass `COST_ANALYZERS` whole.
_OVERSPEND_SKIP_ANALYZERS = frozenset({"relearn"})

#: Analyzers that reason over the raw on-disk transcript tree directly
#: (`root.rglob(...)`), never `ctx.conn` — so their population is NOT bounded
#: by `--max-sessions`/`DEFAULT_MAX_SESSIONS`. `deadweight` is the one in
#: `COST_ANALYZERS` that does this (`compute_deadweight_finding` walks every
#: matching transcript under the projects root for the window). When
#: quickstart's ingest actually truncated at the session cap
#: (`result.limit_reached`), a disk-scan analyzer's figure covers strictly
#: MORE sessions than the ones ingested into the DB and rendered on screen —
#: a population mismatch the existing magnitude ceiling (`_over_ceiling`)
#: cannot catch, since a SMALLER out-of-population figure still clears it.
#: Excluded outright rather than rescaled: there is no honest way to shrink
#: an unbounded-population figure down to the capped one without inventing a
#: number nothing measured.
_POPULATION_UNBOUNDED_ANALYZERS = frozenset({"deadweight"})


def _overspend_analyzers(names: tuple[str, ...], *, population_capped: bool = False) -> list[str]:
    """`names` minus the analyzers that cannot produce a past-overspend figure,
    and — only when the session ingest was actually capped — minus the ones
    whose own population isn't bounded by that cap either."""
    skip = _OVERSPEND_SKIP_ANALYZERS
    if population_capped:
        skip = skip | _POPULATION_UNBOUNDED_ANALYZERS
    return [n for n in names if n not in skip]


def _usable_advice(text: str | None) -> str:
    """`text` if it actually tells the reader what to change, else the pointer.

    The persona gate substitutes a "no fix is shown for it" string for cache
    findings a Claude Code user cannot act on (`cost_proposals.
    CACHE_NO_LEVER_TEXT`); surfacing that as "the fix" would be worse than
    pointing at the full report.
    """
    cleaned = (text or "").strip()
    if not cleaned or "no fix is shown" in cleaned:
        return _OVERSPEND_FALLBACK_ADVICE
    return cleaned


def _compute_past_overspend(
    db, since: datetime, until: datetime, *, population_capped: bool = False,
) -> PastOverspendCallout | None:
    """The largest single `past_overspend_usd` finding over the ingested window.

    Returns None when there is nothing honest to say: the analyzers found no
    priced or token-bearing finding, or anything at all raised. Never raises
    and never fabricates a figure, so a corpus with nothing to report simply
    renders no callout.

    `population_capped` is True only when the ingest itself truncated at the
    session cap (`result.limit_reached`) — see `_POPULATION_UNBOUNDED_ANALYZERS`
    for why that also has to drop any analyzer whose own data source ignores
    the cap.

    The config is an in-memory `TjConfig()` default: quickstart reads and
    writes NO config file and NO on-disk DB, and that must stay true.
    """
    try:
        from tokenjam.core.config import TjConfig
        from tokenjam.core.optimize import build_report
        from tokenjam.core.optimize.cost_proposals import (
            COST_ANALYZERS,
            cost_proposals_from_report,
        )

        # In-memory defaults only. Matches what `load_config` synthesises when
        # no config file exists, so no file is read and none is written.
        config = TjConfig(version="1")
        report = build_report(
            db=db, config=config, since=since, until=until,
            findings=_overspend_analyzers(COST_ANALYZERS, population_capped=population_capped),
        )
        window_days = max((until - since).total_seconds() / 86400.0, 1.0)
        proposals = cost_proposals_from_report(
            report, config, window_days=window_days,
        )
    except Exception:
        # A first run must never die on the analyzers. Worst case the callout
        # is absent; the rest of the report is unaffected.
        return None

    def _build(proposal, **kwargs) -> PastOverspendCallout:
        return PastOverspendCallout(
            title=_strip_money_suffix(str(getattr(proposal, "title", "") or "")),
            evidence=str(getattr(proposal, "evidence", "") or "").strip(),
            advise_text=_usable_advice(getattr(proposal, "advise_text", "")),
            caveat=str(getattr(proposal, "caveat", "") or "").strip(),
            **kwargs,
        )

    # Defensibility ceiling. This render prints the window's own "implied API
    # value" a few lines below the callout, so an avoidable figure LARGER than
    # what the window cost in total is self-refuting on the same screen, no
    # matter how the analyzer derived it. (It happens: some analyzers count a
    # population wider than what quickstart ingested, e.g. a per-session tax
    # summed over every transcript on disk while the transient DB holds the
    # capped, most-recent subset.) Matrix rule: show the LARGEST number the
    # derivation can legitimately support. A figure a reader can disprove by
    # looking one line down is not one of them, so it is dropped rather than
    # rescaled: rescaling would invent a number nothing measured.
    window_cost = float(getattr(getattr(report, "window", None),
                                "total_cost_usd", 0.0) or 0.0)

    def _over_ceiling(p) -> bool:
        usd = p.past_overspend_usd
        return usd is not None and window_cost > 0.0 and usd > window_cost

    eligible = [p for p in proposals if not _over_ceiling(p)]

    priced = [
        p for p in eligible
        if p.past_overspend_usd is not None
        and p.past_overspend_usd >= OVERSPEND_MIN_USD
    ]
    if priced:
        best = max(priced, key=lambda p: p.past_overspend_usd or 0.0)
        return _build(best, usd=best.past_overspend_usd,
                      tokens=best.past_overspend_tokens)

    # No figure clears the floor. Degrade to the token figure and say why,
    # rather than print a number that reads as nothing.
    tokened = [p for p in eligible if (p.past_overspend_tokens or 0) > 0]
    if not tokened:
        return None
    best = max(tokened, key=lambda p: p.past_overspend_tokens or 0)
    reason = (
        "under $1 at API list rates over this window"
        if best.past_overspend_usd is not None
        else "this finding carries no priced figure"
    )
    return _build(best, tokens=best.past_overspend_tokens,
                  tokens_only_reason=reason)


# ───────────────────────────── rendering ──────────────────────────────────

_SINCE_UNIT_WORDS = {"d": "days", "h": "hours", "m": "minutes"}


def _describe_window(since: str) -> str:
    """Human-readable window phrasing for the pre-ingest status line.

    Special-cases the relative `Nd`/`Nh`/`Nm` shapes `--since` accepts (the
    default is `30d`) into "last N days"; anything else (a literal date, an
    ISO datetime) falls back to "history since <value>" rather than guessing.
    """
    m = _re.match(r"^(\d+)([mhd])$", since.strip())
    if m:
        amount, unit = m.groups()
        return f"last {amount} {_SINCE_UNIT_WORDS[unit]}"
    return f"history since {since}"


def _pre_ingest_status(since: str, max_sessions: int | None) -> str:
    """Honest status line printed BEFORE ingest starts.

    Ingest was previously the one silent stretch in the whole command —
    ~40s of dead cursor on a large history before any output. This line
    lands within ~1s of launch; `backfill_progress`'s streaming counter
    takes over immediately after.
    """
    window = _describe_window(since)
    scope = f" (most-recent {max_sessions} sessions)" if max_sessions is not None else ""
    return f"Reading your {window} of Claude Code history{scope}…"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_no_logs(root, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_claude_code_logs", "root": str(root)}))
        return
    console.print(
        f"\n[yellow]No Claude Code logs found at {root}.[/yellow]\n"
        "[dim]This reads your ~/.claude/projects/*.jsonl session logs. This is "
        "normal if Claude Code hasn't run on this machine yet. Use it for a "
        "session, then run [bold]npx tokenjam[/bold] again. Ready to go "
        "deeper now? [bold]npx tokenjam onboard[/bold].[/dim]\n"
    )


def _render_no_sessions(result, since: str, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_sessions_in_window", "since": since}))
        return
    console.print(
        f"\n[yellow]No Claude Code sessions in the last {since}.[/yellow]\n"
        "[dim]Run [bold]npx tokenjam onboard[/bold] to go deeper; it wires "
        "up live capture so [bold]tj context[/bold] can show a wider "
        "window.[/dim]\n"
    )


def _render(diag, timeline, *, since: str,
            limit_reached: bool = False, max_sessions: int | None = None,
            root: Path | None = None,
            overspend: PastOverspendCallout | None = None) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    # ── Lead: reads-your-local-logs + added-value framing. ──
    console.print()
    lead = Text()
    lead.append("TokenJam reads your ", style="dim")
    lead.append("~/.claude/projects/*.jsonl", style="bold")
    lead.append(" session logs and shows you ", style="dim")
    lead.append("where your quota actually goes", style="bold")
    lead.append(".", style="dim")
    console.print(lead)

    # Honest disclosure when the first-run cap truncated the history (#13). This
    # must read as scoping, NOT as "this is your whole history" — so we say so up
    # front and point at the full-picture escape hatches.
    if limit_reached and max_sessions is not None:
        note = Text()
        note.append("Showing your most-recent ", style="yellow")
        note.append(f"{max_sessions} sessions", style="bold yellow")
        note.append(" for a fast first run. Run ", style="yellow")
        note.append("npx tokenjam onboard", style="bold")
        note.append(", then ", style="yellow")
        note.append("tj context", style="bold")
        note.append(" for your full history.", style="yellow")
        console.print(note)

    # ── Quota composition (reuses the issue-#4 diagnostic engine). ──
    sections: list = []
    head = Text()
    head.append("Quota composition", style="bold")
    scope = "most-recent " if limit_reached else "last "
    head.append(f"  ·  {diag.sessions} sessions, {diag.turns} turns "
                f"({scope}{since if not limit_reached else f'{max_sessions}'})",
                style="dim")
    sections.append(head)
    sections.append(Text(""))

    # Quota-weighted, not raw tokens (#119): cache reads are discounted well
    # below a base input token in both API pricing and Anthropic's subscription
    # rate-limit weighting, so a raw token share overstates re-reading's actual
    # quota cost. diag.quota_weighted_reread_share applies that discount (and
    # output's premium) — see context_diagnostic.py's CACHE_READ_QUOTA_WEIGHT.
    reread = Text()
    reread.append(f"{_pct(diag.quota_weighted_reread_share)} ", style="bold red")
    reread.append("of your quota went to ", style="")
    reread.append("re-reading context", style="bold")
    reread.append(" (history, CLAUDE.md, tool output)", style="dim")
    sections.append(reread)

    work = Text()
    work.append(f"{_pct(diag.quota_weighted_work_share)} ", style="bold green")
    work.append("went to ", style="")
    work.append("net-new work", style="bold")
    work.append(" (uncached input + output)", style="dim")
    sections.append(work)

    detail = Text()
    detail.append("\nRe-read:   ", style="dim")
    detail.append(f"{format_tokens(diag.total_reread_tokens)} tokens", style="bold")
    detail.append("  (cache reads)", style="dim")
    detail.append("\nNew work:  ", style="dim")
    detail.append(f"{format_tokens(diag.total_work_tokens)} tokens", style="bold")
    sections.append(detail)

    # Aggregate only — never named past sessions (#119). A user has thousands
    # of sessions and never returns to one closed days ago, so a per-session
    # retrospective callout is unactionable noise; the only place a burn signal
    # is actionable is the LIVE session, which the statusline already nudges.
    if diag.compact_candidates:
        sections.append(Text(""))
        candidate_reread = sum(c.reread_tokens for c in diag.compact_candidates)
        share_of_reread = (
            candidate_reread / diag.total_reread_tokens
            if diag.total_reread_tokens else 0.0
        )
        agg = Text()
        agg.append(
            f"{len(diag.compact_candidates)} of your {diag.sessions} sessions",
            style="bold",
        )
        agg.append(" ran context-heavy enough to warrant a mid-session ")
        agg.append("/compact", style="bold")
        agg.append(f": {_pct(share_of_reread)} of this window's re-read tokens.",
                    style="dim")
        sections.append(agg)
        sections.append(Text(
            "The statusline flags this live, before a session ends; "
            "a closed session can't be reclaimed.",
            style="green",
        ))

    console.print(Panel(
        Group(*sections),
        title="[bold]Where your quota goes[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # ── Statusline live preview (self-contained; omits silently if no
    # candidate session ever crosses the nudge threshold). ──
    _render_statusline_preview(timeline, root)

    # ── Who this is for: BOTH telemetry ingest paths, in one line. Claude
    # Code arrives out of band (these session logs, or the statusline / OTel);
    # the MCP server is an SDK/API path, never the Claude Code one. ──
    console.print()
    both = Text()
    both.append("TokenJam ingests both sides of your spend: ", style="dim")
    both.append("Claude Code", style="bold")
    both.append(" from these local session logs, and ", style="dim")
    both.append("Anthropic SDK / API", style="bold")
    both.append(" traffic from OTel spans or the tokenjam MCP server.", style="dim")
    console.print(both)

    # ── Past overspend: the largest single actionable finding. ──
    _render_past_overspend(overspend)

    summary = Text()
    summary.append("Totals: ", style="dim")
    summary.append(f"{timeline.total_sessions} sessions", style="bold")
    summary.append(f" across {timeline.project_count} project"
                   f"{'s' if timeline.project_count != 1 else ''}, ", style="dim")
    summary.append(f"{format_tokens(timeline.total_tokens)} tokens", style="bold")
    if timeline.total_cost_usd > 0:
        summary.append(f"  ·  implied API value {format_cost(timeline.total_cost_usd)}",
                       style="dim")
    console.print(summary)

    # ── Honesty caveat + opt-in "go deeper" pointer. ──
    console.print(f"  [dim]{diag.caveat}[/dim]")
    console.print()
    deeper = Text()
    deeper.append("Go deeper", style="bold")
    deeper.append(": live capture, Lens (the local dashboard), and the "
                  "zero-token statusline. No signup:", style="dim")
    console.print(deeper)
    console.print()
    console.print(Text(f"  {_go_deeper_command()}", style="bold cyan"))
    console.print()


def _render_past_overspend(callout: PastOverspendCallout | None) -> None:
    """The past-overspend callout: what already went to waste, and the fix.

    Silent-degrades (prints nothing) when there is no qualifying finding, so a
    corpus with nothing to report never sees a fabricated or `$0.00` figure.
    """
    if callout is None:
        return

    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    headline = Text()
    if callout.usd is not None:
        headline.append(format_cost(callout.usd), style="bold red")
    else:
        headline.append(f"{format_tokens(callout.tokens or 0)} tokens",
                        style="bold red")
    headline.append(" of that was avoidable", style="")
    if callout.title:
        headline.append(f": {callout.title}", style="bold")
    else:
        headline.append(".", style="")

    body: list = [headline]
    if callout.usd is None and callout.tokens_only_reason:
        body.append(Text(f"No dollar figure: {callout.tokens_only_reason}.",
                         style="dim"))
    if callout.evidence:
        body.append(Text(callout.evidence, style="dim"))

    body.append(Text(""))
    fix = Text()
    fix.append("Fix: ", style="bold green")
    fix.append(callout.advise_text)
    body.append(fix)
    if callout.caveat:
        body.append(Text(""))
        body.append(Text(callout.caveat, style="dim"))

    console.print()
    console.print(Panel(
        Group(*body),
        title="[bold]What that already cost you[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()


def _go_deeper_command() -> str:
    """The "go deeper" footer CTA, context-aware about how quickstart was reached.

    Bare ``npx tokenjam`` / ``uvx --from tokenjam tj`` runs quickstart from a
    throwaway uvx/pipx-run cache (no persistent install), so the CTA is the
    zero-install one (``npx tokenjam onboard``) — the user re-enters through the
    same door. But when quickstart runs from an already-installed ``tj`` binary
    the user obviously has it installed, so drop the ``npx tokenjam`` prefix and
    point straight at ``tj onboard`` (issue #507).
    """
    from tokenjam.cli.cmd_onboard import _is_ephemeral_runner

    return "npx tokenjam onboard" if _is_ephemeral_runner() else "tj onboard"


# ─────────────────────── statusline live preview ───────────────────────────
#
# `tj quickstart` is a read-only, one-shot report; `tj statusline` is the live
# product it upsells (a zero-token Claude Code statusline, updated every turn).
# This section renders — using the SAME `format_status_line` formatter the
# live statusline calls — the line the user's own most-recent substantial
# session would have shown at the exact turn its re-read share crossed the
# nudge threshold. Forward-looking framing only: it previews the LIVE
# experience, it is not advice about the (already-ended) session shown.


def _display_model_name(raw: str | None) -> str:
    """Best-effort human display name for a raw transcript `model` id.

    The live statusline gets a ready-made `display_name` from Claude Code's
    hook payload (see `cmd_statusline._model_name`); this preview only has the
    raw JSONL `message.model` string (e.g. `claude-opus-4-8-20260115`), so it
    reconstructs the same "Family X.Y" shape. Falls back to the raw string for
    shapes it doesn't recognize (e.g. `<synthetic>`).
    """
    if not raw:
        return "?"
    stripped = _re.sub(r"-\d{8}$", "", raw)  # trailing -YYYYMMDD build stamp
    m = _re.match(r"^claude-([a-z]+)-([\d-]+)$", stripped)
    if m:
        family, version = m.groups()
        return f"{family.capitalize()} {version.replace('-', '.')}"
    m = _re.match(r"^claude-([a-z]+)$", stripped)
    if m:
        return m.group(1).capitalize()
    if _re.match(r"^[a-z]+$", stripped):
        return stripped.capitalize()
    return raw


def _transcript_path_for(session_id: str, root: Path) -> str | None:
    """Resolve a session id to its on-disk transcript path under `root`.

    Same glob shape as the live statusline's `find_transcript` fallback, but
    honors quickstart's own `--root` override instead of hardcoding
    `~/.claude/projects` — quickstart already ingested from `root`, so the
    preview must look in the same place.
    """
    pattern = str(root / "**" / f"{session_id}.jsonl")
    hits = _glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


class _PreviewCandidate:
    """One timeline session's preview-selection scoring (see `_select_preview_session`)."""

    __slots__ = ("session", "turns", "crossing")

    def __init__(self, session: TimelineSession, turns: int,
                 crossing: tuple[int, str | None, AssistantUsage] | None) -> None:
        self.session = session
        self.turns = turns
        self.crossing = crossing


def _walk_for_preview(path: str) -> tuple[int, tuple[int, str | None, AssistantUsage] | None]:
    """Walk one transcript once: return `(total_turns, first_threshold_crossing)`.

    `first_threshold_crossing` is `(turn_index, model, cumulative_usage)` for
    the first turn whose cumulative re-read %% reaches the live statusline's
    nudge threshold (`REREAD_WARN`), or None if the session never crosses it.
    Reuses `core.usage.iter_cumulative_usage` — the exact cumulative walk the
    live statusline's own numbers are built from — so this can't show a figure
    the real statusline wouldn't have shown at that point. Never raises: an
    unreadable transcript degrades to "no candidate" (0, None).
    """
    turns = 0
    crossing: tuple[int, str | None, AssistantUsage] | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for turn_index, model, usage in iter_cumulative_usage(fh):
                turns = turn_index
                if crossing is None:
                    total = usage.total
                    reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
                    if reread_pct >= REREAD_WARN:
                        crossing = (turn_index, model, usage)
    except Exception:
        return 0, None
    return turns, crossing


def _select_preview_session(
    timeline: SessionTimeline, root: Path,
) -> _PreviewCandidate | None:
    """Pick the session to preview: the most-recent substantial session that
    crossed the nudge threshold; if none is substantial enough, the largest
    (by turns) that still crossed it; if none ever crossed it, None.
    """
    candidates: list[_PreviewCandidate] = []
    for session in timeline.sessions:  # already most-recent-first
        path = _transcript_path_for(session.session_id, root)
        if not path:
            continue
        turns, crossing = _walk_for_preview(path)
        if crossing is None:
            continue
        candidate = _PreviewCandidate(session, turns, crossing)
        if turns >= PREVIEW_MIN_TURNS:
            # Sessions are walked most-recent-first, so the first substantial
            # crossing candidate IS the winner — stop here rather than
            # re-reading the rest of a possibly-large (up to `--full`) history.
            # The largest-by-turns fallback below only matters when NO
            # candidate is substantial, a case this early return never hides.
            return candidate
        candidates.append(candidate)

    if not candidates:
        return None
    return max(candidates, key=lambda c: c.turns)


def _render_statusline_preview(
    timeline: SessionTimeline, root: Path | None,
) -> _PreviewCandidate | None:
    """"What you'd see live" preview section — self-contained; prints nothing
    when there is no session to preview (no history, no readable transcript,
    or no session ever crossed the nudge threshold).

    Returns the selected ``_PreviewCandidate`` (or ``None``) so a caller can
    tell whether anything was previewed without re-walking transcripts."""
    if root is None or not timeline.sessions:
        return None

    from rich.text import Text

    picked = _select_preview_session(timeline, root)
    if picked is None:
        return None
    assert picked.crossing is not None  # invariant: only crossing candidates are ever selected

    turn_index, model_raw, usage = picked.crossing
    total = usage.total
    reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
    line = format_status_line(_display_model_name(model_raw), total, reread_pct)

    console.print()
    intro = Text()
    intro.append("With the statusline installed, ", style="dim")
    intro.append(f"session {picked.session.session_id[:12]}", style="bold")
    intro.append(f" would have shown this at turn {turn_index}:", style="dim")
    console.print(intro)
    console.print()
    console.print(Text(f"  {line}", style="bold"))
    console.print()
    outro = Text()
    outro.append("That's live, every turn, for zero model tokens. ", style="dim")
    outro.append("tj onboard", style="bold cyan")
    outro.append(" sets it up.", style="dim")
    console.print(outro)
    console.print()
    return picked
