"""
Summarize analyzer — surfaces static prompt files worth summarizing (Track A).

Unlike the other analyzers, this one reasons over the **filesystem**, not
telemetry: it runs the read-only, catalog-default summarize scan
(`core/summarize/candidates.list_candidates`) and reports the prompt-token
reduction available by summarizing those files' prose, priced over the
analyzed window (see below) so `past_overspend_tokens` and
`past_overspend_usd` are on the same basis. It carries the #111
recoverable-savings contract so the Overview waste band and the
`/cost/components` overlay pick it up with no UI change (registry-driven).

Window-guarded: like every other recoverable finding, it contributes nothing on
a dead telemetry window (`ctx.summary.total_tokens == 0`). A window with no calls
has no per-call saving to attach a recoverable figure to, and surfacing one would
break the empty-window overlay invariant (#211) — a dead window must show no
recoverable waste. The filesystem scan is skipped entirely until the window shows
activity.

**The saving RECURS, but only for the part of a file that is actually
resident.** The catalog lumps five different things together and they are not
loaded the same way: `CLAUDE.md` and `.claude/rules/*.md` are re-sent whole at
the head of every session that loads them, whereas a `.claude/skills/*/SKILL.md`,
`.claude/commands/*.md` or `.claude/agents/*.md` surfaces only its frontmatter
that way and delivers its BODY when it is invoked. Pricing every file's whole
body as always-resident made a skill library that had not been invoked once in
the window read as the most expensive prompt file a user owns. So the figure is
the sum of two separately-observed terms:

    always-resident reduction x (sessions that load it) x (reads per session)
  + on-demand reduction       x (times it was actually invoked)

The first term bills as before — first send at the input rate, each later call
in that session at the cache-read rate (0.100x the input rate for every
Anthropic model in `pricing/models.toml`). The second bills at the input rate,
once per invocation. The split itself lives in `core/summarize/load_semantics`
(shared with `core/optimize/write_budget`, which already applied the same rule
when pricing a WRITE); the invocation counts are observed from Claude Code
transcripts by `core/summarize/invocations`. See `_price_reduction`.

That second term is a FLOOR: an invoked body stays in that session's context
for the calls that follow, and those re-reads are not counted here because the
transcript does not say how many followed. Understating is the safe direction
(Critical Rule 22).

This is also the ONE analyzer whose fix has a NEGATIVE standing cost: it
shrinks the always-loaded footprint that the rule-writing analyzers (`relearn`,
`script`, `reuse`, `resend`) grow. `write_budget.measured_agent_file_tokens`
reads this finding as the denominator of their write budget, which is why
`summarize` is deliberately not a `COST_ANALYZERS` member — see the note there.

Honesty discipline (Critical Rule 14 + `core/summarize/estimate.py`): a window
figure — tokens or dollars — is only attached where the load count is
OBSERVED. A global-scope file (`~/.claude/...`) is loaded by every session in
the window, which telemetry counts directly; a project-scope file is loaded
only by its own repo's sessions, matched through the same `agent_id` -> repo
derivation `analyzers/relearn.py` uses. A file whose loading sessions cannot be
identified contributes NEITHER window figure — never a zero, never a rate
borrowed from a file that did resolve (Critical Rule 22) — but its one-time
per-call reduction still surfaces via `file_reduction_tokens` and each
candidate's own `est_tokens_saved`. The same symmetry governs the on-demand
term: an on-demand file whose invocations could not be observed at all (no
transcript corpus) contributes neither figure, while one observed to have been
invoked ZERO times contributes exactly zero — a measurement, not a gap. Every
user-visible string says "estimated" / "review before applying" — never "saves
you"; the mandatory `caveat` names summary's one risk (meaning may change,
structure won't).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.rate_profile import RateProfile, blended_rate_profile
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN
from tokenjam.core.summarize.invocations import InvocationCounts

logger = logging.getLogger(__name__)

#: Prefix Claude Code backfill stamps on a session's ``agent_id``
#: (``core/backfill.py`` ``_agent_id_from_cwd``). Stripping it yields the repo
#: directory name, which is what a project-scope candidate path is matched on.
_CC_AGENT_PREFIX = "claude-code-"

# Surfaced verbatim next to the recoverable figure (contract requires an explicit
# basis). States the basis of BOTH aggregate fields — tokens as well as dollars —
# so no consumer has to guess that they describe the same quantity, and names
# where the one-time per-call reduction lives instead.
SUMMARIZE_ESTIMATE_BASIS = (
    "Read-only filesystem scan of catalog prompt files (CLAUDE.md / AGENTS.md / "
    "rules / skills / commands / agents / globals); prose is summarized, "
    "structure kept verbatim. `past_overspend_tokens` and `past_overspend_usd` "
    "are on the SAME basis — the same event count, one counted and one priced — "
    "and each file is charged by how it is actually LOADED, not as if all of it "
    "were always in context. Always-resident text (a whole CLAUDE.md / rules "
    "file, and the frontmatter of a skill / command / agent, which is what the "
    "harness lists them by) is charged reduction x sessions that load the file x "
    "reads per session, first send at the input rate and each later call in that "
    "session at the cache-read rate. On-demand text (a skill / command / agent "
    "BODY, which reaches the model only when invoked) is charged reduction x "
    "OBSERVED invocations, at the input rate, once each. Session load counts "
    "come from telemetry — a global-scope file against every distinct session in "
    "the window, a project-scope file against its own repo's sessions only. "
    "Invocation counts are observed, never assumed ({invocation_source}); zero "
    "observed invocations is a measurement and prices that body at zero, whereas "
    "an absent transcript corpus carries NEITHER figure for that file. The "
    "on-demand term is a floor: an invoked body stays in that session's context "
    "afterwards and those re-reads are not counted. The two terms are reported "
    "separately per file (`always_resident_tokens_saved` / "
    "`on_demand_tokens_saved`, alongside `load_class` and `invocations`) and "
    "must not be collapsed back into one: on this corpus the always-resident "
    "term dominates, because a small frontmatter re-read on every call of every "
    "session outweighs a large body delivered a handful of times — which is "
    "exactly why charging only the frontmatter, or only the body, would both be "
    "wrong. A file whose loading sessions cannot be identified carries neither "
    "figure here; its one-time per-call reduction still appears in "
    "`file_reduction_tokens` and each candidate's own `est_tokens_saved`. A "
    "file MEASURED to be resident in no session and invoked zero times is not "
    "listed at all, rather than listed at zero. Advisory; review each rewrite "
    "before applying."
)


def _estimate_basis(invocations: "InvocationCounts | None") -> str:
    """The basis string with the invocation evidence actually used spelled out.

    Critical Rule 14: the basis must state the arithmetic truthfully, so it
    names the observed invocation total rather than the mechanism alone.
    """
    from tokenjam.core.summarize.invocations import INVOCATION_SOURCE

    if invocations is None or not invocations.observed:
        source = (
            f"{INVOCATION_SOURCE} — none available in this environment, so no "
            "on-demand file carries a window figure"
        )
    else:
        source = (
            f"{INVOCATION_SOURCE}; {invocations.total_invocations:,} invocation(s) "
            f"observed across {invocations.sessions_scanned:,} transcript(s)"
        )
    return SUMMARIZE_ESTIMATE_BASIS.format(invocation_source=source)

# Mandatory caveat (Rule 14) — carried as the dataclass default like the other
# recoverable findings' caveats (MODEL_DOWNGRADE_CAVEAT etc.) so no surface can
# drop it. Names summary's ONE risk: structure is guaranteed (restore-by-id),
# meaning is not.
SUMMARIZE_HONESTY_CAVEAT = (
    "Structure is guaranteed; meaning may change — review each rewrite before applying."
)


@dataclass
class SummarizeCandidate:
    """One summarizable prompt file (mirrors core/summarize Candidate, trimmed)."""

    path: str
    kind: str          # "prompt" | "other"
    scope: str         # global | project | repo | path
    est_tokens_saved: int
    total_chars: int = 0     # source size (feeds the aggregate reduction %)
    reduction_pct: int = 0   # per-file prose reduction %, computed server-side (no JS chars/4)
    #: How this file reaches the model: ``always`` (whole body re-sent every
    #: session) vs ``skill`` / ``command`` / ``agent`` (frontmatter always,
    #: body only on invocation). See ``core/summarize/load_semantics``.
    load_class: str = "always"
    #: ``est_tokens_saved`` split the same way, so a consumer can see WHICH
    #: half of the file the window figure came from.
    always_resident_tokens_saved: int = 0
    on_demand_tokens_saved: int = 0
    #: Source size of the always-resident portion — the whole file for an
    #: ALWAYS-class one, the measured frontmatter for an on-demand one. This is
    #: what ``core/optimize/write_budget.measured_agent_file_tokens`` sizes the
    #: write budget against, so the read side and the write side charge the
    #: same standing footprint.
    always_resident_chars: int = 0
    #: How many times this file was OBSERVED being invoked in the window.
    #: ``None`` means "not measured" (no transcript corpus) — never 0, which
    #: is the real, priceable answer "it was never invoked".
    invocations: int | None = None
    #: How many of the window's sessions actually load this file, and what the
    #: reduction is worth across them. ``est_usd_saved`` is ``None`` when the
    #: loading sessions could not be identified or no model was priced — a
    #: zero would read as "compressing this file is worth nothing".
    sessions_loading: int = 0
    est_usd_saved: float | None = None
    #: The actual token volume ``est_tokens_saved`` is worth over the analyzed
    #: window: removed on every read (first send + every re-read) across every
    #: loading session — the SAME event count ``est_usd_saved`` prices in
    #: dollars. ``None`` on the same "no loading session observed"
    #: condition ``est_usd_saved`` uses; the one-time per-call reduction stays
    #: available as ``est_tokens_saved``.
    est_tokens_saved_window: int | None = None


@dataclass
class SummarizeFinding:
    """Filesystem-derived summarize opportunity, on the #111 recoverable contract.

    ``past_overspend_tokens`` and ``past_overspend_usd`` are on
    the SAME basis: both price the reduction over the analyzed window
    (removed on every read, across every loading session), so a rollup that
    sums tokens across analyzers and one that sums dollars across analyzers
    describe the same underlying quantity. The one-time, per-call file
    reduction the curate/diff UI cares about lives separately in
    ``file_reduction_tokens`` and each candidate's own ``est_tokens_saved`` —
    it is never the aggregate field's basis.
    """

    candidates: list[SummarizeCandidate] = field(default_factory=list)
    files: int = 0
    past_overspend_usd: float | None = None
    past_overspend_tokens: int | None = None
    #: One-time sum of every candidate's ``est_tokens_saved`` — the aggregate
    #: per-call prose reduction, independent of how many sessions/calls
    #: actually reread it. Feeds the curate/diff UI and ``reduction_pct``;
    #: NOT summed into cross-analyzer token rollups (``estimated_recoverable_
    #: tokens`` is the window-priced figure that belongs there).
    file_reduction_tokens: int | None = None
    estimate_basis: str = ""
    estimate_confidence: str = "heuristic"
    caveat: str = SUMMARIZE_HONESTY_CAVEAT
    # Prose-reduction %s computed server-side (single source of truth — the Lens
    # screen renders these instead of re-deriving chars/CHARS_PER_TOKEN in JS):
    #   reduction_pct     = aggregate saved ÷ source tokens across all candidates
    #   avg_reduction_pct = mean of the per-file reduction %s
    reduction_pct: int | None = None
    avg_reduction_pct: int | None = None
    # The observed inputs behind `past_overspend_usd`, carried so the
    # figure is inspectable rather than a black box: how many sessions the
    # window held, how many calls each made on average (every one of which
    # re-sends these files), and which models the rate blend came from.
    sessions_examined: int = 0
    calls_per_session: float | None = None
    rate_basis: str = ""
    #: Whether the on-demand half of the model had any evidence at all. False
    #: means no Claude Code transcript corpus was readable, so every
    #: skill/command/agent candidate degrades to no window figure rather than
    #: being priced as if it were never invoked.
    invocations_observed: bool = False
    #: Total invocation events observed and how many transcripts were read for
    #: them — the inputs behind the on-demand term, kept inspectable for the
    #: same reason ``sessions_examined``/``calls_per_session`` are.
    invocations_total: int = 0
    transcripts_examined: int = 0


def _src_tokens(total_chars: int) -> int:
    """Source-token estimate for a file's raw size, on the shared chars→tokens
    constant (not a magic /4) so the % matches the rest of the pipeline."""
    return round(total_chars / CHARS_PER_TOKEN)


def _reduction_pct(est_tokens_saved: int, total_chars: int) -> int:
    """Per-file prose reduction % (saved ÷ source tokens), on the shared basis."""
    src = _src_tokens(total_chars)
    return round(est_tokens_saved / src * 100) if src > 0 else 0


@dataclass(frozen=True)
class _LoadProfile:
    """How often the window's sessions would re-send an always-on prompt file.

    ``sessions_total`` is every DISTINCT session in the window (what a
    global-scope file is loaded by) — counted once across the whole window, not
    summed from the per-``agent_id`` groups below, where a session that touched
    two agent_ids appears in both and inflates the total (measured at 22% on a
    real 30-day corpus, applied to every global candidate).
    ``sessions_by_repo``/``calls_by_repo`` narrow that for a project-scope
    file; those per-repo counts stay per-``agent_id`` on purpose — a session
    that touched two repos really does load both repos' `CLAUDE.md`.
    ``calls_per_session`` is the window-WIDE average number
    of LLM calls a session makes — the first sends the file at the input rate,
    each later one re-reads it at the cache-read rate. A project-scope file
    must price against its OWN repo's average, not this blend across every
    other repo/agent in the window (see ``_repo_calls_per_session``); this
    field stays window-wide because it's also what a global-scope file
    legitimately prices against.
    """

    sessions_total: int
    sessions_by_repo: dict[str, int]
    calls_by_repo: dict[str, int]
    calls_per_session: float
    rates: RateProfile


def _load_profile(ctx: AnalyzerContext) -> _LoadProfile | None:
    """Observed session and call counts for the window, per repo.

    ``None`` when the window carries no LLM call or no priced model — the
    finding then reports tokens with no dollars rather than inventing a load
    count. Never raises: a DB hiccup degrades to the tokens-only shape.
    """
    rates = blended_rate_profile(
        ctx.conn, since=ctx.since, until=ctx.until, agent_id=ctx.agent_id,
    )
    if rates is None:
        return None
    clauses = ["name = 'gen_ai.llm.call'", "start_time >= $1", "start_time < $2"]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    where = " AND ".join(clauses)
    try:
        rows = ctx.conn.execute(
            "SELECT agent_id, COUNT(DISTINCT session_id), COUNT(*) "
            "FROM spans WHERE " + where + " GROUP BY agent_id",
            params,
        ).fetchall()
        # Counted separately, NOT summed from `rows`: a session that touched
        # two agent_ids is one session in the window but two rows above.
        totals = ctx.conn.execute(
            "SELECT COUNT(DISTINCT session_id), COUNT(*) FROM spans WHERE " + where,
            params,
        ).fetchone()
    except Exception:
        logger.debug("summarize analyzer: load-profile query failed", exc_info=True)
        return None

    sessions_by_repo: dict[str, int] = {}
    calls_by_repo: dict[str, int] = {}
    sessions_total = int((totals or (0, 0))[0] or 0)
    calls_total = int((totals or (0, 0))[1] or 0)
    # DO NOT "fix" the per-repo counts below to match `sessions_total`. They
    # are deliberately NOT distinct across repos, and that is correct: a
    # session that touched two repos really does load BOTH repos' `CLAUDE.md`,
    # so it belongs to both per-repo counts. Only the WINDOW total had to be
    # counted once — summing these groups to get it was the defect (it made a
    # multi-repo session inflate every global candidate). The two numbers
    # answer different questions and are supposed to disagree.
    for agent_id, sessions, calls in rows:
        sessions = int(sessions or 0)
        calls = int(calls or 0)
        label = str(agent_id or "")
        if label.startswith(_CC_AGENT_PREFIX):
            label = label[len(_CC_AGENT_PREFIX):]
        if label:
            sessions_by_repo[label] = sessions_by_repo.get(label, 0) + sessions
            calls_by_repo[label] = calls_by_repo.get(label, 0) + calls
    if sessions_total <= 0:
        return None
    return _LoadProfile(
        sessions_total=sessions_total,
        sessions_by_repo=sessions_by_repo,
        calls_by_repo=calls_by_repo,
        calls_per_session=calls_total / sessions_total,
        rates=rates,
    )


def _sessions_loading(path: str, scope: str, profile: _LoadProfile) -> int:
    """How many of the window's sessions send this file at the head of every call.

    A global-scope file lives under ``~/.claude`` and is loaded by every
    session. A project/repo/path-scoped one is loaded only by sessions in its
    own repo, matched by walking the file's ancestor directory names against
    the repo labels telemetry recorded. An unmatched path returns 0, which
    leaves the file tokens-only rather than charging it to every session.
    """
    if scope == "global":
        return profile.sessions_total
    ancestors = {parent.name for parent in Path(path).parents if parent.name}
    return sum(
        count for repo, count in profile.sessions_by_repo.items() if repo in ancestors
    )


def _repo_calls_per_session(path: str, scope: str, profile: _LoadProfile) -> float:
    """This candidate's own repo's average calls-per-session, on the SAME
    ancestor-matching basis ``_sessions_loading`` uses.

    A global-scope file legitimately spans every session in the window, so it
    keeps the window-wide average. A project-scope file uses only its own
    repo's observed call rate — blending in every other repo/agent in the
    window would over- or under-state the recoverable figure whenever that
    repo's actual session behavior differs from the window average. Falls
    back to the window-wide average only when the repo match carries no
    session evidence (in practice unreachable from ``_price_reduction``, which
    already returns ``None`` for a zero-session candidate before this is
    consulted).
    """
    if scope == "global":
        return profile.calls_per_session
    ancestors = {parent.name for parent in Path(path).parents if parent.name}
    matched_sessions = sum(
        count for repo, count in profile.sessions_by_repo.items() if repo in ancestors
    )
    if matched_sessions <= 0:
        return profile.calls_per_session
    matched_calls = sum(
        count for repo, count in profile.calls_by_repo.items() if repo in ancestors
    )
    return matched_calls / matched_sessions


def _reads_per_session(calls_per_session: float) -> int:
    """Total times one session reads an always-on file: the first send plus
    every later re-read in that session — at least 1. Shared by
    ``_price_reduction`` and ``_tokens_saved_over_window`` so the dollar and
    token figures price the exact same event count."""
    return max(round(calls_per_session), 1)


def _price_reduction(
    resident_tokens: int,
    on_demand_tokens: int,
    sessions: int,
    calls_per_session: float,
    invocations: int,
    rates: RateProfile,
) -> float | None:
    """What removing this file's prose is worth over the window, by load class.

    The always-resident part (a whole `CLAUDE.md`, or a skill/command/agent's
    frontmatter) is worth the reduction on each loading session, sent once at
    the input rate and re-read on that session's every later call at the
    cache-read rate. The on-demand part (a skill/command/agent BODY) is worth
    the reduction once per OBSERVED invocation, at the input rate — it is not
    in context at all until then.

    ``calls_per_session`` is the candidate's OWN repo's average (see
    ``_repo_calls_per_session``) for a project-scope file, or the window-wide
    average for a global-scope one — never a blend of the two.

    ``None`` when no session loads the file — the saving is real but this
    window carries no evidence of its size, and a zero would misreport that as
    "worth nothing". A file that IS loaded but was never invoked returns a real
    figure (its frontmatter term), which is the honest answer.
    """
    if sessions <= 0:
        return None
    total = 0.0
    if resident_tokens > 0:
        rereads = _reads_per_session(calls_per_session) - 1
        total += rates.cost_of(float(resident_tokens), rereads) * sessions
    if on_demand_tokens > 0 and invocations > 0:
        total += rates.cost_of(float(on_demand_tokens), 0) * invocations
    return total


def _tokens_saved_over_window(
    resident_tokens: int,
    on_demand_tokens: int,
    sessions: int,
    calls_per_session: float,
    invocations: int,
) -> int | None:
    """Actual token volume this file's reduction is worth across the window —
    the EXACT event count ``_price_reduction`` prices in dollars, so the two
    fields stay on the same basis (Critical Rule 28).

    ``None`` on the same "no evidence" condition ``_price_reduction`` returns
    ``None`` for: no session in the window was observed loading this file.
    """
    if sessions <= 0:
        return None
    total = 0
    if resident_tokens > 0:
        total += resident_tokens * _reads_per_session(calls_per_session) * sessions
    if on_demand_tokens > 0 and invocations > 0:
        total += on_demand_tokens * invocations
    return round(total)


def _load_split(candidate: Any) -> tuple[int, int]:
    """A scan candidate's reduction split into (always-resident, on-demand).

    Reads the split ``core/summarize/candidates`` measured on the real text.
    A candidate object that predates those fields — a stub from another caller,
    or a hand-built fixture — carries neither half, and its whole reduction
    falls back to always-resident (the pre-split behaviour) rather than
    silently dropping to zero. The two halves of a real candidate can never
    BOTH be zero while the whole is not: each is floored independently from a
    slice of the same prose, and a candidate needs 100+ prose words to exist.
    """
    total = int(getattr(candidate, "est_tokens_saved", 0) or 0)
    resident = int(getattr(candidate, "always_resident_tokens_saved", 0) or 0)
    on_demand = int(getattr(candidate, "on_demand_tokens_saved", 0) or 0)
    if resident <= 0 and on_demand <= 0:
        return total, 0
    return resident, on_demand


def _is_measured_zero(candidate: SummarizeCandidate) -> bool:
    """True when this file was MEASURED to cost nothing over the window.

    A `.claude/commands/x.md` with no frontmatter that was never invoked is
    resident in no session and delivered on no call: not a candidate at all,
    rather than a candidate worth `$0.00`. Rendering the zero would invite the
    reader to think the analyzer looked for a saving and found none, when the
    truth is there is nothing here to summarize this window (Critical Rule 22 —
    never show a figure the user cannot act on).

    Deliberately keyed on a MEASURED zero, never on a missing one: a candidate
    whose window figure degraded to ``None`` (no loading session observed, or
    no transcript corpus) is kept, because "not measured" is not "worth
    nothing" and suppressing it would hide a file we simply failed to price.
    """
    return (
        candidate.est_tokens_saved_window == 0
        and candidate.est_usd_saved == 0
    )


def _invocation_counts(ctx: AnalyzerContext) -> InvocationCounts:
    """Observed skill/command/agent invocations for the window.

    Never raises: any failure degrades to ``observed=False``, which makes every
    on-demand candidate report no window figure rather than one priced as if
    nothing had ever been invoked.
    """
    from tokenjam.core.summarize.invocations import count_invocations
    from tokenjam.core.transcript_cache import default_cache_dir

    try:
        return count_invocations(
            ctx.since, ctx.until, cache_dir=default_cache_dir(ctx.config),
        )
    except Exception:
        logger.debug("summarize analyzer: invocation scan failed", exc_info=True)
        return InvocationCounts()


@register("summarize")
def run(ctx: AnalyzerContext) -> None:
    """Attach a SummarizeFinding: catalog-default candidates + per-call token saving.

    Reasons over the filesystem (config-driven scan), not `ctx.conn`. The scan is
    catalog-default (a handful of known prompt files) so it's cheap enough for the
    polling Overview; a filesystem hiccup never breaks the optimize report.
    """
    finding = SummarizeFinding(estimate_basis=_estimate_basis(None))

    # Window-guard: a dead telemetry window has no calls to realize a per-call
    # saving against, so — like every recoverable finding — contribute nothing
    # rather than leak a filesystem figure into the empty-window overlay (#211).
    # Also skips the scan entirely on an idle window.
    if ctx.summary.total_tokens == 0:
        ctx.report.findings["summarize"] = finding
        return

    from tokenjam.core.summarize.candidates import list_candidates

    try:
        scan = list_candidates(config=ctx.config)  # read-only, never writes
    except Exception:
        # Empty finding on any scan failure so a filesystem hiccup never breaks the
        # optimize report — but log it: a silent broad-swallow would hide a real
        # code/config regression in list_candidates as if it were a benign hiccup.
        logger.debug(
            "summarize analyzer: candidate scan failed; returning empty finding",
            exc_info=True,
        )
        ctx.report.findings["summarize"] = finding
        return

    # Observed load counts + blended rates for the window. `None` (dead or
    # unpriced window) leaves every candidate tokens-only, exactly as before.
    profile = _load_profile(ctx)
    # Observed invocation counts for the on-demand half of the model. Scanned
    # once for the whole finding, off the same corpus + persistent parse cache
    # `deadweight`/`relearn` already use.
    invocations = _invocation_counts(ctx)
    finding.estimate_basis = _estimate_basis(invocations)
    finding.invocations_observed = invocations.observed
    finding.invocations_total = invocations.total_invocations
    finding.transcripts_examined = invocations.sessions_scanned

    candidates: list[SummarizeCandidate] = []
    for c in scan.candidates:
        if c.est_tokens_saved <= 0:
            continue
        load_class = getattr(c, "load_class", load_semantics.ALWAYS)
        on_demand = load_class in load_semantics.ON_DEMAND_CLASSES
        resident_tokens, on_demand_tokens = _load_split(c)
        sessions = _sessions_loading(c.path, c.scope, profile) if profile else 0
        calls_per_session = (
            _repo_calls_per_session(c.path, c.scope, profile) if profile else 0.0
        )
        # Rule 28 corollary (a): "never invoked" (a measurement) prices the
        # body at zero; "no corpus to look in" is not a measurement, so the
        # whole candidate degrades rather than being quietly priced as zero.
        measured_invocations: int | None = None
        if not on_demand:
            measured_invocations = 0
        elif invocations.observed:
            measured_invocations = invocations.get(
                getattr(c, "invocation_key", "") or "",
            )
        priceable = profile is not None and measured_invocations is not None
        candidate = SummarizeCandidate(
            path=c.path,
            kind="prompt" if c.is_prompt else "other",
            scope=c.scope,
            est_tokens_saved=c.est_tokens_saved,
            total_chars=c.total_chars,
            reduction_pct=_reduction_pct(c.est_tokens_saved, c.total_chars),
            load_class=load_class,
            always_resident_tokens_saved=resident_tokens,
            on_demand_tokens_saved=on_demand_tokens,
            always_resident_chars=int(
                getattr(c, "always_resident_chars", 0) or 0,
            ) or (c.total_chars if not on_demand else 0),
            invocations=measured_invocations if on_demand else None,
            sessions_loading=sessions,
            est_usd_saved=(
                _price_reduction(
                    resident_tokens, on_demand_tokens, sessions,
                    calls_per_session, measured_invocations or 0, profile.rates,
                )
                if priceable and profile is not None else None
            ),
            est_tokens_saved_window=(
                _tokens_saved_over_window(
                    resident_tokens, on_demand_tokens, sessions,
                    calls_per_session, measured_invocations or 0,
                )
                if priceable else None
            ),
        )
        if _is_measured_zero(candidate):
            continue
        candidates.append(candidate)
    finding.candidates = candidates
    finding.files = len(finding.candidates)
    if finding.candidates:
        # One-time aggregate (curate/diff basis) — always available regardless
        # of whether any loading session was observed.
        finding.file_reduction_tokens = sum(
            c.est_tokens_saved for c in finding.candidates
        )
        window_tokens = [
            c.est_tokens_saved_window for c in finding.candidates
            if c.est_tokens_saved_window is not None
        ]
        priced = [c.est_usd_saved for c in finding.candidates if c.est_usd_saved is not None]
        # None, not 0, when nothing resolved: "no session in this window was
        # observed loading these files" is a different statement from
        # "compressing them is worth nothing" (anti-pattern #22). Applies
        # symmetrically to tokens and dollars now that both are window-priced
        # — a candidate contributes to either both sums or neither.
        finding.past_overspend_tokens = sum(window_tokens) if window_tokens else None
        finding.past_overspend_usd = round(sum(priced), 6) if priced else None
        if profile is not None:
            finding.sessions_examined = profile.sessions_total
            finding.calls_per_session = round(profile.calls_per_session, 2)
            finding.rate_basis = profile.rates.basis
        # Prose-reduction %s, computed here so the UI has a single compute path,
        # on the one-time file_reduction_tokens basis (a window-priced numerator
        # here would read as >100% reduction once sessions/calls multiply in):
        #   reduction_pct     = token-weighted aggregate (saved ÷ source tokens)
        #   avg_reduction_pct = mean of the per-file reduction %s
        total_src = sum(_src_tokens(c.total_chars) for c in finding.candidates)
        if total_src > 0:
            finding.reduction_pct = round(
                finding.file_reduction_tokens / total_src * 100
            )
        finding.avg_reduction_pct = round(
            sum(c.reduction_pct for c in finding.candidates) / len(finding.candidates)
        )
    ctx.report.findings["summarize"] = finding
