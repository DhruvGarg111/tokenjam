"""
Context re-send analyzer ("resend"): the product's headline waste category,
previously unmeasured.

This corpus's own benchmark (Princeton HAL, 9 runs, 21,562 calls) found that
**93.8% of prompt tokens sent to real agents were context they already sent**
(benchmarks/RESULTS.md, "2. Repeat-context detection"). No existing analyzer
measures this. `cache_efficacy` computes a caching-ADOPTION rate
(cache_tokens / (input_tokens + cache_tokens)); it reads 0.0 whenever a
scaffold never turned `cache_control` on, even if identical content is
re-sent every single turn. `core/context_diagnostic.py`'s `reread_share` is
adjacent but cache-READ based (a billing signal, nonzero only if caching
happened to be enabled) and is never imported by this package. This analyzer
is the structural gap: it measures repeat context independent of whether
caching was ever turned on, so it flags exactly what `cache_efficacy` misses.

Metric (benchmarks/RESULTS.md:223-231, preserved verbatim; do not invent a
variant):

    prompt_size(turn) = input_tokens + cache_tokens
    repeat_share = 1 - (max(prompt_size) / sum(prompt_size))

aggregated **token-weighted** across sessions:

    repeat_share = 1 - (sum of each session's max / sum of every prompt
                         token across all sessions)

This is an explicitly CONSERVATIVE LOWER BOUND (per the benchmark): if a
session's prompt size only ever grows turn over turn, `sum - max` is exactly
the repeated portion and the bound is tight; a session whose prompt size
sometimes shrinks (e.g. a mid-session `/compact`) only makes this an
UNDERESTIMATE of the true repeat share, never an overestimate.

Honesty discipline (CLAUDE.md Rule 14 / anti-pattern #22): `repeat_share`
itself is a measured token-share, not a savings claim; it is shown
regardless of pricing or caching state.

**Two dollar figures live on this finding, they are NEVER summed, and only
ONE of them may ever be called waste.**

`cost_of_waste_usd` is an OBSERVATION and is COST, not waste: what the
re-sent volume actually cost over the window, priced per token class at the
rates it really billed at (cache reads at the cache-read rate, uncached
repeat at the input rate). Nothing is projected and nothing is discounted,
because nothing is being claimed — this answers "what did re-sending context
cost me", which is a question the data answers exactly.

**It must never be rendered as "wasted" or "overspent".** Waste is only ever
the portion that could have been avoided; unavoidable spend is cost. The
difference between this figure and `estimated_recoverable_usd` is NOT a
measurement of how much re-sending was inherently necessary. Most of it is
simply OUTSIDE the avoidability analysis:

  * sessions handed to `downsize`'s driver-role case (Critical Rule 27) are
    priced here as cost but their avoidable share is reported on that card;
  * sessions below ``MIN_SESSION_CONTEXT_TOKENS`` are dropped from the
    avoidability calculation entirely while still counting as cost;
  * inside the sessions that ARE analysed, only the compaction-bounded
    main-thread re-read tail is claimable, which is much smaller than the
    raw repeat volume this figure prices.

None of those three establish that the excluded money was unavoidable — only
that this analyzer did not analyse it. The coverage fields below
(`cost_in_scope_usd` / `cost_driver_role_usd` / `cost_no_lever_usd`,
`offload_ceiling_usd`, `coverage_note`) exist so a surface can state that
explicitly instead of letting the ratio of the two headline numbers imply a
94%-unavoidable claim the data never made.

`estimated_recoverable_usd` is what the fix actually returns, is derived from
THIS user's corpus rather than a cross-corpus constant, and is THE figure any
surface leading with "waste"/"overspend" must show. The lever is a compound
one — offload context-heavy in-thread work to a subagent AND right-size that
subagent — and both halves are measured here; see `RESEND_ESTIMATE_BASIS`.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tokenjam.core.context_diagnostic import (
    RecurringInclusion,
    TurnComposition,
    compute_context_diagnostic,
    load_turn_compositions,
)
from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
from tokenjam.core.optimize.analyzers.resend_tail import (
    MIN_SESSION_CONTEXT_TOKENS,
    introduced_tokens,
    main_thread_turns,
    premium_driver_role,
    resend_tail_tokens,
    resend_tail_tokens_per_turn,
    session_context_tokens,
)
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.pricing import get_rates

# A window needs at least this many sessions and this many total LLM turns
# before the aggregate repeat-share means anything: a 1-2 session sample is
# noise, not a headline. Named separately from MIN_TURNS below because a
# window can clear one and not the other (e.g. 3 single-turn sessions clear
# the session count but carry zero possible signal).
MIN_SESSIONS_FOR_SIGNAL = 3
MIN_TURNS_FOR_SIGNAL = 6

# Cross-corpus calibration: the only real-world validated "how much of
# cache-blind context actually converts into savings" figure this codebase
# has produced. Measured on this repo's own HAL benchmark corpus (9 Princeton
# HAL runs, 21,562 calls) when prompt caching was added to previously
# cache-blind Anthropic-provider calls: spend fell from $778.16 to $246.57,
# a 68.3% reduction cross-checked against one real ground-truth case (see
# benchmarks/RESULTS.md, "1. Caching recommendations"). That 68.3% is a
# DIFFERENT metric from the 93.8% repeat-share above (dollars vs tokens,
# Anthropic-only vs all-providers); it is not a nested fraction of it.
#
# It is a calibration constant from ANOTHER corpus, so it prices the TOKENS
# claim only — the cache_control / compaction lever, whose mechanism is
# cross-corpus by nature. The DOLLAR claim no longer inherits it: the offload
# lever's avoidable fraction is measured from the user's own `sub_agent_id`
# telemetry instead (see `_measure_offloadable_share`), because that lever's
# realisable share is a property of how THIS user already works, not of a
# benchmark suite.
AVOIDABLE_FRACTION_OF_REPEAT = 0.683

# --- The offload lever, measured -------------------------------------------
# Claude Code's durable fix for repeated context is not caching and not
# `/compact`: it is keeping context-heavy work OFF the long-lived parent
# thread. A subagent's tool outputs live in the subagent's own context, so
# they are never re-billed on any later parent call. The saving is the tail
# that never happens:
#
#     saving_offload  ~ introduced_tokens x tail_calls x cache_read_rate
#     saving_rightsize ~ offloaded_tokens x (premium_rate - right_sized_rate)
#
# and the two COMPOUND — moving the work off the parent thread does not stop
# the subagent that now does it from running on a cheaper model.
#
# Scope, and why it is disjoint from every other analyzer's claim (Critical
# Rule 27 — two analyzers claiming `estimated_recoverable_*` must draw from
# disjoint spans). Two guards, one per neighbour:
#
#   * vs `subagent` — this claim is computed over MAIN-THREAD turns only
#     (`sub_agent_id IS NULL`), so it cannot overlap `subagent`, which filters
#     to `sub_agent_id IS NOT NULL`.
#   * vs `downsize` — `downsize` now claims the same offload mechanism for the
#     sessions where a PREMIUM model drove the work inline (its driver-role
#     case). Those sessions are skipped here. The partition test is the shared
#     `resend_tail.premium_driver_role`, called by both analyzers, so the two
#     populations cannot drift apart the way two independently-tuned thresholds
#     would. `downsize`'s SECONDARY tiny-session case stays disjoint the way it
#     always was: MIN_SESSION_CONTEXT_TOKENS is an order of magnitude above its
#     5K structural ceiling.
#
# The tail arithmetic itself moved to `analyzers/resend_tail.py` so both sides
# compute it from one implementation; the private aliases below keep every
# existing call site and test working unchanged.
_resend_tail_tokens_per_turn = resend_tail_tokens_per_turn
_resend_tail_tokens = resend_tail_tokens
_introduced_tokens = introduced_tokens

RESEND_HONESTY_CAVEAT = (
    "Structural token-share, not a savings claim: a conservative lower bound "
    "(benchmarks/RESULTS.md, HAL corpus: 93.8% of prompt tokens re-sent). "
    "Measured independent of whether caching is enabled: this can read high "
    "even when every re-sent byte was already a cheap cache read. Review "
    "sessions before restructuring."
)

RESEND_ESTIMATE_BASIS = (
    "repeat_tokens = sum(prompt_size) - max(prompt_size) per session "
    "(prompt_size = input_tokens + cache_tokens per turn), aggregated "
    "token-weighted across sessions. TOKENS claim (compaction lever): "
    "repeat_tokens x 68.3% avoidable-fraction (see AVOIDABLE_FRACTION_OF_REPEAT "
    "docstring); cache-agnostic, since compaction cuts gross token volume "
    "regardless of caching state, and cross-corpus calibrated rather than "
    "measured here. USD claim (subagent-offload + right-sizing lever, measured "
    "on YOUR data): for each main-thread turn of a context-heavy session, the "
    "material it introduces (uncached input + output) is re-read by every later "
    "main-thread turn until the next compaction, billed at the cache-read rate "
    "— that tail is what offloading the work to a subagent removes, because a "
    "subagent's tool output never enters the parent context. Only the share of "
    "that volume you demonstrably CAN offload is claimed, and that share is "
    "measured from your own sub_agent_id telemetry (how much of the "
    "context-introducing volume already runs in subagents, in the sessions "
    "where you delegate at all) rather than inherited from a benchmark. "
    "Right-sizing stacks on top: the same offloaded volume priced at the "
    "cheaper same-family model's input rate instead of the premium one. "
    "Computed over main-thread spans only, so it never overlaps the subagent "
    "analyzer's own claim."
)

@dataclass(frozen=True)
class OffloadableShare:
    """``offloadable_share`` plus the provenance a reader needs to judge it.

    The share alone is a bare scalar that reads like a corpus-wide property.
    It is not: it is measured over the delegating minority and applied to the
    non-delegating majority. Carrying the sample size and the spread alongside
    it is what lets every downstream surface disclose that instead of
    re-deriving it (or, as before, silently dropping it).
    """
    share:          float
    sessions:       int            # sessions the share was measured over
    sessions_total: int            # sessions in the window
    median:         float | None   # per-session share, inside the sample
    minimum:        float | None
    maximum:        float | None

    @property
    def sampled_fraction(self) -> float:
        return self.sessions / self.sessions_total if self.sessions_total else 0.0


#: Sentence appended to :data:`RESEND_ESTIMATE_BASIS` disclosing that
#: ``offloadable_share`` is a BEHAVIOURAL sample, how big that sample is, and
#: how dispersed it is. The share is measured only over the sessions that
#: delegate at all and is then applied to sessions that never delegate, so the
#: sample size and spread are load-bearing caveats, not footnotes: a share
#: drawn from 7% of the corpus carries much less weight than one drawn from
#: most of it, and a reader cannot tell which they are looking at unless the
#: number says so itself.
def _offloadable_share_disclosure(measured: OffloadableShare) -> str:
    spread = (
        f" Per-session share inside that sample ranges {measured.minimum:.1%}"
        f"-{measured.maximum:.1%} (median {measured.median:.1%}), so it is a "
        f"central tendency of a dispersed sample, not a constant."
        if measured.median is not None else ""
    )
    return (
        f" BEHAVIOURAL BASIS AND SAMPLE SIZE of the {measured.share:.1%} "
        f"offloadable share: it is measured across the "
        f"{measured.sessions:,} of {measured.sessions_total:,} session(s) "
        f"({measured.sampled_fraction:.1%}) that dispatch a subagent at all, "
        f"then applied unchanged to the sessions that never delegate — which "
        f"are exactly the ones being advised. It therefore describes how much "
        f"work you ALREADY offload when you offload, not how much of this "
        f"window's in-thread work was structurally offloadable; no per-tool-call "
        f"delegability measure is computed anywhere today.{spread}"
    )

RESEND_COST_OF_WASTE_BASIS = (
    "OBSERVED COST, not waste and not recoverable: what re-sent context "
    "actually cost over the window, priced per token class at the rates it "
    "really billed at — cache reads at the cache-read rate, the still-uncached "
    "share of the repeat volume at the input rate. Nothing here is projected or "
    "discounted because nothing is being claimed. Do NOT read this as a saving, "
    "and do NOT read the gap between it and the avoidable figure as a "
    "measurement of what was unavoidable: multi-turn work does inherently "
    "re-send some context, but most of the gap is simply outside the "
    "avoidability analysis (sessions analysed on the model-role card, sessions "
    "below the context floor, and the volume outside the compaction-bounded "
    "main-thread tail). The figure a fix actually returns is "
    "estimated_recoverable_usd, which is smaller and derived separately."
)

COMPACTION_FIX = (
    "Run /compact (or start a fresh session) once accumulated context crosses "
    "your working set. The repeated volume this finding measures is the same "
    "content being re-sent turn over turn: trimming it directly cuts future "
    "prompt size, regardless of whether caching is on. This is a manual, "
    "per-session action, so it never fixes the pattern going forward — treat "
    "it as immediate relief for an already-full session, not the durable fix."
)

# The durable claude-code lever: a rung-1 CLAUDE.md rule (same write machinery
# `script`/`reuse`/`verbosity` use via `cost_proposals._persona_gated_write_fields`)
# so the context that would otherwise get re-sent every turn never accumulates
# on the main thread in the first place. Unlike `/compact`, this persists
# across sessions and is on the CC action surface (a workspace file an
# orchestrating agent reads), which is why it leads for a claude-code window
# instead of `/compact` (founder critique, 2026-07-25: a real CC user abandons
# an over-full session and starts fresh rather than compacting it, so telling
# them to compact isn't a useful recommendation).
SUBAGENT_OFFLOAD_FIX = (
    "Offload context-heavy sub-tasks (broad file reads, multi-file search, "
    "long tool-output loops, exploratory investigation) to a subagent instead "
    "of running them inline in the main thread. A subagent's own tool logs "
    "and intermediate output stay in its own context; only its short "
    "conclusion returns to the caller, so the material that keeps getting "
    "re-sent turn over turn never accumulates on the main thread to begin "
    "with. Where available, pair this with a hook that warns once context "
    "crosses a size threshold, as a second, automated nudge toward the same "
    "behavior."
)

#: The second half of the compound lever. Offloading decides WHERE the work
#: runs; this decides what it runs ON. Both are settable in the same agent
#: file's frontmatter, so the two land as one artifact rather than two cards.
RIGHTSIZE_FIX_TEMPLATE = (
    "Then right-size what you offload to. A subagent doing broad reads and "
    "returning a short conclusion rarely needs the premium tier: pin both its "
    "model and its reasoning effort in its own definition file so every future "
    "dispatch inherits them instead of defaulting to whatever the parent runs "
    "on."
)

# Cap on evidence rows carried in the finding payload; aggregates are over ALL
# sessions with measurable prompt volume, not just the capped examples.
TOP_N_EXAMPLES = 10


@dataclass
class ResendSessionExample:
    """One session's repeat-share breakdown: an evidence row, not the
    aggregate. Ranked by `repeat_tokens` descending (heaviest re-send first).
    """
    session_id: str
    turns: int
    prompt_tokens_sum: int
    prompt_tokens_max: int
    repeat_share: float
    repeat_tokens: int
    provider: str
    model: str


@dataclass
class ResendFinding:
    """Structural context-resend finding. See module docstring for the
    metric and the honesty discipline behind the recoverable estimates."""
    sessions_examined:   int = 0   # all sessions with an LLM turn in window
    multi_turn_sessions: int = 0   # subset with >= 2 turns (can structurally repeat)
    turns_examined:      int = 0
    # The headline: token-weighted aggregate repeat share across every
    # session with measurable prompt volume. None below the data threshold.
    repeat_share:        float | None = None
    repeat_share_median: float | None = None   # per-session median (benchmark parity)
    repeat_share_p90:    float | None = None   # per-session p90 (benchmark parity)
    repeat_tokens:       int = 0    # sum(session sum - session max), the raw resend volume
    prompt_tokens_total: int = 0    # denominator (sum of prompt_size over every turn)
    examples: list[ResendSessionExample] = field(default_factory=list)
    # The "why": recurring inclusions (re-read files, re-run searches,
    # re-pasted prompts/outputs) reused from context_diagnostic rather than
    # reimplemented (capture-gated; empty + a note when no capture toggle is on).
    recurring_examples: list[RecurringInclusion] = field(default_factory=list)
    # All three fixes are always carried: the lever differs by persona
    # (agent harness user: subagent-offload, with compaction as a secondary
    # immediate-relief note; SDK user: cache_control), and the renderer
    # picks which to lead with. `fix_cache_control` is "" when no example
    # session had a model to name in the snippet.
    fix_compaction:        str = COMPACTION_FIX
    fix_subagent_offload:  str = SUBAGENT_OFFLOAD_FIX
    fix_rightsize:         str = RIGHTSIZE_FIX_TEMPLATE
    fix_cache_control:     str = ""
    caveat:            str = RESEND_HONESTY_CAVEAT
    estimate_basis:    str = ""
    estimate_confidence: str = "heuristic"
    estimated_recoverable_tokens: int | None = None
    estimated_recoverable_usd:    float | None = None
    # COST OF WASTE — an observation, never a saving, and NEVER summed with
    # `estimated_recoverable_usd` anywhere. See the module docstring and
    # `RESEND_COST_OF_WASTE_BASIS`. `None` when no turn in the window carried a
    # priced model (a zero would read as "re-sending context is free").
    cost_of_waste_usd:      float | None = None
    cost_of_waste_tokens:   int = 0
    cost_of_waste_basis:    str = RESEND_COST_OF_WASTE_BASIS
    # The two halves of the compound recoverable claim, kept visible so the
    # headline is never a black box. They ARE summed into
    # `estimated_recoverable_usd` — unlike cost-of-waste, these price the same
    # fix and are independent of one another (moving work off the parent thread
    # does not stop it from also running on a cheaper model).
    offload_recoverable_usd:   float | None = None
    rightsize_recoverable_usd: float | None = None
    #: Share of context-introducing volume this user already routes through
    #: subagents, in the sessions where they delegate at all — the measured
    #: replacement for the inherited 68.3% constant on the dollar claim.
    #: `None` when no session in the window delegates, in which case no
    #: offload dollar figure is claimed.
    offloadable_share:         float | None = None
    #: Provenance of the share above: how many sessions it was measured over,
    #: out of how many, and how dispersed it is inside that sample. Disclosed
    #: because the share is a BEHAVIOURAL sample of the delegating minority
    #: applied to the non-delegating majority — see `_offloadable_share_disclosure`.
    offloadable_share_sessions:       int = 0
    offloadable_share_sessions_total: int = 0
    offloadable_share_median:         float | None = None
    # --- Coverage: the two dollar figures' POPULATIONS -----------------------
    # `cost_of_waste_usd` is priced over EVERY session with repeat volume;
    # `estimated_recoverable_usd` only over the sessions that survive the
    # driver-role partition and the context floor. Presenting the two as
    # views of one quantity (and letting their ratio read as "94% of it was
    # unavoidable") is only honest if the differing coverage is stated, so the
    # split is measured here rather than left implicit. The three cost fields
    # partition `cost_of_waste_usd` exactly, up to rounding.
    #: Observed cost inside the sessions the avoidable figure WAS computed over.
    cost_in_scope_usd:     float | None = None
    #: Observed cost in the sessions ceded whole to `downsize`'s driver-role
    #: case. Their avoidable share is reported there, not here — it is analysed
    #: on another card, NOT established as unavoidable.
    cost_driver_role_usd:  float | None = None
    #: Observed cost in sessions with no offload lever at all: below
    #: `MIN_SESSION_CONTEXT_TOKENS` of accumulated main-thread context, or too
    #: short to have a re-read tail. Excluded from the avoidability calc while
    #: still counted as cost.
    cost_no_lever_usd:     float | None = None
    sessions_in_scope:     int = 0
    sessions_no_lever:     int = 0
    #: Ceiling of the offload term inside the in-scope sessions: the full
    #: compaction-bounded main-thread re-read tail priced at a 100% offloadable
    #: share. The gap between this and `cost_in_scope_usd` is the volume the
    #: tail definition excludes; the gap between it and `offload_recoverable_usd`
    #: is the behavioural share discount.
    offload_ceiling_usd:   float | None = None
    #: Plain-language statement of everything above, rendered on the card so a
    #: reader never has to infer coverage from the ratio of two numbers.
    coverage_note:         str = ""
    #: Context-heavy sessions handed to `downsize`'s driver-role case instead
    #: of being claimed here (Critical Rule 27). Surfaced so the partition is
    #: visible on the payload rather than being an invisible subtraction.
    driver_role_sessions:      int = 0
    notes: list[str] = field(default_factory=list)


def _dominant_provider_model(turns: list[TurnComposition]) -> tuple[str, str]:
    """(provider, model) of the most-called pair in a session's turns."""
    counts = Counter((t.provider or "unknown", t.model) for t in turns)
    if not counts:
        return "unknown", ""
    return counts.most_common(1)[0][0]


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0.0-1.0) of a non-empty list. No numpy
    dependency; mirrors cache_efficacy.py's own local helper."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = pct * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _cache_control_snippet(model: str, tokens: int) -> str:
    """The one-paste fix for the SDK-adoption lever, this session's own
    numbers (mirrors cache_efficacy.py's per-agent snippet style)."""
    return (
        f"# {model}: ~{tokens:,} tokens of this session's context are resent "
        "unchanged turn over turn and are not yet benefiting from caching\n"
        + json.dumps({
            "type": "text",
            "text": "<the stable prefix you resend every turn>",
            "cache_control": {"type": "ephemeral"},
        }, indent=2)
    )


def _measure_offloadable_share(
    by_session: dict[str, list[TurnComposition]],
) -> OffloadableShare | None:
    """Share of context-introducing volume this user already routes through
    subagents, measured across the sessions where they delegate at all, WITH
    the provenance that share has to be read with.

    This is the corpus-measured replacement for inheriting a cross-corpus
    constant. Telemetry carries ``sub_agent_id``, so in-thread and offloaded
    work are directly comparable: sessions that already lean on subagents show
    what fraction of the material is offloadable IN PRACTICE for this user's
    kind of work. Sessions that never delegate are excluded from the
    measurement (they have nothing to measure) but are exactly where the
    saving is then claimed.

    That last sentence is the weakness, and it is why this returns an
    :class:`OffloadableShare` rather than a bare float: a scalar generalized
    from the delegating minority onto the non-delegating majority is only
    defensible if every surface carrying it also carries the sample size and
    the spread. A truly STRUCTURAL measure — the share of tail tokens
    introduced by tool calls a subagent could have absorbed — would need
    per-tool-call delegability, which nothing in this codebase computes today;
    until it does, this stays behavioural and says so.

    ``None`` when no session in the window delegates — nothing to measure, so
    nothing is claimed rather than a fraction being invented.
    """
    delegated = 0
    total = 0
    sampled = 0
    per_session: list[float] = []
    for turns in by_session.values():
        if not any(t.sub_agent_id for t in turns):
            continue
        sampled += 1
        s_delegated = 0
        s_total = 0
        for turn in turns:
            introduced = _introduced_tokens(turn)
            s_total += introduced
            if turn.sub_agent_id:
                s_delegated += introduced
        delegated += s_delegated
        total += s_total
        if s_total > 0:
            per_session.append(min(s_delegated / s_total, 1.0))
    if total <= 0 or delegated <= 0:
        return None
    return OffloadableShare(
        share=min(delegated / total, 1.0),
        sessions=sampled,
        sessions_total=len(by_session),
        median=round(statistics.median(per_session), 4) if per_session else None,
        minimum=round(min(per_session), 4) if per_session else None,
        maximum=round(max(per_session), 4) if per_session else None,
    )


#: The three populations a session can fall into once the avoidability calc
#: runs. Named rather than inline so the cost split and the recoverable calc
#: cannot drift apart: both read this ONE classification per session.
COVERAGE_IN_SCOPE = "in_scope"
COVERAGE_DRIVER_ROLE = "driver_role"
COVERAGE_NO_LEVER = "no_lever"


def _coverage_class(
    session_turns: list[TurnComposition], has_share: bool,
) -> str:
    """Which population this session belongs to for the AVOIDABLE figure.

    Exactly the gates the recoverable loop used to apply inline, lifted out so
    the cost figure can be partitioned by the same predicate rather than being
    a single undifferentiated total whose relationship to the avoidable figure
    is unknowable.
    """
    if not has_share:
        return COVERAGE_NO_LEVER
    if premium_driver_role(session_turns) is not None:
        return COVERAGE_DRIVER_ROLE
    main_turns = main_thread_turns(session_turns)
    if len(main_turns) < 2:
        return COVERAGE_NO_LEVER
    if session_context_tokens(main_turns) < MIN_SESSION_CONTEXT_TOKENS:
        return COVERAGE_NO_LEVER
    return COVERAGE_IN_SCOPE


def _coverage_note(finding: ResendFinding) -> str:
    """State, in words, what the avoidable figure does and does not cover.

    The defect this exists to close: a cost figure spanning every session
    shown beside an avoidable figure spanning a filtered subset invites the
    reader to compute a ratio and conclude the remainder was unavoidable. It
    was not — it was analysed elsewhere, filtered out, or outside the tail
    definition. None of those is a finding of necessity, and the card has to
    say so rather than leaving the ratio to speak.
    """
    cost = finding.cost_of_waste_usd
    if cost is None or cost <= 0:
        return ""
    parts = [
        f"COVERAGE. The cost figure covers every session with repeat volume; "
        f"the avoidable figure was computed over {finding.sessions_in_scope:,} "
        f"of them."
    ]
    if finding.driver_role_sessions and finding.cost_driver_role_usd:
        parts.append(
            f"{finding.driver_role_sessions:,} session(s) carrying "
            f"${finding.cost_driver_role_usd:,.2f} of that cost are analysed on "
            f"the model-role card instead (a premium model drove them inline), "
            f"so their avoidable portion is reported there, not counted here."
        )
    if finding.sessions_no_lever and finding.cost_no_lever_usd:
        parts.append(
            f"{finding.sessions_no_lever:,} session(s) carrying "
            f"${finding.cost_no_lever_usd:,.2f} never accumulate the "
            f"{MIN_SESSION_CONTEXT_TOKENS:,} tokens of main-thread context an "
            f"offload lever needs, so they are dropped from the avoidability "
            f"calculation while still counting as cost."
        )
    if finding.cost_in_scope_usd and finding.offload_ceiling_usd is not None:
        parts.append(
            f"Inside the sessions that were analysed, only the "
            f"compaction-bounded main-thread re-read tail is offloadable at "
            f"all — ${finding.offload_ceiling_usd:,.2f} of the "
            f"${finding.cost_in_scope_usd:,.2f} cost there — and only the "
            f"measured share of that tail is priced as avoidable."
        )
    parts.append(
        "The difference between the two figures is therefore NOT a measurement "
        "of what was unavoidable. It is what this analyzer did not analyse."
    )
    return " ".join(parts)


def _capture_flags(config) -> tuple[bool, bool, bool]:
    capture = getattr(config, "capture", None)
    return (
        bool(capture and getattr(capture, "tool_inputs", False)),
        bool(capture and getattr(capture, "prompts", False)),
        bool(capture and getattr(capture, "tool_outputs", False)),
    )


@register("resend")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ResendFinding to ctx.report.findings."""
    finding = ResendFinding()

    # `with_tool_activity=True` is required by the disjointness partition, not
    # by anything this analyzer measures itself: `premium_driver_role` reads
    # `tool_fanout` and `delegates`, which default to inert values without the
    # tool-span join. Loading them here is what makes this analyzer's view of
    # the partition identical to `downsize`'s rather than accidentally wider.
    turns = load_turn_compositions(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id,
        ordered=True, with_tool_activity=True,
    )
    if not turns:
        finding.notes.append("No LLM turns in the window.")
        ctx.report.findings["resend"] = finding
        return

    by_session: dict[str, list[TurnComposition]] = defaultdict(list)
    for t in turns:
        by_session[t.session_id].append(t)

    finding.sessions_examined = len(by_session)
    finding.turns_examined = len(turns)
    finding.multi_turn_sessions = sum(1 for ts in by_session.values() if len(ts) >= 2)

    if len(by_session) < MIN_SESSIONS_FOR_SIGNAL:
        finding.notes.append(
            f"Only {len(by_session)} session(s) in the window (need >= "
            f"{MIN_SESSIONS_FOR_SIGNAL}): too few sessions to measure a "
            "stable repeat-share."
        )
        ctx.report.findings["resend"] = finding
        return
    if len(turns) < MIN_TURNS_FOR_SIGNAL:
        finding.notes.append(
            f"Only {len(turns)} LLM turn(s) in the window (need >= "
            f"{MIN_TURNS_FOR_SIGNAL}): too few turns to measure repeat-share."
        )
        ctx.report.findings["resend"] = finding
        return

    # Measured on this user's own telemetry, not inherited: how much of the
    # context-introducing volume already runs in subagents where they delegate
    # at all. `None` means the corpus can't answer it, so no dollar claim.
    measured_share = _measure_offloadable_share(by_session)
    offloadable_share = measured_share.share if measured_share is not None else None

    total_sum = 0
    total_max = 0
    examples: list[ResendSessionExample] = []
    waste_usd_total = 0.0
    waste_tokens_total = 0
    any_waste_priced = False
    offload_usd_total = 0.0
    offload_ceiling_total = 0.0
    rightsize_usd_total = 0.0
    offload_tokens_total = 0
    # The observed cost, partitioned by the SAME predicate that decides
    # whether a session enters the avoidable figure — so the two numbers'
    # populations are stated rather than silently different.
    cost_by_class: dict[str, float] = {
        COVERAGE_IN_SCOPE: 0.0, COVERAGE_DRIVER_ROLE: 0.0, COVERAGE_NO_LEVER: 0.0,
    }
    sessions_by_class: dict[str, int] = {
        COVERAGE_IN_SCOPE: 0, COVERAGE_DRIVER_ROLE: 0, COVERAGE_NO_LEVER: 0,
    }

    for sid, session_turns in by_session.items():
        prompt_sizes = [t.new_input_tokens + t.reread_tokens for t in session_turns]
        s_sum = sum(prompt_sizes)
        if s_sum <= 0:
            # No measurable prompt volume at all: excluded from the share
            # distribution, same treatment RESULTS.md gives the one
            # zero-volume HAL trajectory in its corpus.
            continue
        s_max = max(prompt_sizes)
        total_sum += s_sum
        total_max += s_max

        repeat_share = 1.0 - (s_max / s_sum)
        repeat_tokens = s_sum - s_max
        provider, model = _dominant_provider_model(session_turns)
        examples.append(ResendSessionExample(
            session_id=sid, turns=len(session_turns),
            prompt_tokens_sum=s_sum, prompt_tokens_max=s_max,
            repeat_share=round(repeat_share, 4), repeat_tokens=repeat_tokens,
            provider=provider, model=model,
        ))

        if repeat_tokens <= 0:
            continue

        # COST OF WASTE (observed), priced per TURN at that turn's OWN model's
        # rate — a session that mixes models (e.g. opus for some turns, haiku
        # for others) must not have every turn priced at whichever model
        # happened to dominate the turn count. `repeat_tokens` is inherently a
        # session-level quantity (it comes from the sum-vs-max prompt-size
        # comparison, not a per-turn one), so its uncached share is allocated
        # across turns in proportion to each turn's own `new_input_tokens` —
        # the same proportion the old single blended fraction expressed in
        # aggregate, just applied per turn instead of once. Every cache read
        # IS re-sent context by definition, billed at that turn's cache-read
        # rate; the still-uncached share billed at that turn's input rate.
        # Nothing discounted, nothing projected — this is what it cost, not
        # what a fix returns. NEVER summed with the recoverable figures below.
        session_waste_usd = 0.0
        for t in session_turns:
            turn_rates = get_rates(t.provider or "unknown", t.model)
            if turn_rates is None or turn_rates.input_per_mtok <= 0:
                continue  # this turn's model unpriced: contributes no dollar figure
            uncached_repeat = repeat_tokens * (t.new_input_tokens / s_sum) if s_sum else 0.0
            session_waste_usd += (
                t.reread_tokens / 1_000_000 * turn_rates.cache_read_per_mtok
                + uncached_repeat / 1_000_000 * turn_rates.input_per_mtok
            )
            waste_tokens_total += t.reread_tokens + round(uncached_repeat)
            any_waste_priced = True
        waste_usd_total += session_waste_usd

        # RECOVERABLE (the compound offload + right-size lever). Main-thread
        # turns only, and only in sessions whose context is heavy enough for
        # the lever to exist — see MIN_SESSION_CONTEXT_TOKENS for why that also
        # keeps this disjoint from `downsize`. Priced per turn at that turn's
        # own model's rate, same reasoning as cost-of-waste above.
        #
        # ONE classification decides both which sessions enter this figure and
        # which bucket their observed cost lands in, so the coverage the card
        # states can never disagree with the coverage the code applied.
        klass = _coverage_class(session_turns, offloadable_share is not None)
        cost_by_class[klass] += session_waste_usd
        sessions_by_class[klass] += 1
        if klass is COVERAGE_DRIVER_ROLE:
            # Claimed in full by `downsize`'s driver-role case (Critical Rule
            # 27). A premium model that drove this session inline is a bigger,
            # differently-framed version of the same offload lever, and the two
            # cards must not price the same tokens. The cost figure above is
            # deliberately NOT skipped: it is an observation of what re-sending
            # context cost, never a saving, and is never summed with any
            # recoverable figure on any surface. That overlap is now stated on
            # the card (`coverage_note`) instead of being an invisible
            # subtraction that makes the remainder look unavoidable.
            continue
        if klass is COVERAGE_NO_LEVER:
            continue
        main_turns = main_thread_turns(session_turns)

        # The tail that offloading removes: material re-read by later
        # main-thread turns purely because the work stayed in the thread.
        if offloadable_share is None:  # pragma: no cover - implied by klass
            continue
        for t, tail_tokens in zip(main_turns, _resend_tail_tokens_per_turn(main_turns)):
            turn_rates = get_rates(t.provider or "unknown", t.model)
            if turn_rates is None or turn_rates.cache_read_per_mtok <= 0:
                continue
            offloadable_tail = tail_tokens * offloadable_share
            offload_usd_total += offloadable_tail / 1_000_000 * turn_rates.cache_read_per_mtok
            # Same tail at a hypothetical 100% share: the ceiling the share
            # discount is applied to, kept so the card can separate "outside
            # the tail definition" from "discounted by the measured share"
            # rather than presenting one opaque gap.
            offload_ceiling_total += tail_tokens / 1_000_000 * turn_rates.cache_read_per_mtok

            # Right-sizing stacks independently: the same offloaded material
            # still has to be read once by whatever runs it, so pricing it at
            # the cheaper same-family model's input rate instead of this
            # turn's is a second, non-overlapping cut. Skipped when no
            # cheaper alternative is priced for this turn's own model.
            offloaded_material = _introduced_tokens(t) * offloadable_share
            offload_tokens_total += round(offloadable_tail + offloaded_material)
            alt = lookup_downgrade(t.provider or "unknown", t.model)
            alt_rates = get_rates(t.provider or "unknown", alt) if alt else None
            if alt_rates is not None:
                rate_gap = max(0.0, turn_rates.input_per_mtok - alt_rates.input_per_mtok)
                rightsize_usd_total += offloaded_material / 1_000_000 * rate_gap

    if total_sum <= 0:
        finding.notes.append(
            "No session in the window carried measurable prompt-token volume."
        )
        ctx.report.findings["resend"] = finding
        return

    finding.prompt_tokens_total = total_sum
    finding.repeat_tokens = total_sum - total_max
    finding.repeat_share = round(1.0 - (total_max / total_sum), 4)

    shares = [e.repeat_share for e in examples]
    finding.repeat_share_median = round(statistics.median(shares), 4)
    finding.repeat_share_p90 = round(_percentile(shares, 0.90), 4)

    examples.sort(key=lambda e: e.repeat_tokens, reverse=True)
    finding.examples = examples[:TOP_N_EXAMPLES]

    finding.estimated_recoverable_tokens = round(
        AVOIDABLE_FRACTION_OF_REPEAT * finding.repeat_tokens
    )
    finding.cost_of_waste_usd = round(waste_usd_total, 6) if any_waste_priced else None
    finding.cost_of_waste_tokens = waste_tokens_total
    finding.offloadable_share = (
        round(offloadable_share, 4) if offloadable_share is not None else None
    )
    if measured_share is not None:
        finding.offloadable_share_sessions = measured_share.sessions
        finding.offloadable_share_sessions_total = measured_share.sessions_total
        finding.offloadable_share_median = measured_share.median
    if any_waste_priced:
        finding.cost_in_scope_usd = round(cost_by_class[COVERAGE_IN_SCOPE], 6)
        finding.cost_driver_role_usd = round(cost_by_class[COVERAGE_DRIVER_ROLE], 6)
        finding.cost_no_lever_usd = round(cost_by_class[COVERAGE_NO_LEVER], 6)
        finding.offload_ceiling_usd = round(offload_ceiling_total, 6)
    finding.sessions_in_scope = sessions_by_class[COVERAGE_IN_SCOPE]
    finding.sessions_no_lever = sessions_by_class[COVERAGE_NO_LEVER]
    driver_role_sessions = sessions_by_class[COVERAGE_DRIVER_ROLE]
    if offloadable_share is not None and offload_tokens_total > 0:
        finding.offload_recoverable_usd = round(offload_usd_total, 6)
        finding.rightsize_recoverable_usd = round(rightsize_usd_total, 6)
        # The two halves compound: offloading decides where the work runs,
        # right-sizing decides what it runs on. Neither cancels the other, so
        # they sum — unlike cost-of-waste, which never enters this figure.
        finding.estimated_recoverable_usd = round(
            offload_usd_total + rightsize_usd_total, 6
        )
    else:
        finding.notes.append(
            "No dollar figure for the offload lever: this window has no "
            "session that both delegates to a subagent (nothing to measure "
            "your offloadable share from) and carries enough main-thread "
            "context for offloading to pay. The token figure above still "
            "stands."
        )
    finding.driver_role_sessions = driver_role_sessions
    if driver_role_sessions:
        finding.notes.append(
            f"{driver_role_sessions} context-heavy session(s) in this window "
            "were driven inline by a premium-tier model that never dispatched "
            "a subagent. Their offload saving is claimed by the model-role "
            "card instead of here, so the two never price the same tokens; "
            "the cost figure above still covers them."
        )
    finding.estimate_basis = RESEND_ESTIMATE_BASIS + (
        _offloadable_share_disclosure(measured_share) if measured_share is not None else ""
    )
    finding.coverage_note = _coverage_note(finding)

    heaviest = finding.examples[0] if finding.examples else None
    if heaviest is not None and heaviest.model and heaviest.repeat_tokens > 0:
        finding.fix_cache_control = _cache_control_snippet(
            heaviest.model, heaviest.repeat_tokens
        )

    tool_inputs_captured, prompts_captured, tool_outputs_captured = _capture_flags(ctx.config)
    if tool_inputs_captured or prompts_captured or tool_outputs_captured:
        diag = compute_context_diagnostic(
            ctx.conn, ctx.since, ctx.until, agent_id=ctx.agent_id,
            tool_inputs_captured=tool_inputs_captured,
            prompts_captured=prompts_captured,
            tool_outputs_captured=tool_outputs_captured,
        )
        finding.recurring_examples = diag.recurring
    else:
        finding.notes.append(
            "Enable `[capture] tool_inputs = true` / `prompts = true` / "
            "`tool_outputs = true` in tj.toml, then `tj backfill claude-code "
            "--reingest`, to see WHICH re-read files, re-run searches, "
            "re-pasted prompts, or re-pasted outputs are driving this number."
        )

    ctx.report.findings["resend"] = finding
