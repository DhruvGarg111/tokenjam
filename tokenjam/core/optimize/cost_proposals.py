"""Adapt cost-analyzer findings into Review-inbox proposals ("advisories").

The self-improve loop's relearn detector already produces
``RelearnCluster`` proposals that the Lens Improve inbox renders and that a
user can mark and apply. The *cost* analyzers (``downsize`` model
over-sizing, ``cache`` efficacy, ``cache-recommend`` breakpoint placement,
``trim`` prompt bloat, ``subagent`` right-sizing, ``deadweight`` dead MCP
servers, ``script`` deterministic workflows, ``reuse`` repeated planning
skeletons, ``verbosity`` high-output outliers) produce findings of a
different shape. This module adapts each finding into a ``CostProposal`` so
the inbox can list them BESIDE the relearn proposals, typed by a distinct
``kind`` field.

Two structural facts carry over from the relearn ``advise_only`` lane and are
NOT optional here:

  * **Advise-only by default, apply-capable where a real workspace surface
    exists.** Most cost fixes live in the user's own application code (a
    model-routing decision, a cache-prefix change, a prompt edit) that
    tokenjam cannot write into — those cards have NO apply path, exactly like
    an ``advise_only`` ``RelearnCluster`` (empty ``suggested_target``). A
    minority (``subagent``, the per-agent slice of ``downsize``, ``script``,
    ``reuse``) DO have a workspace surface an orchestrating agent reads
    before acting (a CLAUDE.md rubric, a model-id key, a new skill note) and
    route through the same rung-gated ``relearn_apply.apply_relearn_fix``
    machinery the relearn lane uses (``apply_capable=True``, ``rung``,
    ``scope``, ``proposed_fix``). ``verbosity`` shares that same class of
    surface in principle but is deliberately kept advise-only for every
    persona (see ``_verbosity_to_proposals``) — the finding is cohort-scoped
    but the artifact would be global, which fails the no-quality-tax gate.
    Every other card carries a recommendation and, where sensible, a
    copyable config/code suggestion; the user applies it themselves.
  * **Estimated, never causal.** Every saving figure a cost finding carries is
    a heuristic ESTIMATE (house style, CLAUDE.md Rule 14). The adapter
    preserves the finding's own ``estimate_basis`` and labels the figure
    ``estimated`` — never proof tokenjam's advice caused a savings change.

The adapter is pure: it reads an already-built ``OptimizeReport`` and returns
proposals. It never touches the DB, the store, or the network.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

# House-style label strings. Kept verbatim on every cost proposal so no channel
# can surface a savings figure without the honesty framing (Rule 14).
COST_ESTIMATE_CONFIDENCE = "estimated"
COST_CORRELATIONAL_CAVEAT = (
    "Estimated, correlational figure; not a causal savings claim. The "
    "recommendation lives in your own application code. Review the evidence "
    "before changing anything."
)

#: The analyzers this wiring covers, by registration name. ``cache-recommend``
#: requires ``[capture] prompts = true``; when that's off the analyzer itself
#: returns an ``enabled=False`` finding with no candidates (same shape as
#: ``trim``'s own disabled-state guard), so including it here unconditionally
#: costs nothing when capture is off.
#:
#: ``resend`` (context re-send, the product's headline waste category — see
#: ``analyzers/context_resend.py``) is included: it has real recoverable
#: savings and previously had no adapter here, silently excluding it from the
#: Review inbox's Cost-advisories tab.
#:
#: ``summarize`` (prompt summarization) IS here (re-filing the
#: purged #326). It used to be deliberately excluded — own review surface, the
#: write-budget's denominator, "don't add cards" — and #326 tried to soften the
#: consequence with a link-only "$X more, not summed here" footnote instead of
#: a card. The founder decision that revisited this rejects the old reasoning
#: outright: **the Review inbox is the complete index of everything
#: actionable, not the list of things whose apply flow happens to live here.**
#: Where the fix is APPLIED is a routing detail, not a reason to keep a real,
#: measured figure off the one surface a user reads for "what's outstanding".
#: So ``summarize`` gets a normal card like every other analyzer here — see
#: ``_summarize_to_proposals`` — except its card routes to the ``tj summarize``
#: curate/diff surface instead of offering an inline apply (``advise_only``,
#: no ``apply_kind``): unlike a model-id swap or an MCP-server removal, the fix
#: is a reviewed rewrite (structure kept, prose compressed), not a value this
#: adapter can safely one-click. The write-budget coupling that motivated the
#: old exclusion is unaffected: ``_apply_write_budget`` reads
#: ``report.findings["summarize"]`` directly (not this proposal list) to size
#: every OTHER analyzer's rule-writing budget, and this card is never
#: ``apply_capable`` so it never enters that netting pass itself.
#: ``relearn`` (recurring agent failures) IS here, for the same founder
#: decision that re-admitted ``summarize``: the Review inbox is the complete
#: index of everything actionable, not the list of things whose apply flow
#: happens to live here. relearn was the last analyzer whose measured cost
#: reached NO aggregate surface at all — it produces ``RelearnCluster``s, not
#: ``CostProposal``s, so ``past_overspend_rollup`` was structurally blind to
#: it and the Dashboard hero's "what waste already cost you" omitted the entire
#: self-improve loop. ``_relearn_to_proposals`` adds exactly one aggregate
#: card, never one per cluster (relearn's own per-cluster rows in the Review
#: inbox are richer and stay the detail view), and it is never
#: ``apply_capable``: the apply path is relearn's own reviewed write flow.
COST_ANALYZERS = (
    "downsize", "cache", "cache-recommend", "trim", "subagent", "deadweight",
    "script", "reuse", "verbosity", "resend", "summarize", "relearn",
)


def _disabled_analyzers(persona: str) -> frozenset[str]:
    """The persona skip-gate set, imported lazily.

    Deferred import: ``runner`` reaches this module through the analyzer
    registry, so a module-level import would cycle.
    """
    from tokenjam.core.optimize.runner import disabled_analyzers_for_persona

    return disabled_analyzers_for_persona(persona)


def cost_analyzers_for_persona(persona: str) -> tuple[str, ...]:
    """``COST_ANALYZERS`` minus the ones with no fix this persona can apply.

    ``COST_ANALYZERS`` is an INDEPENDENT second analyzer-selection surface
    (the Review inbox's recompute selects from it, not from
    ``ANALYZER_ORDER``), so the persona skip gate has to be mirrored here or a
    disabled analyzer's findings still reach the inbox as apply-able cards.
    Same map, same reasons — see ``runner.PERSONA_DISABLED_ANALYZERS``.
    """
    disabled = _disabled_analyzers(persona)
    return tuple(name for name in COST_ANALYZERS if name not in disabled)

# The rung-1/rung-2 apply notes below all route through the SAME workspace-note
# machinery `subagent` already uses (`relearn_apply.apply_relearn_fix`, rung 1
# = CLAUDE.md note, rung 2 = a new .claude/skills/<slug>/SKILL.md). None of
# these three analyzers has a workspace file it can edit outright the way a
# model-routing swap does — the fix is behavioral (an orchestrator or the model
# itself reading guidance), same class of surface as the subagent rubric.

# The rung-2 skill note a `script` proposal writes: the observed tool-call
# pattern is deterministic enough that a script could run it directly instead
# of dispatching a full agent turn.
_SCRIPT_SKILL_INTRO = (
    "This tool-call pattern repeated across many sessions with the same "
    "structural shape (same tools, same argument types, different values). "
    "Consider replacing it with a deterministic script that runs these calls "
    "directly, and reserve the agent turn for the parts that actually need a "
    "model's judgment."
)

# The rung-1 note a `reuse` proposal writes: the planning skeleton recurs.
_REUSE_NOTE_INTRO = (
    "This class of task shares a planning skeleton: the same tool sequence "
    "follows the first planning call, session after session, with only the "
    "argument values differing (dates, versions, paths). Consider templating "
    "the plan for this shape instead of re-planning it from scratch each "
    "time. Review before reusing: a skeleton match is a candidate, not proof "
    "the plan is identical."
)

# The rung-1 sizing-rubric note a CC-origin subagent proposal writes into the
# workspace CLAUDE.md when applied. A shape-based default, not a per-subagent
# edit — it names the observed oversized dispatches and states the routing rule.
SUBAGENT_RUBRIC_INTRO = (
    "Right-size Task-dispatched subagents: default a subagent to the cheapest "
    "same-family model that fits its shape, and only reach for a premium-tier "
    "model (Opus / Fable) when the subtask genuinely needs deep reasoning. A "
    "subagent that does little tool work and returns a short result rarely needs "
    "the premium tier."
)

# The downsize card's claude-code CTA. Mirrors `cmd_optimize._render_downgrade_
# cta`'s claude-code branch: an interactive CC session can't pass `--original`/
# `--candidate` to swap its own model mid-turn, so "route to a cheaper model"
# is not a fix this persona can act on. Used wherever the card would otherwise
# hand a CC window the same generic model-swap text an SDK caller gets (see
# ticket-level "no CC user gets a raw-model-swap CTA" requirement).
_DOWNSIZE_CC_LEVER = (
    "You can't switch your own interactive model mid-session, so this is not a "
    "fix to paste into your own request the way an SDK caller would. The "
    "actionable levers instead: `tj route export --target ccr` (or --target "
    "litellm) to route future calls through a cheaper model, `tj optimize "
    "subagent` to right-size subagent models and context, `/compact` to trim "
    "context mid-session, or a CLAUDE.md/subagent directive telling this agent "
    "to dispatch cheaper subagents for this shape of work."
)

# Appended (not substituted) for a "mixed" window: the swap text above already
# applies to the sdk share, this just adds the claude-code share's own lever —
# same "both, labeled" precedent as `_render_downgrade_cta`'s mixed branch.
_DOWNSIZE_MIXED_CC_NOTE = (
    " For the Claude Code sessions in this window: you can't switch your own "
    "interactive model mid-session — use `tj route export`, `tj optimize "
    "subagent`, or `/compact` instead."
)


@dataclass
class CostProposal:
    """One cost analyzer's finding, shaped for the Review inbox.

    Mirrors the fields the inbox already reads off a ``RelearnCluster`` (title,
    evidence, an estimate with its basis, ``advise_only``) plus a cost-specific
    ``target_key``.
    """
    kind:      str                     # always "cost" — the inbox discriminator
    analyzer:  str                     # "downsize" | "cache" | "trim" | "subagent"
    signature: str                     # stable identity for dedup
    title:     str
    # WHICH thing is flagged, machine-readable (downsize: the oversized
    # model(s); cache: a provider/model; trim: an agent/step).
    target_key: dict[str, Any]
    # Human-readable evidence line: which model/step/cache + the measured
    # baseline number.
    evidence:   str
    # The measured baseline numbers, machine-readable (for rendering + as the
    # verify pass's pre-window reference where useful).
    baseline:   dict[str, Any]
    # Recommendation the user applies themselves + an optional copyable snippet.
    advise_text: str
    suggestion:  str = ""
    # COST OF WASTE — what the flagged behaviour actually COST over the window,
    # fully observed. An ANALYZER-SIDE input, never rendered directly: the
    # stamper below routes it onto `observed_cost_*`. Structurally separate
    # from `past_overspend_*` and MUST NEVER be summed with it, on any surface:
    # the avoidable figure is a SUBSET of this cost, and the difference is
    # spend that was never analysed, not spend shown to be unavoidable (see
    # `analyzers/context_resend.py`'s module docstring).
    cost_of_waste_usd:            float | None = None
    cost_of_waste_tokens:         int | None   = None
    cost_of_waste_basis:          str          = ""
    # PAST OVERSPEND — **THE canonical per-analyzer dollar/token figure.** One
    # field, one meaning, one time basis; there is no second name for it and no
    # paced variant of it anywhere in the tree (see the field contract in the
    # repo `CLAUDE.md`). Every surface — Dashboard hero, Review inbox headline,
    # per-card headline, CLI, MCP — renders THIS field or the rollup of it.
    #
    # **It is the AVOIDABLE amount, observed over the analyzed window, past
    # tense.** Waste is what could have been avoided; unavoidable spend is
    # cost, not waste (the same rule `reuse` already applies by pricing
    # `reps - 1` rather than `reps` — the necessary first instance is not
    # waste). A figure a reader will call "overspend" may therefore only ever
    # be an avoidable figure.
    #
    # Set by the ADAPTER, straight off its analyzer's own
    # `past_overspend_usd`/`_tokens`; `_with_past_overspend` adds only the
    # basis string (and the `observed_cost_*` pair below), so no adapter
    # invents its own tense or wording. ``None`` when the finding produced no
    # priced figure for this item — never coerced to 0.0, which would state
    # "worth nothing" for "not measured" (relearn's aggregate card relies on
    # this: it has no fixed window, so it carries a cost figure only).
    #
    # It deliberately does NOT carry `cost_of_waste_usd`. That figure spans
    # EVERY session, whereas the avoidable figure is computed over a filtered
    # subset (see `resend`'s driver-role partition and context floor). Leading
    # with the big number asserted, implicitly, that the ~94% difference had
    # been shown to be unavoidable. It had not — it was analysed on another
    # card, filtered out before analysis, or outside the tail definition. The
    # cost figure still ships, on `observed_cost_*` below, labelled as cost
    # and never as waste.
    #
    # Per-analyzer derivation, verified against source: downsize
    # (`actual_cost - alt_cost` over the window), deadweight (window tax, "NO
    # projection folded in"), subagent (deltas priced off already-incurred
    # tokens), summarize (per-call reduction over observed calls), reuse (avg
    # cost x (reps - 1)), resend (the measured offload + right-size terms).
    #
    # **NEVER paced.** There is no ratio to apply: pacing left the tree with
    # `estimated_monthly_*` and `compute_projection_ratio`. Multiplying an
    # observation by a forward ratio would make it a forecast — and a paired
    # display whose two sides sat on different time bases would attribute a
    # pacing artifact (measured 1.0714 on the reference corpus) to
    # avoidability.
    past_overspend_usd:           float | None = None
    past_overspend_tokens:        int | None   = None
    past_overspend_basis:         str          = ""
    # OBSERVED COST — what the flagged behaviour cost in total over the same
    # window, including the part that was never shown to be avoidable.
    # Populated ONLY where that total is a genuinely different quantity from
    # the avoidable figure above (today: `resend` alone, from
    # `cost_of_waste_usd`); `None` everywhere else, which is what makes a card
    # render one number instead of saying the same thing twice.
    #
    # Rendered with COST wording only — "re-sending context cost you $X" — and
    # never as waste, overspend, or anything a reader could take as claimable.
    # NEVER summed with `past_overspend_usd` (they overlap: the avoidable
    # figure is a subset of this cost) and never summed into the recoverable
    # rollup. `past_overspend_rollup` reports it as its own separate total.
    observed_cost_usd:            float | None = None
    observed_cost_tokens:         int | None   = None
    observed_cost_basis:          str          = ""
    # Plain-language statement of how the two figures' POPULATIONS differ,
    # supplied by the analyzer. Required whenever `observed_cost_usd` is set:
    # a cost spanning all sessions shown beside an avoidable figure spanning a
    # filtered subset must never be presented as two views of one quantity
    # without the coverage being stated.
    coverage_note:                str          = ""
    estimate_basis:       str = ""
    estimate_confidence:  str = COST_ESTIMATE_CONFIDENCE
    correlational:        bool = True
    # Structural: a cost proposal never has an apply path (see module docstring).
    advise_only:          bool = True
    caveat:               str = COST_CORRELATIONAL_CAVEAT
    # Best-effort service scope for the marker/expectation the user creates on
    # "mark applied" (Expectation.agent_id). "" when the finding spans agents.
    agent_id:             str = ""
    # Workspace-apply plumbing (subagent right-sizing only). Unlike the three
    # advise-only analyzers, a CC-origin subagent finding HAS a writable surface
    # — a rung-1 sizing rubric note in the workspace's CLAUDE.md — so its card
    # can route an actual, reversible, human-gated write through the existing
    # relearn apply path (``relearn_apply.apply_relearn_fix``). The adapter (not
    # the analyzer) supplies these; ``apply_capable`` gates the apply action, and
    # a proposal with no clean workspace surface degrades to advise-only like the
    # other three (``apply_capable=False``, ``advise_only=True``).
    apply_capable:        bool = False
    rung:                 int  = 0
    scope:                str  = ""
    proposed_fix:         str  = ""
    # Model-routing apply kinds (``core.optimize.model_apply``). Set only where
    # the edit is a deterministic rewrite of a value already written down: an
    # agent file's ``model:`` key, or one exact model-id string in a repo the
    # user registered. Empty everywhere else, which leaves the card advise-only
    # with its one-paste artifact.
    apply_kind:           str  = ""
    agent_name:           str  = ""
    current_model:        str  = ""
    proposed_model:       str  = ""
    source_path:          str  = ""
    target_path:          str  = ""
    # Why the direct apply is not on offer, when it is not. Rendered on the card
    # next to the one-paste fix so a fallback is never silent.
    apply_blocked_reason: str  = ""
    # The exact fix, with this agent's own measured values already substituted
    # in. Every advise-only card carries one.
    one_paste_fix:        str  = ""
    # Net-of-standing-cost accounting (`core/optimize/write_budget.py`), filled
    # for every card whose fix is a PERMANENT artifact the user keeps: a rung-1
    # CLAUDE.md rule or a rung-2 skill note. Those are re-sent on every future
    # session, so the four `estimated_*` fields above are reported NET of that
    # standing cost and the pre-net figures are parked here, inspectable. A
    # card with no write to offer (every advise-only cost card) writes nothing,
    # therefore stands nothing, and passes through untouched.
    gross_recoverable_usd:    float | None = None
    gross_recoverable_tokens: int | None   = None
    standing_cost_tokens_per_session: int = 0
    standing_cost_tokens:     int          = 0
    standing_cost_usd:        float | None = None
    standing_cost_basis:      str          = ""
    #: gross / standing. Below 1.0 the rule costs more to keep than it saves.
    payback_ratio:            float | None = None
    net_negative:             bool         = False
    # Whether the permanent write is actually on offer after the budget pass,
    # and why not when it isn't. A suppressed write degrades exactly the way
    # the persona gate already degrades one: advise-only, with the identical
    # text still carried as a copyable `suggestion`.
    write_offered:            bool         = False
    write_blocked_reason:     str          = ""


# --------------------------------------------------------------------------- #
# Per-analyzer adapters. Each reads ONE finding dataclass and returns 0..N
# proposals. All tolerate a None/empty finding (returns []).
# --------------------------------------------------------------------------- #

def _downsize_to_proposal(
    finding: Any, config: Any = None, persona: str = "unknown",
) -> list[CostProposal]:
    """The model-over-sizing card(s).

    When the finding carries per-agent price rows, each agent gets its own card
    with its own arithmetic (and, where the preconditions hold, the gated model-
    id swap). Those rows partition the same candidate sessions, so the window-
    wide card is NOT emitted alongside them: one source of over-sized spend,
    one card.

    Without those rows (no pricing data for a model on either side) the finding
    falls back to the single window-wide card. ``DowngradeFinding.suggestions``
    maps each oversized model to its cheaper same-family alternative, and the
    delta-verify pass measures the model-mix cost delta across ALL flagged
    models, so one proposal listing them keeps that aggregate estimate coherent.

    ``persona`` gates the CTA exactly like ``cmd_optimize._render_downgrade_
    cta`` gates the CLI's: a ``"claude-code"`` window can't switch its own
    interactive model, so it never gets the raw "route to a cheaper model"
    instruction — see ``_DOWNSIZE_CC_LEVER``.
    """
    if finding is None:
        return []
    # The driver-role card is a SEPARATE card from the tiny-session one: it
    # describes a different session population (disjoint by construction — see
    # `analyzers/resend_tail.premium_driver_role`) and a different lever, so
    # merging the two would put one number on top of two unrelated derivations.
    # Exactly one window-wide card, never one per agent, so this adds at most a
    # single row to the inbox.
    proposals = _driver_role_proposals(finding, persona)
    if getattr(finding, "candidate_sessions", 0) <= 0:
        return proposals
    per_agent = _downsize_agent_proposals(finding, config, persona)
    if per_agent:
        return proposals + per_agent
    suggestions: dict[str, str] = dict(getattr(finding, "suggestions", {}) or {})
    if not suggestions:
        return proposals
    models = sorted(suggestions.keys())
    model_list = ", ".join(models)
    evidence = (
        f"{finding.candidate_sessions} of {finding.total_sessions} sessions "
        f"({finding.percent_of_sessions:.0f}%) ran on a larger-than-needed model "
        f"({model_list}); candidate sessions are {finding.percent_of_tokens:.0f}% "
        f"of the window's tokens."
    )
    caveat = str(getattr(finding, "caveat", "") or "")
    suggestion = "\n".join(f"{m} -> {alt}" for m, alt in sorted(suggestions.items()))
    if persona == "claude-code":
        advise = (_DOWNSIZE_CC_LEVER + " " + caveat).strip()
        suggestion = ""
    else:
        advise = (
            "Route the flagged structural-shaped work to the cheaper same-family "
            "model before it runs. Suggested swaps: "
            + "; ".join(f"{m} → {alt}" for m, alt in sorted(suggestions.items()))
            + ". " + caveat
        ).strip()
        if persona == "mixed":
            advise += _DOWNSIZE_MIXED_CC_NOTE
    return [CostProposal(
        kind="cost",
        analyzer="downsize",
        signature="cost:downsize",
        title="Model over-sizing (route to a cheaper same-family model)",
        target_key={"models": models, "suggestions": suggestions},
        evidence=evidence,
        baseline={
            "candidate_sessions": int(finding.candidate_sessions),
            "total_sessions": int(finding.total_sessions),
            "actual_cost_usd": float(finding.actual_cost_usd),
            "alternative_cost_usd": float(finding.alternative_cost_usd),
            "percent_of_tokens": float(finding.percent_of_tokens),
        },
        advise_text=advise,
        suggestion=suggestion,
        one_paste_fix=suggestion,
        # The TINY-SESSION share only: the finding's figures are the sum of
        # both cases, and the driver-role half already has its own card above,
        # so carrying the sum here would claim it twice under two signatures —
        # exactly the failure Critical Rule 27 describes. Subtractive rather
        # than recomputed, so a window with no driver case passes the finding's
        # own numbers through byte for byte (including a `None`).
        past_overspend_usd=_minus_driver_usd(finding),
        past_overspend_tokens=_minus_driver_tokens(finding),
        estimate_basis=_tiny_session_basis(finding),
        # `model_downgrade.py`'s own `monthly_savings_usd` /
        # `monthly_tokens_in_candidates` is a `30/window_days` projection
        # computed inside the analyzer. It is deliberately NOT carried onto
        # this card: a card states one past-tense observation, never a paced
        # one. That field survives only to back the CLI's `tj optimize` line.
    )]


def _minus_driver_usd(finding: Any) -> float | None:
    """The finding's recoverable dollars with the driver-role card's share
    removed, so the two downsize cards partition one figure instead of each
    carrying the total. ``None`` passes straight through."""
    total = getattr(finding, "past_overspend_usd", None)
    if total is None:
        return None
    driver = float(getattr(finding, "driver_recoverable_usd", 0.0) or 0.0)
    return round(max(float(total) - driver, 0.0), 6)


def _minus_driver_tokens(finding: Any) -> int | None:
    """Token counterpart of :func:`_minus_driver_usd`."""
    total = getattr(finding, "past_overspend_tokens", None)
    if total is None:
        return None
    driver = int(getattr(finding, "driver_tokens", 0) or 0)
    return max(int(total) - driver, 0)


def _tiny_session_basis(finding: Any) -> str:
    """The basis for the tiny-session card.

    The finding's own `estimate_basis` describes BOTH cases, which would be
    misleading on a card that only claims one of them — but only once the
    driver-role case actually fired. With no driver session in the window the
    finding's basis already describes exactly this card, so it passes through.
    """
    if int(getattr(finding, "driver_sessions", 0) or 0) <= 0:
        return str(getattr(finding, "estimate_basis", "") or "")
    return (
        "candidate sessions routed to a cheaper model over the window — "
        "structural fit only, no quality validation. The premium-driver-role "
        "share of this window is claimed on its own card, not here."
    )


_DRIVER_ROLE_ADVICE = (
    "Route this shape of work to workers instead of doing it inline. Add a "
    "standing rule to CLAUDE.md telling the agent to dispatch a subagent for "
    "context-heavy sub-tasks (broad file reads, multi-file search, long "
    "tool-output loops, exploratory investigation) rather than running them in "
    "the main thread, and pin the worker's model in its own "
    "`.claude/agents/<name>.md` frontmatter so every dispatch inherits the "
    "cheaper tier. This is not a request to downgrade your own interactive "
    "session: the driver stays on the premium model and keeps making the "
    "decisions. What changes is that the worker's tool output lives in the "
    "worker's context, so it never gets re-read by every later turn of yours."
)


def _driver_role_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """The model-ROLE card: a premium model drove undelegated work inline.

    One window-wide card, deliberately never one per agent — the standing
    don't-fill-the-inbox constraint means this case adds exactly one row no
    matter how many sessions or agents it covers.

    Persona-agnostic advice, unlike the tiny-session card's: the fix here is a
    CLAUDE.md rule plus an agent-file `model:` pin, both of which are on the
    Claude Code action surface, so there is no "you can't switch your own
    interactive model" caveat to apply.
    """
    sessions = int(getattr(finding, "driver_sessions", 0) or 0)
    usd = float(getattr(finding, "driver_recoverable_usd", 0.0) or 0.0)
    if sessions <= 0 or usd <= 0:
        return []
    substitutes: dict[str, str] = dict(getattr(finding, "driver_substitutes", {}) or {})
    models = sorted(substitutes.keys())
    total_sessions = int(getattr(finding, "total_sessions", 0) or 0)
    offload = float(getattr(finding, "driver_offload_usd", 0.0) or 0.0)
    tier = float(getattr(finding, "driver_tier_usd", 0.0) or 0.0)
    tail = int(getattr(finding, "driver_tail_tokens", 0) or 0)
    swap_text = ", ".join(f"{m} → {substitutes[m]}" for m in models)
    evidence = (
        f"{sessions} of {total_sessions} sessions ran "
        f"{', '.join(models) or 'a premium model'} as the driver and never "
        f"dispatched a subagent: {tail:,} tokens were re-read purely because "
        f"that work stayed in the main thread. Routing it to a worker "
        f"({swap_text or 'a cheaper same-family model'}) would have saved "
        f"${offload:,.2f} of re-reads plus ${tier:,.2f} of tier difference."
    )
    one_paste = (
        "# CLAUDE.md\n"
        "Offload context-heavy sub-tasks (broad file reads, multi-file search, "
        "long tool-output loops, exploratory investigation) to a subagent "
        "instead of running them inline in the main thread.\n"
        + "\n".join(
            f"\n# .claude/agents/<name>.md\n---\nmodel: {substitutes[m]}\n---"
            for m in models[:1]
        )
    )
    return [CostProposal(
        kind="cost",
        analyzer="downsize",
        signature="cost:downsize:driver-role",
        title="Premium model in the driver role (route the work, not the thread)",
        target_key={"models": models, "suggestions": substitutes},
        evidence=evidence,
        baseline={
            "driver_sessions": sessions,
            "total_sessions": total_sessions,
            "driver_offload_usd": offload,
            "driver_tier_usd": tier,
            "driver_tail_tokens": tail,
            "driver_tokens": int(getattr(finding, "driver_tokens", 0) or 0),
            "substitutes": substitutes,
        },
        advise_text=_DRIVER_ROLE_ADVICE,
        suggestion=one_paste,
        one_paste_fix=one_paste,
        past_overspend_usd=round(usd, 6),
        past_overspend_tokens=int(getattr(finding, "driver_tokens", 0) or 0),
        estimate_basis=str(getattr(finding, "driver_estimate_basis", "") or ""),
    )]


def _per_agent_cache_recoverable_by_model(finding: Any) -> dict[tuple[str, str], tuple[float, int]]:
    """Sum of ``past_overspend_usd``/``past_overspend_tokens``
    already claimed by the root-caused per-agent cards (A1 uncached / A2
    thrash / A3 lookback), keyed by (provider, model).

    The generic per-(provider, model) efficacy row and these per-agent checks
    both read from the SAME underlying spans — a flagged agent's own calls
    are part of the aggregate the generic row's efficacy is computed over. So
    the dollars a per-agent card claims must be subtracted from the generic
    row's figure before it's surfaced, or the Review-inbox rollup (which sums
    every open card's ``past_overspend_usd`` with no analyzer
    allowlist) double-counts the same waste under two signatures. See
    ``_cache_to_proposals``.
    """
    totals: dict[tuple[str, str], tuple[float, int]] = {}
    groups = (
        getattr(finding, "uncached_agents", []) or [],
        getattr(finding, "thrash_agents", []) or [],
        getattr(finding, "lookback_miss_agents", []) or [],
    )
    for group in groups:
        for c in group:
            usd = getattr(c, "past_overspend_usd", None) or 0.0
            tokens = getattr(c, "past_overspend_tokens", None) or 0
            if usd <= 0 and tokens <= 0:
                continue
            key = (c.provider, c.model)
            prev_usd, prev_tokens = totals.get(key, (0.0, 0))
            totals[key] = (prev_usd + usd, prev_tokens + tokens)
    return totals


def _money(value: float) -> str:
    """A dollar figure with enough precision to stay honest at small values.

    Never rendered bare: every call site pairs it with an estimated/measured tag
    and the construction footnote.
    """
    if abs(value) >= 1.0:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def _agent_arithmetic_line(row: Any) -> str:
    """The card's arithmetic, spelled out with both sides of the comparison."""
    window = (
        f"{row.sessions} session(s) over {row.window_days:.0f} day(s): "
        f"{row.input_tokens:,} input, {row.output_tokens:,} output, "
        f"{row.cache_tokens:,} cache read and {row.cache_write_tokens:,} cache "
        f"write tokens. At {row.model} rates that is "
        f"{_money(row.current_cost_usd)}; the same tokens at {row.alt_model} "
        f"rates are {_money(row.alternative_cost_usd)}. Difference: "
        f"{_money(row.delta_usd)} over the window, up to "
        f"{_money(row.projected_30d_delta_usd)} per 30 days at this rate "
        f"(estimated)."
    )
    if row.thinking_share_of_output is not None:
        window += (
            f" Thinking tokens were {row.thinking_share_of_output * 100:.0f}% of "
            f"this agent's output tokens over the same sessions "
            f"({row.thinking_tokens:,} of {row.output_tokens:,}, measured); "
            f"they bill as output on both models."
        )
    return window


def _model_swap_plumbing(row: Any, config: Any) -> dict[str, Any]:
    """Whether this agent's swap can be written directly, and where.

    The direct write is offered only when the user registered a local source
    path for the agent and every precondition in
    ``model_apply.model_swap_precheck`` holds. Otherwise the card keeps its
    one-paste artifact and states the reason.
    """
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_MODEL_SWAP,
        model_swap_precheck,
    )

    agents = getattr(config, "agents", None) or {}
    agent_cfg = agents.get(row.agent_id) if hasattr(agents, "get") else None
    source_path = str(getattr(agent_cfg, "source_path", "") or "")
    check = model_swap_precheck(source_path, row.model)
    if not check["ok"]:
        return {"apply_capable": False, "apply_blocked_reason": check["reason"]}
    return {
        "apply_capable": True,
        "apply_kind": APPLY_KIND_MODEL_SWAP,
        "source_path": source_path,
        "target_path": check["target_path"],
        "current_model": row.model,
        "proposed_model": row.alt_model,
        "apply_blocked_reason": "",
    }


def _downsize_agent_proposals(
    finding: Any, config: Any, persona: str = "unknown",
) -> list[CostProposal]:
    """One card per agent, carrying that agent's own price arithmetic.

    Replaces the window-wide card when per-agent rows exist, so one source of
    over-sized spend produces exactly one card rather than an aggregate plus its
    own parts.

    When the direct model-id swap is ``apply_capable`` (a registered, git-clean
    source path — see ``_model_swap_plumbing``), the fix is a real write to
    that agent's own config, not "switch your interactive model": it stays
    persona-agnostic. Only the NON-apply-capable fallback text needs the
    claude-code CTA, same reasoning as the window-wide card above.
    """
    proposals: list[CostProposal] = []
    for row in getattr(finding, "per_agent", []) or []:
        if row.delta_usd <= 0:
            # The proposed model is not actually cheaper for this agent's token
            # mix. There is nothing to recover, so there is no card: a rollup
            # that summed this would be summing a loss.
            continue
        plumbing = _model_swap_plumbing(row, config) if config is not None else {
            "apply_capable": False,
            "apply_blocked_reason": (
                "no tj config was available to look up a registered source path."
            ),
        }
        one_paste = (
            f"{row.model} -> {row.alt_model}\n"
            f"# Set this agent's model id to {row.alt_model} where it is "
            f"configured, then redeploy or restart the agent."
        )
        advise = (
            f"Route {row.agent_id}'s flagged structural-shaped work from "
            f"{row.model} to {row.alt_model}. The price difference above is "
            f"arithmetic on this agent's measured tokens, given the switch; "
            f"whether the cheaper model answers as well is not measured here, "
            f"so review the example sessions first."
        )
        if plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can make this exact substitution in "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call. After it is applied you must redeploy "
                f"or restart the agent: measurement starts at the first call "
                f"that runs on {row.alt_model}, not at the moment of the write."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
            if persona == "claude-code":
                advise += " " + _DOWNSIZE_CC_LEVER
            elif persona == "mixed":
                advise += _DOWNSIZE_MIXED_CC_NOTE
        proposals.append(CostProposal(
            kind="cost",
            analyzer="downsize",
            signature=f"cost:downsize:{row.agent_id}",
            title=f"Model over-sizing in {row.agent_id} ({row.model} to {row.alt_model})",
            target_key={
                "agent_id": row.agent_id,
                "models": [row.model],
                "suggestions": {row.model: row.alt_model},
            },
            evidence=_agent_arithmetic_line(row),
            baseline={
                "agent_id": row.agent_id,
                "provider": row.provider,
                "model": row.model,
                "alt_model": row.alt_model,
                "sessions": row.sessions,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "cache_tokens": row.cache_tokens,
                "cache_write_tokens": row.cache_write_tokens,
                "current_cost_usd": row.current_cost_usd,
                "alternative_cost_usd": row.alternative_cost_usd,
                "delta_usd": row.delta_usd,
                "projected_30d_delta_usd": row.projected_30d_delta_usd,
                "thinking_tokens": row.thinking_tokens,
                "thinking_share_of_output": row.thinking_share_of_output,
            },
            advise_text=advise,
            suggestion=one_paste,
            one_paste_fix=one_paste,
            past_overspend_usd=row.delta_usd,
            past_overspend_tokens=row.total_tokens,
            estimate_basis=row.estimate_basis,
            # `row.projected_30d_delta_usd` is this row's own `window_days`-based
            # self-projection and never reaches the card, for the same reason as
            # the window-wide downsize card above: one past-tense figure, never
            # a paced one.
            agent_id=row.agent_id if row.agent_id != "unknown" else "",
            advise_only=not plumbing.get("apply_capable", False),
            apply_capable=bool(plumbing.get("apply_capable")),
            apply_kind=str(plumbing.get("apply_kind", "")),
            source_path=str(plumbing.get("source_path", "")),
            target_path=str(plumbing.get("target_path", "")),
            current_model=str(plumbing.get("current_model", "")),
            proposed_model=str(plumbing.get("proposed_model", "")),
            apply_blocked_reason=str(plumbing.get("apply_blocked_reason", "")),
        ))
    return proposals


def _placement_to_proposals(
    finding: Any, *, pricing_mode: str = "api", persona: str = "unknown",
) -> list[CostProposal]:
    """One card for the batch-placement candidates (advise-only).

    Advise-only is not a formality here: moving a workload to the batch lane is
    an architectural change in the user's own application, and the card says so
    beside the number.

    The Batch API's flat discount is an api-billed price lever — a
    subscription or local plan can't pull it, so ``pricing_mode`` gates the
    dollar figure exactly like the CLI's ``_render_placement`` already does
    (CLAUDE.md anti-pattern #22: never show a figure the reader can't act
    on). Without this the web Review inbox showed a batch-placement dollar
    figure the CLI deliberately suppresses for the same finding.

    ``persona`` gates the WHOLE card, not just the dollar figure: the Batch
    API is a synchronous-endpoint replacement in the user's own APPLICATION
    code, and an interactive Claude Code session has no application code of
    its own to move to a batch lane — the lever is structurally unreachable
    for that persona regardless of ``pricing_mode``. A ``"claude-code"``
    window gets nothing here (not even the advise-only/no-dollar branch);
    ``"mixed"``/``"sdk"``/``"unknown"`` are unaffected — the sdk share of a
    mixed window can still act on it.
    """
    if finding is None or persona == "claude-code":
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    if not candidates:
        return []
    agents = ", ".join(c.agent_id for c in candidates[:5])
    total = float(getattr(finding, "candidate_cost_usd", 0.0) or 0.0)
    saving = float(getattr(finding, "past_overspend_usd", 0.0) or 0.0)
    percent = float(getattr(finding, "percent_of_window_cost", 0.0) or 0.0)
    cadence = ", ".join(
        f"{c.agent_id} every {c.median_gap_seconds / 3600:.1f}h across "
        f"{c.sessions} runs"
        for c in candidates[:5]
    )
    evidence = (
        f"{len(candidates)} workload(s) ran on a regular cadence with no human "
        f"turn after the first model call ({cadence}). They are "
        f"{percent:.0f}% of the window's spend, {_money(total)} (measured)."
    )
    if pricing_mode == "api":
        advise = (
            f"The Batch API bills a flat 50% of standard prices, so the same work "
            f"on the batch lane is {_money(saving)} less over this window "
            f"(estimated). {getattr(finding, 'friction', '')} Nothing here is "
            f"applied for you; the change lives in your own application code."
        )
        recoverable_usd: float | None = saving
    else:
        advise = (
            f"The Batch API's flat discount is an api-billed price lever, so no "
            f"dollar figure is shown for this plan. "
            f"{getattr(finding, 'friction', '')} Nothing here is "
            f"applied for you; the change lives in your own application code."
        )
        recoverable_usd = None
    return [CostProposal(
        kind="cost",
        analyzer="placement",
        signature="cost:placement:batch",
        title="Batch API candidates (unattended, cadence-regular workloads)",
        target_key={"agents": [c.agent_id for c in candidates], "placement": "batch"},
        evidence=evidence,
        baseline={
            "candidates": [
                {
                    "agent_id": c.agent_id, "sessions": c.sessions,
                    "median_gap_seconds": c.median_gap_seconds, "gap_cv": c.gap_cv,
                    "cost_usd": c.cost_usd, "tokens": c.tokens,
                    "estimated_batch_saving_usd": c.estimated_batch_saving_usd,
                }
                for c in candidates
            ],
            "candidate_cost_usd": total,
            "window_cost_usd": float(getattr(finding, "window_cost_usd", 0.0) or 0.0),
            "percent_of_window_cost": percent,
        },
        advise_text=advise,
        suggestion=agents,
        one_paste_fix=(
            "# Submit these workloads through the Batch API instead of the "
            "synchronous endpoint:\n"
            + "\n".join(f"#   {c.agent_id}" for c in candidates)
        ),
        past_overspend_usd=recoverable_usd,
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        agent_id=candidates[0].agent_id if len(candidates) == 1 else "",
    )]


#: Where the summarize card's affordance routes to instead of an inline apply
#: — the ``tj summarize`` curate/diff screen (see ``_summarize_to_proposals``).
SUMMARIZE_REVIEW_HREF = "#/optimize/summarize"


def _summarize_to_proposals(finding: Any) -> list[CostProposal]:
    """One window-wide card for the ``summarize`` (oversized catalog prompt
    files) finding — see ``COST_ANALYZERS``'s docstring for why this is now a
    normal peer card rather than a link-only disclosure.

    One card, never one per file: ``finding.candidates`` can list several
    files, but the standing don't-fill-the-inbox constraint (#596) means this
    adds at most a single row regardless of how many files the scan flags.

    Deliberately never ``apply_capable``: the fix is a reviewed rewrite
    (structure kept, prose compressed, one file at a time) driven by
    ``core/summarize/``'s prepare/check/apply lifecycle, not a value this
    adapter can safely one-click. The card's copy and its UI affordance both
    route to that surface (``SUMMARIZE_REVIEW_HREF``) instead of offering an
    "Apply" button that would misrepresent a multi-step review as one click.
    """
    if finding is None:
        return []
    files = int(getattr(finding, "files", 0) or 0)
    usd = getattr(finding, "past_overspend_usd", None)
    tokens = getattr(finding, "past_overspend_tokens", None)
    if files <= 0 or (usd is None and tokens is None):
        # No candidates, or candidates with no observed load evidence to price
        # against (see `SummarizeFinding`'s "carries neither figure" case) —
        # a card with nothing to lead with would misstate "worth nothing" for
        # "not measured this window".
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    shown = ", ".join(c.path for c in candidates[:5])
    if len(candidates) > 5:
        shown += f", +{len(candidates) - 5} more"
    reduction = getattr(finding, "avg_reduction_pct", None)
    evidence = (
        f"{files} catalog prompt file(s) carry compressible prose"
        + (f" ({reduction}% average reduction)" if reduction is not None else "")
        + f": {shown}."
    )
    plural = "" if files == 1 else "s"
    headline = _money(usd) if usd is not None else f"~{tokens:,} tok"
    advise = (
        f"Review {files} oversized file{plural} in the summarize curate -> "
        "diff -> apply surface (`tj summarize list` / `tj summarize check` / "
        "`tj summarize apply`, or the Summarize screen in the web UI). This "
        "card links there instead of applying inline: the fix is a reviewed "
        "rewrite — structure kept verbatim, prose compressed, one file at a "
        "time — not a one-click removal."
    )
    return [CostProposal(
        kind="cost",
        analyzer="summarize",
        signature="cost:summarize",
        title=f"Review {files} oversized file{plural}, {headline}",
        target_key={"href": SUMMARIZE_REVIEW_HREF, "files": [c.path for c in candidates]},
        evidence=evidence,
        baseline={
            "files": files,
            "file_reduction_tokens": getattr(finding, "file_reduction_tokens", None),
            "sessions_examined": int(getattr(finding, "sessions_examined", 0) or 0),
            "calls_per_session": getattr(finding, "calls_per_session", None),
            "avg_reduction_pct": reduction,
        },
        advise_text=advise,
        past_overspend_usd=usd,
        past_overspend_tokens=tokens,
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        caveat=str(getattr(finding, "caveat", "") or COST_CORRELATIONAL_CAVEAT),
    )]


#: Where a relearn card routes: the Review inbox, which is where relearn's own
#: per-cluster rows and their apply flow already live.
RELEARN_REVIEW_HREF = "#/review"


def _relearn_to_proposals(finding: Any) -> list[CostProposal]:
    """One window-wide card for the ``relearn`` (recurring agent failures)
    finding — see ``COST_ANALYZERS``'s docstring for why relearn is a member.

    ONE card, never one per cluster. relearn already renders every cluster
    individually in the Review inbox, with a richer treatment than a cost card
    could give (apply path, rung badge, example sessions). What it lacked was
    any presence on an AGGREGATE surface: it is a ``RelearnCluster``, not a
    ``CostProposal``, so ``past_overspend_rollup`` could not see a cent of it
    and the Dashboard hero's "what waste already cost you" total silently
    omitted the entire self-improve loop. This adapter closes exactly that gap and nothing
    more.

    ONE number, past tense. ``cost_of_waste_*`` is what the recurrences ALREADY
    COST, summed across EVERY cluster including the ones with no fix template
    and the ones whose rule is uneconomic to keep. Ungated and un-netted on
    purpose: "we have no action for this" is not "this was unavoidable", and a
    fix's future maintenance cost is not subtracted from money already spent.
    ``_with_past_overspend`` routes it onto ``observed_cost_*``.

    ``past_overspend_usd`` — the canonical avoidable field — stays ``None`` on
    this card, deliberately, and must NOT be coerced to 0.0: relearn scans
    unbounded history and has no fixed window to observe an avoidable figure
    over, so the card states "Not priced" rather than "worth nothing". relearn
    prices each occurrence at one re-issued turn TIMES its measured re-read
    tail, and that tail is re-sent context — the exact quantity ``resend``
    already claims in full. Two analyzers claiming the same spans in
    ``past_overspend_rollup`` is CLAUDE.md rule 27, and the
    rollup's signature dedup cannot catch it. The claim is not lost: it lives
    where it always has, on relearn's own per-cluster rows in the Review inbox,
    counted once. ``baseline`` carries the re-read share so the overlap is
    inspectable rather than merely asserted.

    Because ``_with_past_overspend`` stamps ``observed_cost_usd`` from
    ``cost_of_waste_usd`` for any cost-only card, this card is bound by the
    same contract ``coverage_note`` states elsewhere on this module: a cost
    figure with no avoidable figure beside it must say why, not leave the gap
    to imply the spend was unavoidable. ``_relearn_coverage_note`` supplies
    it, naming the rule-27 disjointness against ``resend`` and breaking the
    gated clusters down by why each carries no fix of its own.

    Deliberately never ``apply_capable``: relearn's apply path is its own
    reviewed rung-1/rung-2 write flow (``POST /relearn/apply``), which this
    adapter must not shadow with a second, thinner one — the same routing
    choice ``_summarize_to_proposals`` makes.
    """
    if finding is None:
        return []
    clusters = list(getattr(finding, "clusters", []) or [])
    past_usd = getattr(finding, "past_overspend_usd", None)
    past_tokens = int(getattr(finding, "past_overspend_tokens", 0) or 0)
    if not clusters or (past_usd is None and past_tokens <= 0):
        # Nothing observed, or nothing priceable — a card with no figure to
        # lead with would state "worth nothing" for "not measured".
        return []

    occurrences = sum(int(getattr(c, "occurrences", 0) or 0) for c in clusters)
    window_days = getattr(finding, "window_days", None)
    span = f" over {window_days:.0f} days" if window_days else ""
    shown = ", ".join(str(getattr(c, "title", "") or c.signature) for c in clusters[:3])
    if len(clusters) > 3:
        shown += f", +{len(clusters) - 3} more"
    plural = "" if len(clusters) == 1 else "s"
    evidence = (
        f"{len(clusters)} recurring failure cluster{plural} "
        f"({occurrences} occurrence(s)){span}: {shown}."
    )
    headline = _money(past_usd) if past_usd is not None else f"~{past_tokens:,} tok"
    advise = (
        f"Review the {len(clusters)} recurring failure cluster{plural} in the "
        "Review inbox (or `tj optimize relearn --json`). Each carries its own "
        "example sessions and, where a fix template matched, a reviewable "
        "rung-1/rung-2 write. Clusters with no fix template still appear here: "
        "they cost this money whether or not our library has a remedy for "
        "them yet."
    )
    return [CostProposal(
        kind="cost",
        analyzer="relearn",
        signature="cost:relearn",
        title=f"{len(clusters)} recurring agent failure{plural} already cost {headline}",
        target_key={
            "href": RELEARN_REVIEW_HREF,
            "clusters": [str(getattr(c, "signature", "")) for c in clusters],
        },
        evidence=evidence,
        baseline={
            "clusters": len(clusters),
            "occurrences": occurrences,
            "sessions_scanned": int(getattr(finding, "sessions_scanned", 0) or 0),
            "transcript_sessions_scanned": int(
                getattr(finding, "transcript_sessions_scanned", 0) or 0,
            ),
            "archived_sessions_scanned": int(
                getattr(finding, "archived_sessions_scanned", 0) or 0,
            ),
            "window_days": window_days,
            "corpus_basis": str(getattr(finding, "corpus_basis", "") or ""),
            # Counted, never claimed — see `relearn.BELOW_THRESHOLD_BASIS`.
            "below_threshold_occurrences": int(
                getattr(finding, "below_threshold_occurrences", 0) or 0,
            ),
            "below_threshold_past_overspend_usd": getattr(
                finding, "below_threshold_past_overspend_usd", None,
            ),
            # The re-read SHARE of the figure above (a component, not an
            # addend). Carried so the overlap with `resend`'s re-sent-context
            # figure is inspectable — the reason this card states no forward
            # claim at all (CLAUDE.md rule 27; see the docstring).
            "past_reread_tokens": int(getattr(finding, "past_reread_tokens", 0) or 0),
            "past_reread_usd": getattr(finding, "past_reread_usd", None),
            # relearn's OWN forward claim, kept under its own name on its own
            # finding because it is a genuinely different quantity (fix-gated,
            # over an unbounded corpus). Carried here for reference only: it
            # must never reach `past_overspend_usd`, or it would enter
            # `past_overspend_rollup` beside resend's claim on the same spans.
            "relearn_claim_usd": getattr(finding, "estimated_recoverable_usd", None),
            "relearn_claim_tokens": getattr(finding, "estimated_recoverable_tokens", None),
        },
        advise_text=advise,
        cost_of_waste_usd=past_usd,
        cost_of_waste_tokens=past_tokens or None,
        cost_of_waste_basis=str(getattr(finding, "past_overspend_basis", "") or ""),
        coverage_note=_relearn_coverage_note(clusters),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        caveat=str(getattr(finding, "caveat", "") or COST_CORRELATIONAL_CAVEAT),
    )]


#: Short, reader-facing labels for the ``write_budget`` suppression reasons a
#: gated ``RelearnCluster`` carries, keyed by the exact reason string so a
#: future reason added there degrades to "for another reason" instead of
#: silently going unlabelled.
def _relearn_gate_labels() -> dict[str, str]:
    from tokenjam.core.optimize import write_budget as wb

    return {
        wb.REASON_PLACEHOLDER: "have no derived fix template",
        wb.REASON_NET_NEGATIVE: (
            "are modelled as net-negative to codify (the rule's standing cost "
            "would exceed what it recovers; the value of PREVENTING the "
            "failure is not counted)"
        ),
        wb.REASON_BUDGET_FULL: (
            "are budget-deferred (this window's permanent-rule budget is "
            "already allocated to higher-value fixes)"
        ),
        wb.REASON_FAMILY_MERGED: "are covered by another cluster's shared rule",
        wb.REASON_CEILING_REACHED: "are blocked by the standing-context ceiling",
    }


def _relearn_coverage_note(clusters: list[Any]) -> str:
    """State, in words, why this card carries no avoidable figure and what the
    gated clusters' money is doing instead of leaving the gap unexplained.

    The defect this closes: a cost line with a blank where the avoidable
    figure belongs invites the reader to read the blank as "this was
    unavoidable" — the exact CLAUDE.md rule 27 failure mode already fixed once
    for ``resend`` (see ``_coverage_note`` in ``analyzers/context_resend.py``).
    relearn's case isn't a filtered SUBSET like resend's, though: the forward
    claim is entirely absent from this card by design (see
    ``_relearn_to_proposals``'s docstring) because it would price the same
    re-read tokens ``resend`` already claims in full. Absence of a claim is
    not evidence the spend was unavoidable — it is what this card does not
    attempt to price at all, and the second paragraph names exactly which
    clusters are gated and why, so "no fix yet" is never confused with "no
    cost".
    """
    plural = "" if len(clusters) == 1 else "s"
    parts = [
        f"COVERAGE. The cost figure above covers all {len(clusters)} recurring-"
        f"failure cluster{plural} observed over this window. No avoidable "
        "figure is reported on this card: relearn's own forward claim would "
        "price the same re-read tokens the resend card already claims in "
        "full, and counting them on both cards would double the same spend "
        "(CLAUDE.md rule 27), so this aggregate deliberately carries none."
    ]
    labels = _relearn_gate_labels()
    counts: dict[str, int] = {}
    for c in clusters:
        if getattr(c, "write_offered", True):
            continue
        reason = str(getattr(c, "write_blocked_reason", "") or "")
        label = labels.get(reason, "are gated for another reason")
        counts[label] = counts.get(label, 0) + 1
    if counts:
        gated = sum(counts.values())
        gated_plural = "" if gated == 1 else "s"
        breakdown = "; ".join(f"{n} {label}" for label, n in counts.items())
        parts.append(
            f"Of those, {gated} cluster{gated_plural} carry no permanent fix "
            f"of their own: {breakdown}. Each still cost real money; the gate "
            "is a gap in what we could act on, not a finding that the failure "
            "was harmless."
        )
    parts.append(
        "The absence of an avoidable figure here is not a measurement of what "
        "was unavoidable. It is what this analyzer did not, or could not, "
        "price."
    )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Persona-gated fix TEXT — the `cache` family (`cache` / `cache_thrash` /
# `cache-recommend`) only. Unrelated to `_persona_gated_write_fields` below:
# that helper picks between a workspace WRITE and a snippet fallback for
# script/reuse/verbosity. No cache proposal has a write to offer in the
# first place — every cache fix is an edit to a `cache_control` (or TTL)
# field on the raw Anthropic API request, code a Claude Code session never
# constructs itself (the harness builds that request on the user's behalf).
# There's no workspace surface to fall back to for that persona; the
# instruction itself is the thing they can't act on, so the only honest
# move is to say why instead of stating it as their fix (CLAUDE.md
# anti-pattern #22 — never hand out an instruction the reader can't
# follow). The measured finding underneath (efficacy percentage, token
# counts, evidence) is unaffected — it's true and useful regardless of who
# can act on the fix, so only `advise_text`/`suggestion` are ever gated,
# never `evidence`/`baseline`/the estimate fields.
#
# "unknown" stays on the actionable branch (unlike
# `_persona_gated_write_fields`, which groups "unknown" with "sdk" to keep a
# WRITE off by default): here the safe default is the opposite direction,
# since withholding is the risky move (a real fix denied) rather than
# over-offering one (an unusable write silently succeeding). "mixed" keeps
# the instruction too — the sdk share of a mixed window can act on it, and
# suppressing it would cost that share a real fix for no benefit to the
# claude-code share.
# --------------------------------------------------------------------------- #

CACHE_NO_LEVER_TEXT = (
    "Prompt caching is controlled by cache_control fields on the raw "
    "Anthropic API request. A Claude Code session doesn't construct that "
    "request itself; the harness does. There's no code here for you to "
    "edit, so no fix is shown for it."
)


def _persona_gated_cache_fields(
    persona: str, advise_text: str, suggestion: str = "",
) -> dict[str, Any]:
    """Decide, from the window's dominant persona, whether the cache_control
    instruction (and any snippet) is shown at all. See the module-section
    comment above for the reasoning."""
    if persona == "claude-code":
        return {"advise_text": CACHE_NO_LEVER_TEXT, "suggestion": ""}
    return {"advise_text": advise_text, "suggestion": suggestion}


def _cache_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per flagged (provider, model) cache-efficacy row.

    Reduced by whatever the more specific per-agent root-cause cards (A1/A2/
    A3) already claim for that same (provider, model) — see
    ``_per_agent_cache_recoverable_by_model`` — so the rollup never sums the
    same underlying waste twice under two different signatures.

    The instruction is gated by ``persona`` — see
    ``_persona_gated_cache_fields``; a ``"claude-code"`` window gets the
    honest no-lever reason instead of a cache_control edit it can't make.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        estimate_cache_recoverable,
    )

    already_claimed = _per_agent_cache_recoverable_by_model(finding)

    proposals: list[CostProposal] = []
    for row in getattr(finding, "flagged", []) or []:
        usd, tokens = estimate_cache_recoverable([row])
        claimed_usd, claimed_tokens = already_claimed.get((row.provider, row.model), (0.0, 0))
        basis = str(getattr(finding, "estimate_basis", "") or "")
        if claimed_usd > 0 or claimed_tokens > 0:
            usd = round(max(0.0, (usd or 0.0) - claimed_usd), 6)
            tokens = max(0, (tokens or 0) - claimed_tokens)
            basis = (
                basis + (" " if basis else "")
                + f"Reduced by ${claimed_usd:.4f} already attributed to more "
                "specific per-agent cache proposals for this model, so the "
                "rollup does not double-count the same spend."
            )
        evidence = (
            f"{row.provider}/{row.model}: {row.efficacy * 100:.0f}% of input "
            f"tokens served from cache over {row.input_tokens:,} input tokens "
            f"(caching support: {row.support})."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache:{row.provider}:{row.model}",
            title=f"Low cache efficacy on {row.model}",
            target_key={"provider": row.provider, "model": row.model},
            evidence=evidence,
            baseline={
                "provider": row.provider,
                "model": row.model,
                "input_tokens": int(row.input_tokens),
                "cache_tokens": int(row.cache_tokens),
                "efficacy": float(row.efficacy),
                "efficacy_ceiling": float(getattr(finding, "efficacy_ceiling", 0.80)),
            },
            past_overspend_usd=usd,
            past_overspend_tokens=tokens,
            estimate_basis=basis,
            **_persona_gated_cache_fields(
                persona,
                "Add a stable cache prefix / enable prompt caching for this model "
                "so repeated context is served from cache instead of re-billed as "
                "fresh input.",
            ),
        ))
    return proposals


def _cache_uncached_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A1 uncached-agent candidate (see
    ``analyzers.cache_efficacy``): an agent group making cacheable calls with
    prompt caching never attempted. Scored through the same efficacy metric
    as ``_cache_to_proposals`` (agent-scoped).

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``.
    """
    if finding is None:
        return []
    proposals: list[CostProposal] = []
    for c in getattr(finding, "uncached_agents", []) or []:
        evidence = (
            f"{c.agent_id}: {c.calls} calls on {c.model} with zero prompt "
            f"caching attempted (no cache reads, no cache writes) across "
            f"{c.sessions} session(s); assumed stable prefix "
            f"~{c.assumed_prefix_tokens:,} tokens (this agent's own p25 input size)."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache-uncached:{c.agent_id}",
            title=f"Uncached agent: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "calls": c.calls, "sessions": c.sessions,
                "assumed_prefix_tokens": c.assumed_prefix_tokens,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(
                persona,
                "Add a cache_control breakpoint on this agent's stable prefix "
                "(system prompt / tool definitions) so repeated calls read "
                "from cache instead of paying full input price every time.",
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _cache_thrash_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A2 cache-thrash candidate. Card text branches on the
    detected root cause: a TTL-cadence card (honest break-even, which may say
    the switch isn't worth it) versus an instability checklist card.

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``. The instability checklist is written for
    someone reading their own prompt-assembly code, so it's gated exactly
    like the TTL branch's cache_control edit — neither is something a Claude
    Code session can act on.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        SILENT_INVALIDATOR_CHECKLIST,
    )

    proposals: list[CostProposal] = []
    for c in getattr(finding, "thrash_agents", []) or []:
        evidence = (
            f"{c.agent_id}: caching attempted on {c.model} but read:write "
            f"ratio is {c.read_write_ratio:.2f} over {c.calls} calls "
            f"({c.cache_read_tokens:,} cache-read tokens vs "
            f"{c.cache_write_tokens:,} cache-write tokens); median inter-call "
            f"gap {c.inter_call_gap_p50_minutes:.1f} min."
        )
        if c.cause == "ttl":
            if c.ttl_worth_it:
                advise = (
                    "Calls land more than 5 minutes apart, so the default "
                    "5-minute cache write is expiring before it's reused. "
                    "Switching to the 1-hour cache TTL is estimated to pay "
                    "off at this cadence."
                )
            else:
                advise = (
                    "Calls land more than 5 minutes apart, so the default "
                    "5-minute cache write is expiring before it's reused. "
                    "The 1-hour TTL's write premium doesn't clear at this "
                    "cadence: caching not worth it at this cadence."
                )
        else:
            advise = (
                "Calls land close enough together that a TTL expiry doesn't "
                "explain the miss rate; the prefix itself is likely changing "
                "between calls. " + SILENT_INVALIDATOR_CHECKLIST
            )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache_thrash",
            signature=f"cost:cache-thrash:{c.agent_id}",
            title=f"Cache thrash: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "calls": c.calls, "cache_write_tokens": c.cache_write_tokens,
                "cache_read_tokens": c.cache_read_tokens,
                "read_write_ratio": c.read_write_ratio, "cause": c.cause,
                "inter_call_gap_p50_minutes": c.inter_call_gap_p50_minutes,
                "ttl_worth_it": c.ttl_worth_it,
                "ttl_breakeven_usd": c.ttl_breakeven_usd,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(persona, advise, c.cache_control_snippet),
        ))
    return proposals


def _cache_lookback_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per A3 20-block-lookback-miss candidate. Weakest-
    confidence check of the three; the analyzer only classifies an agent here
    when A1/A2 don't already explain its cache waste.

    Persona-gated the same way as ``_cache_to_proposals`` — see
    ``_persona_gated_cache_fields``.
    """
    if finding is None:
        return []
    from tokenjam.core.optimize.analyzers.cache_efficacy import LOOKBACK_BLOCK_LIMIT

    proposals: list[CostProposal] = []
    for c in getattr(finding, "lookback_miss_agents", []) or []:
        evidence = (
            f"{c.agent_id}: {c.miss_count} cache miss(es) on {c.model}, each "
            f"directly following a turn with an estimated "
            f"{c.avg_prior_turn_blocks:.0f} content blocks (lookback limit: "
            f"{LOOKBACK_BLOCK_LIMIT})."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache",
            signature=f"cost:cache-lookback:{c.agent_id}",
            title=f"20-block lookback miss: {c.agent_id}",
            target_key={"agent_id": c.agent_id, "provider": c.provider, "model": c.model},
            evidence=evidence,
            baseline={
                "agent_id": c.agent_id, "provider": c.provider, "model": c.model,
                "miss_count": c.miss_count,
                "avg_prior_turn_blocks": c.avg_prior_turn_blocks,
            },
            past_overspend_usd=c.past_overspend_usd,
            past_overspend_tokens=c.past_overspend_tokens,
            estimate_basis=c.estimate_basis,
            agent_id=c.agent_id,
            **_persona_gated_cache_fields(
                persona,
                "Anthropic's cache breakpoint search looks back at most "
                f"{LOOKBACK_BLOCK_LIMIT} content blocks. Long tool-heavy "
                "turns push the prior breakpoint out of range; add an "
                "intermediate cache_control breakpoint every ~15 blocks in "
                "long tool-use turns.",
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _cache_recommend_to_proposals(
    finding: Any, cache_finding: Any = None, persona: str = "unknown",
) -> list[CostProposal]:
    """One proposal per Anthropic prefix candidate ``cache-recommend`` flags —
    a placement recommendation for a `cache_control` breakpoint, carrying a
    ready-to-paste snippet (``CachePrefixCandidate.cache_control_snippet``,
    built in the analyzer, modelled on ``cache_efficacy``'s own snippet
    builders).

    Same class of finding as ``_cache_to_proposals``'s A1/A2/A3 root causes —
    an edit to a raw Anthropic API request a Claude Code session never
    constructs itself — so it's gated by the same rule; see
    ``_persona_gated_cache_fields``.

    Reduced by whatever ``cache``'s own per-agent root-cause cards (A1/A2/A3)
    already claim for the same model — the same dedup rule ``_cache_to_
    proposals`` applies against its own generic row, via the same
    ``_per_agent_cache_recoverable_by_model`` helper — so a prefix already
    counted as an uncached-agent / thrash / lookback-miss recovery doesn't
    also get counted here under a third signature. ``cache_finding`` is
    optional (the two analyzers are wired independently and either can be
    absent from a report); with none supplied, nothing is subtracted.
    """
    if finding is None or not getattr(finding, "enabled", False):
        return []
    candidates = list(getattr(finding, "candidates", []) or [])
    if not candidates:
        return []

    already_claimed = (
        _per_agent_cache_recoverable_by_model(cache_finding)
        if cache_finding is not None else {}
    )

    proposals: list[CostProposal] = []
    for c in candidates:
        usd = c.past_overspend_usd
        tokens = c.past_overspend_tokens
        basis = str(getattr(finding, "estimate_basis", "") or "")
        claimed_usd, claimed_tokens = already_claimed.get(("anthropic", c.model), (0.0, 0))
        if claimed_usd > 0 or claimed_tokens > 0:
            usd = round(max(0.0, (usd or 0.0) - claimed_usd), 6)
            tokens = max(0, (tokens or 0) - claimed_tokens)
            basis = (
                basis + (" " if basis else "")
                + f"Reduced by ${claimed_usd:.4f} already attributed to "
                "per-agent cache proposals for this model, so the rollup "
                "does not double-count the same spend."
            )
        preview = c.sample_chars[:60].replace("\n", " ").strip()
        evidence = (
            f'A stable prompt prefix (starting "{preview}...") recurred '
            f"across {c.occurrences} calls on {c.model or 'an unrecorded model'}, "
            f"averaging {c.avg_input_tokens:,.0f} input tokens per call; "
            f"~{c.estimated_cacheable_tokens:,} tokens of that prefix estimated "
            "cacheable."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="cache-recommend",
            signature=f"cost:cache-recommend:{c.prefix_hash}",
            title=f"Repeated prompt prefix on {c.model or 'unrecorded model'}",
            target_key={"prefix_hash": c.prefix_hash, "model": c.model},
            evidence=evidence,
            baseline={
                "prefix_hash": c.prefix_hash,
                "model": c.model,
                "occurrences": c.occurrences,
                "avg_input_tokens": c.avg_input_tokens,
                "estimated_cacheable_tokens": c.estimated_cacheable_tokens,
            },
            past_overspend_usd=usd,
            past_overspend_tokens=tokens,
            estimate_basis=basis,
            **_persona_gated_cache_fields(
                persona,
                "Add a cache_control breakpoint right after this prefix so "
                f"repeat calls on {c.model or 'this model'} read it from cache "
                "instead of paying full input price again.",
                c.cache_control_snippet,
            ),
        ))
    return proposals


def _trim_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per flagged agent/step (grouped from ``per_prompt``)."""
    if finding is None or not getattr(finding, "enabled", False):
        return []
    per_prompt = list(getattr(finding, "per_prompt", []) or [])
    if not per_prompt:
        return []

    # Group per_prompt by agent_id — the flagged "step". Sum bloat across an
    # agent's prompts so one card represents one step.
    by_agent: dict[str, dict[str, Any]] = {}
    for p in per_prompt:
        agent = str(getattr(p, "agent_id", "") or "unknown")
        acc = by_agent.setdefault(agent, {
            "bloat_chars": 0, "prompt_chars": 0, "token_reduction": 0, "prompts": 0,
        })
        acc["bloat_chars"] += int(getattr(p, "bloat_chars", 0) or 0)
        acc["prompt_chars"] += int(getattr(p, "prompt_chars", 0) or 0)
        acc["token_reduction"] += int(getattr(p, "estimated_token_reduction", 0) or 0)
        acc["prompts"] += 1

    # Prorate the finding-level dollar estimate across agents by bloat share, so
    # each card carries a coherent (labeled) slice rather than the whole figure.
    total_bloat = sum(a["bloat_chars"] for a in by_agent.values()) or 1
    finding_usd = getattr(finding, "past_overspend_usd", None)

    proposals: list[CostProposal] = []
    for agent, acc in sorted(by_agent.items()):
        if acc["bloat_chars"] <= 0:
            continue
        share = acc["bloat_chars"] / total_bloat
        usd = round(finding_usd * share, 6) if finding_usd is not None else None
        evidence = (
            f"{agent}: {acc['bloat_chars']:,} low-significance characters across "
            f"{acc['prompts']} prompt(s) (~{acc['token_reduction']:,} trimmable "
            f"input tokens)."
        )
        proposals.append(CostProposal(
            kind="cost",
            analyzer="trim",
            signature=f"cost:trim:{agent}",
            title=f"Prompt bloat in {agent}",
            target_key={"agent_id": agent},
            evidence=evidence,
            baseline={
                "agent_id": agent,
                "bloat_chars": acc["bloat_chars"],
                "prompt_chars": acc["prompt_chars"],
                "estimated_token_reduction": acc["token_reduction"],
            },
            advise_text=(
                "Trim the low-significance regions from this step's prompt "
                "template (boilerplate, repeated instructions, dead context) so "
                "every call carries fewer input tokens."
            ),
            past_overspend_usd=usd,
            past_overspend_tokens=acc["token_reduction"] or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            agent_id=agent if agent != "unknown" else "",
        ))
    return proposals


#: A ``sub_agent_id`` only names an agent definition when it is a plain slug.
#: Claude Code stamps a UUID for inline Task dispatches, and there is no file to
#: edit for those: that is the guidance-block fallback case, not a lookup to
#: guess at.
_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

#: Cap on transcripts read to locate the repos a finding's sessions ran in.
_MAX_SCOPE_SESSIONS = 20


def _session_cwds(session_ids: list[str], config: Any) -> dict[str, str]:
    """``session_id -> repo cwd``, read from the sessions' own transcripts.

    Reuses the relearn detector's resolver rather than re-deriving a cwd from
    the encoded project directory name, which is unreliable. Best-effort: an
    unreadable transcript simply contributes no cwd.
    """
    from tokenjam.core.optimize.analyzers.relearn import _repo_cwd_map_for
    from tokenjam.core.transcript import resolve_projects_root

    override = getattr(getattr(config, "loop", None), "transcript_path", None)
    root = resolve_projects_root(override)
    pairs = [(sid, sid) for sid in session_ids[:_MAX_SCOPE_SESSIONS]]
    return _repo_cwd_map_for(pairs, root)


def _agent_model_plumbing(over_powered: list[Any], config: Any) -> dict[str, Any]:
    """Whether a flagged subagent has a definition file whose model can be set.

    Scope routing is relearn's: sessions concentrated in one repo write into
    that repo's ``.claude/agents/``, sessions spanning repos write into the
    user-global one. The flagged rows are cost-ordered, so the first subagent
    with a real definition file is the most expensive one that can be fixed
    outright. No file means the guidance block stays the fix, which is the
    inline Task-tool case.
    """
    from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
    from tokenjam.core.optimize.analyzers.relearn import _scope_for
    from tokenjam.core.optimize.model_apply import (
        APPLY_KIND_AGENT_MODEL,
        default_agent_file_path,
    )

    named = [
        r for r in over_powered
        if _AGENT_NAME_RE.match(str(getattr(r, "sub_agent_id", "") or ""))
    ]
    if not named or config is None:
        return {}
    cwds = _session_cwds([str(r.session_id) for r in over_powered], config)
    repos = {Path(cwd).name for cwd in cwds.values() if cwd}
    scope = _scope_for(repos)
    repo_cwd = next(iter(cwds.values()), "") if len(repos) == 1 else ""

    for row in named:
        proposed = lookup_downgrade(str(row.provider), str(row.model))
        if not proposed:
            continue
        name = str(row.sub_agent_id)
        path = default_agent_file_path(scope, repo_cwd, name)
        if not path or not Path(path).is_file():
            continue
        return {
            "apply_kind": APPLY_KIND_AGENT_MODEL,
            "agent_name": name,
            "target_path": path,
            "scope": scope,
            "current_model": str(row.model),
            "proposed_model": proposed,
        }
    return {}


def _subagent_to_proposals(finding: Any, config: Any = None) -> list[CostProposal]:
    """One proposal covering the subagent right-sizing finding.

    Unlike the three advise-only analyzers, this one is workspace-appliable for
    the common (CC-origin) case: the fan-out model choice is made by the
    orchestrating agent, which reads the workspace's CLAUDE.md — so a rung-1
    sizing rubric note IS a legitimate, reversible workspace fix. The subagent
    analyzer runs only over Claude Code data (``sub_agent_id`` is populated by
    the CC backfill; other runtimes carry NULL and are ignored), so a finding
    here is CC-origin, hence ``apply_capable``. If no oversized model is priced
    (nothing to key a delta on), the proposal degrades to advise-only.

    The delta-verify pass measures the fan-out model-mix cost delta across the
    over-powered models, so a single proposal listing them keeps the finding-
    level estimate coherent (mirrors the downsize adapter).
    """
    if finding is None:
        return []
    flagged = list(getattr(finding, "flagged", []) or [])
    over_powered = [r for r in flagged if "over_powered" in (getattr(r, "flags", []) or [])]
    if not over_powered:
        return []

    models = sorted({str(r.model) for r in over_powered})
    subagents = len({(r.session_id, r.sub_agent_id) for r in over_powered})
    pct = float(getattr(finding, "percent_of_cost", 0.0) or 0.0) * 100
    model_list = ", ".join(models)
    evidence = (
        f"{subagents} subagent dispatch(es) ran on a premium-tier model "
        f"({model_list}) but did little work (small output, few tool calls). "
        f"Subagents are {pct:.0f}% of the window's cost."
    )
    proposed_fix = (
        SUBAGENT_RUBRIC_INTRO
        + f"\n\nObserved oversized dispatches ran on: {model_list}. Route that "
        "shape to the cheaper same-family model next time."
    )
    # Apply-capable when we have a concrete model to name in the rubric; else
    # degrade to advise-only (no clean workspace surface to write).
    apply_capable = bool(models)
    # The stronger surface, when the flagged subagent has a definition file: set
    # its `model:` key outright instead of writing a rubric the orchestrator has
    # to read and honor. Falls back to that rubric when there is no file.
    try:
        agent_apply = _agent_model_plumbing(over_powered, config)
    except Exception:
        agent_apply = {}
    if agent_apply:
        advise_extra = (
            f" {agent_apply['agent_name']} has its own definition file, so "
            f"tokenjam can set its model key to "
            f"{agent_apply['proposed_model']} directly. The change is committed "
            f"where the file is in a repo and reverts in one call. Its next "
            f"dispatch runs on the new model, which is where measurement starts."
        )
        return [CostProposal(
            kind="cost",
            analyzer="subagent",
            signature=f"cost:subagent:{agent_apply['agent_name']}",
            title=(
                f"Over-powered subagent {agent_apply['agent_name']} "
                f"({agent_apply['current_model']} to {agent_apply['proposed_model']})"
            ),
            target_key={
                "models": models, "subagent": True,
                "agent_name": agent_apply["agent_name"],
            },
            evidence=evidence,
            baseline={
                "flagged_subagents": subagents,
                "flagged_cost_usd": float(getattr(finding, "flagged_cost_usd", 0.0) or 0.0),
                "subagent_cost_usd": float(getattr(finding, "subagent_cost_usd", 0.0) or 0.0),
                "percent_of_cost": float(getattr(finding, "percent_of_cost", 0.0) or 0.0),
                "agent_name": agent_apply["agent_name"],
                "current_model": agent_apply["current_model"],
                "proposed_model": agent_apply["proposed_model"],
            },
            advise_text=(
                "Lower the model tier for the flagged Task dispatches. "
                + str(getattr(finding, "caveat", "") or "") + advise_extra
            ).strip(),
            suggestion=f"model: {agent_apply['proposed_model']}",
            one_paste_fix=(
                f"# In {agent_apply['target_path']}, frontmatter:\n"
                f"model: {agent_apply['proposed_model']}"
            ),
            past_overspend_usd=getattr(finding, "past_overspend_usd", None),
            past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            advise_only=False,
            apply_capable=True,
            scope=agent_apply["scope"],
            apply_kind=agent_apply["apply_kind"],
            agent_name=agent_apply["agent_name"],
            target_path=agent_apply["target_path"],
            current_model=agent_apply["current_model"],
            proposed_model=agent_apply["proposed_model"],
        )]
    return [CostProposal(
        kind="cost",
        analyzer="subagent",
        signature="cost:subagent",
        title="Over-powered subagent dispatches (route the fan-out to a cheaper model)",
        target_key={"models": models, "subagent": True},
        evidence=evidence,
        baseline={
            "flagged_subagents": subagents,
            "flagged_cost_usd": float(getattr(finding, "flagged_cost_usd", 0.0) or 0.0),
            "subagent_cost_usd": float(getattr(finding, "subagent_cost_usd", 0.0) or 0.0),
            "percent_of_cost": float(getattr(finding, "percent_of_cost", 0.0) or 0.0),
        },
        advise_text=(
            "Lower the model tier for the flagged Task dispatches. On Claude Code "
            "this is a sizing rubric in your CLAUDE.md (apply it below) that the "
            "orchestrating agent reads before it spawns subagents. "
            + str(getattr(finding, "caveat", "") or "")
        ).strip(),
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        advise_only=not apply_capable,
        apply_capable=apply_capable,
        rung=1 if apply_capable else 0,
        scope="project" if apply_capable else "",
        proposed_fix=proposed_fix if apply_capable else "",
    )]


def _mcp_remove_plumbing(server: Any) -> dict[str, Any]:
    """Whether ``server``'s config entry can be removed directly, and where.

    Unlike ``model_swap`` there is no search step: ``ConfiguredServer``
    already resolved the exact config file at detection time. This just
    re-verifies that still holds (the file can have moved, or a human can
    have already removed the entry by hand) at proposal-build time, so the
    card's pre-filled target is current, not stale analyzer-time data.
    """
    from tokenjam.core.optimize.analyzers.deadweight import (
        APPLY_KIND_MCP_REMOVE,
        mcp_remove_precheck,
    )

    check = mcp_remove_precheck(server.source, server.name)
    if not check["ok"]:
        return {"apply_capable": False, "apply_blocked_reason": check["reason"]}
    return {
        "apply_capable": True,
        "apply_kind": APPLY_KIND_MCP_REMOVE,
        "source_path": check["target_path"],
        "target_path": check["target_path"],
        "apply_blocked_reason": "",
    }


def _deadweight_to_proposals(finding: Any) -> list[CostProposal]:
    """One proposal per dead-weight MCP server (Component C1).

    Reads ONLY ``DeadweightFinding.dead_servers`` — the C2 tax table (which
    lists every configured server, dead or alive, purely for ranked
    visibility) never feeds a proposal here, so a server's schema-injection
    tax is never counted both in the tax table AND a proposal (the same
    dedup guarantee ``compute_deadweight_finding`` itself enforces on
    ``past_overspend_tokens`` / ``past_overspend_usd``).

    ``past_overspend_usd`` is carried straight off the analyzer's own
    ``ServerDeadweight.estimated_tax_usd_window`` — a WINDOW-scoped figure
    with no projection folded in (#273: this used to be a fixed 90-day
    projection, which made it incomparable to every other analyzer's window
    figure and corrupted any sum across them). The analyzer already prices
    the token tax through ``core/pricing.py`` at the dominant model observed
    in that server's sessions (never a hardcoded rate; see
    ``deadweight._pricing_note``). Stays ``None`` when no priced model was
    observed for that server — this adapter never invents a rate itself.

    Apply-capable, like ``downsize``'s ``model_swap`` cards: the fix is a
    deterministic edit of a value already written down (the server's own
    ``mcpServers`` entry), so it routes through the same
    ``relearn_apply.apply_relearn_fix`` machinery under
    ``APPLY_KIND_MCP_REMOVE`` — reversible, git-committed where the config
    lives in a repo, one-step revert. Falls back to the one-paste ``claude
    mcp remove`` command, with the reason stated, when the precondition
    doesn't hold (file missing, malformed, or the entry already gone).
    """
    if finding is None:
        return []
    proposals: list[CostProposal] = []
    for server in getattr(finding, "dead_servers", []) or []:
        evidence = (
            f"`{server.name}` MCP server ({server.scope} scope, configured at "
            f"{server.source}) made 0 tool calls across {server.sessions_present} "
            f"session(s) in the window."
        )
        if server.deferred_sessions:
            evidence += (
                f" ToolSearch deferred its schema in {server.deferred_sessions} "
                f"of those session(s)."
            )
        scope_flag = "user" if server.scope == "user" else "project"
        plumbing = _mcp_remove_plumbing(server)
        advise = (
            server.fix + " Removing (or project-scoping) it is reversible "
            "and loses no data; it only stops the standing schema-injection "
            "tax on future sessions."
        )
        if plumbing.get("apply_capable"):
            advise += (
                f" tokenjam can remove this exact entry from "
                f"{plumbing['target_path']}, with the change committed and "
                f"revertable in one call."
            )
        elif plumbing.get("apply_blocked_reason"):
            advise += f" Applying it here is not on offer: {plumbing['apply_blocked_reason']}"
        proposals.append(CostProposal(
            kind="cost",
            analyzer="deadweight",
            signature=f"cost:deadweight:{server.name}",
            title=f"Unused MCP server: {server.name}",
            target_key={
                "server": server.name, "scope": server.scope, "source": server.source,
            },
            evidence=evidence,
            baseline={
                "sessions_present": server.sessions_present,
                "invocations": server.invocations,
                "deferred_sessions": server.deferred_sessions,
                "scope": server.scope,
                "source": server.source,
                "example_sessions": list(server.example_sessions),
                "priced_model": server.priced_model,
            },
            advise_text=advise,
            suggestion=f"claude mcp remove {server.name} --scope {scope_flag}",
            past_overspend_tokens=server.estimated_tax_tokens_window or None,
            past_overspend_usd=server.estimated_tax_usd_window,
            estimate_basis=server.tax_construction,
            advise_only=not plumbing.get("apply_capable", False),
            apply_capable=bool(plumbing.get("apply_capable")),
            apply_kind=str(plumbing.get("apply_kind", "")),
            agent_name=server.name,
            source_path=str(plumbing.get("source_path", "")),
            target_path=str(plumbing.get("target_path", "")),
            scope=server.scope,
            apply_blocked_reason=str(plumbing.get("apply_blocked_reason", "")),
        ))
    return proposals


def _cluster_hash(value: Any) -> str:
    """Stable 12-hex-char identity for a cluster's structural key. Deterministic
    across runs over the same underlying signature (a JSON-serialisable
    structure), used only where the analyzer itself doesn't already hand back
    a cluster id (contrast ``ReuseCluster.cluster_id``, which does)."""
    encoded = json.dumps(value, sort_keys=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Persona-gated fix modality — `script` / `reuse` only. (`verbosity` shares
# the same write SURFACE — see its own module comment above `_verbosity_to_
# proposals` — but never offers the write at all, for any persona: its
# cohort-scoped flag would become a global CLAUDE.md rule if written, which
# fails the no-quality-tax gate outright regardless of who reads it.)
#
# Both write the SAME class of artifact when apply-capable: a rung-1
# CLAUDE.md note or rung-2 `.claude/skills/<slug>/SKILL.md` file (see
# `relearn_apply.default_target_path`). Nothing in an SDK-only service's
# request path ever reads a CLAUDE.md or a `.claude/skills/` note — those are
# read by an interactive coding-agent harness. Offering that write to an SDK
# caller is a write that visibly succeeds and changes nothing: a quiet lie in
# the user's favour (CLAUDE.md anti-pattern #22). The finding underneath is
# still true for an SDK caller; only the fix MODALITY is wrong, so this never
# drops the recommendation — it demotes the card to advise-only and carries
# the identical text as a copy-pasteable `suggestion` instead (the same
# ``CostProposal.suggestion`` field every advise-only card already renders as
# a first-class "the fix" block with a Copy button).
# --------------------------------------------------------------------------- #

def _persona_gated_write_fields(
    persona: str, proposed_fix: str, rung: int, scope: str,
) -> dict[str, Any]:
    """Decide, from the window's dominant persona, whether the rung-1/rung-2
    workspace write is offered — and fill in the ``CostProposal`` fields that
    follow from that decision.

    * ``"claude-code"`` — unchanged: the write is genuinely actionable, so it
      stays offered exactly as before.
    * ``"sdk"`` and ``"unknown"`` — no write offered. ``"unknown"`` means no
      session in the window carries an identifiable agent_id and no declared
      plan settles it either (`core.framing.dominant_persona`) — the exact
      shape of a pure-SDK caller who never ran ``tj onboard``. Grouping it
      with ``"sdk"`` here mirrors ``cmd_optimize._render_downgrade_cta``'s
      CTA, and avoids the one failure mode called out for this fix: silently
      offering a write to a persona that turns out to be SDK.
    * ``"mixed"`` — both audiences are meaningfully represented (same
      precedent as ``_render_downgrade_cta``, which renders both CTAs side
      by side rather than picking one), and a script/reuse/verbosity finding
      isn't attributable to one side of the mix or the other. The write
      stays on offer for the claude-code share; the identical recommendation
      is ALSO carried as ``suggestion`` so the sdk share of the mix isn't
      left with a card that looks actionable but silently isn't, for them.
    """
    write_offered = persona in {"claude-code", "mixed"}
    fields: dict[str, Any] = {
        "advise_only": not write_offered,
        "apply_capable": write_offered,
        "rung": rung if write_offered else 0,
        "scope": scope if write_offered else "",
        "proposed_fix": proposed_fix if write_offered else "",
    }
    # Every persona except a clean claude-code window also gets the snippet
    # fallback — "mixed" needs it alongside the write (see above), and
    # "sdk"/"unknown" need it in place of the write.
    if persona != "claude-code":
        fields["suggestion"] = proposed_fix
    return fields


def _script_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per flagged deterministic-tool-call cluster.

    Apply-capable at rung 2: a skill note naming the repeated call pattern and
    recommending a script in its place. No agent-file/model-swap surface
    exists here (this isn't a model-routing finding), so unlike ``subagent``
    there is only the one apply shape. The skill's slug is derived from the
    title, which embeds the cluster's own hash so two clusters never collide
    on the same skill file (`relearn_apply`'s create-only guard would otherwise
    let a second cluster's apply silently overwrite the first's skill note).

    The write is only actually offered for a ``"claude-code"``/``"mixed"``
    ``persona`` — see ``_persona_gated_write_fields``. An ``"sdk"``/
    ``"unknown"`` window gets the identical recommendation as a
    copy-pasteable snippet instead; the skill note would sit unread by any
    SDK service's request path.
    """
    if finding is None:
        return []
    clusters = list(getattr(finding, "clusters", []) or [])
    if not clusters:
        return []
    degraded = bool(getattr(finding, "degraded", False))
    caveat = str(getattr(finding, "caveat", "") or "")

    proposals: list[CostProposal] = []
    for cluster in clusters:
        if cluster.total_cost_usd <= 0 and cluster.total_tokens <= 0:
            continue
        cluster_hash = _cluster_hash(cluster.signature)
        tool_names = [step.get("tool", "?") for step in cluster.signature]
        tool_list = " -> ".join(tool_names) or "(no tools recorded)"
        title = f"Deterministic tool pattern: {tool_list} ({cluster_hash})"
        evidence = (
            f"{cluster.instances} sessions ran the same tool-call structure "
            f"({tool_list}), averaging {cluster.avg_tokens:,} input+output "
            f"tokens and {_money(cluster.avg_cost_usd)} per session."
        )
        if degraded:
            evidence += (
                " Clustered on tool names only (enable [capture] tool_inputs "
                "in tj.toml for the finer argument-shape signature)."
            )
        advise = (
            _SCRIPT_SKILL_INTRO + " " + caveat
        ).strip()
        proposals.append(CostProposal(
            kind="cost",
            analyzer="script",
            signature=f"cost:script:{cluster_hash}",
            title=title,
            target_key={"signature": cluster.signature, "instances": cluster.instances},
            evidence=evidence,
            baseline={
                "instances": cluster.instances,
                "avg_cost_usd": cluster.avg_cost_usd,
                "avg_tokens": cluster.avg_tokens,
                "avg_duration_seconds": cluster.avg_duration_seconds,
                "example_session_id": cluster.example_session_id,
                "degraded": degraded,
                "apply_sessions": cluster.instances,
                "apply_examples": [
                    {"session_id": sid, "repo": "", "snippet": tool_list[:160]}
                    for sid in (cluster.example_session_ids or [])
                ],
            },
            advise_text=advise,
            past_overspend_usd=cluster.total_cost_usd or None,
            past_overspend_tokens=cluster.total_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            **_persona_gated_write_fields(persona, advise, rung=2, scope="project"),
        ))
    return proposals


def _reuse_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal per repeated planning-skeleton cluster.

    Apply-capable at rung 1: a CLAUDE.md note naming the recurring skeleton.
    Uses the finding's conservative ``cache_reuse_recoverable_*`` figure (you
    already paid for the plan once), not the ``script_replacement_*`` upper
    bound, matching ``ReuseFinding``'s own aggregate.

    The write is only actually offered for a ``"claude-code"``/``"mixed"``
    ``persona`` — see ``_persona_gated_write_fields``.
    """
    if finding is None:
        return []
    clusters = list(getattr(finding, "clusters", []) or [])
    if not clusters:
        return []

    proposals: list[CostProposal] = []
    for cluster in clusters:
        if cluster.cache_reuse_recoverable_usd <= 0 and cluster.cache_reuse_recoverable_tokens <= 0:
            continue
        tool_list = ", ".join(cluster.tool_signature) or "(no tool calls after the plan)"
        title = f"Repeated planning skeleton: {tool_list} ({cluster.cluster_id})"
        evidence = (
            f"{cluster.repetitions} sessions shared a planning-call skeleton "
            f"(tool sequence after the plan: {tool_list}), averaging "
            f"{cluster.avg_planning_tokens:,} planning tokens "
            f"({_money(cluster.avg_planning_cost_usd)} per call)."
        )
        advise = (
            _REUSE_NOTE_INTRO + " " + str(cluster.caveat or "")
        ).strip()
        proposals.append(CostProposal(
            kind="cost",
            analyzer="reuse",
            signature=f"cost:reuse:{cluster.cluster_id}",
            title=title,
            target_key={
                "cluster_id": cluster.cluster_id,
                "tool_signature": list(cluster.tool_signature),
            },
            evidence=evidence,
            baseline={
                "repetitions": cluster.repetitions,
                "avg_planning_tokens": cluster.avg_planning_tokens,
                "avg_planning_cost_usd": cluster.avg_planning_cost_usd,
                "skeleton_session_id": cluster.skeleton_session_id,
                "apply_sessions": cluster.repetitions,
                "apply_examples": [
                    {"session_id": sid, "repo": "", "snippet": tool_list[:160]}
                    for sid in (cluster.example_session_ids or [])
                ],
            },
            advise_text=advise,
            past_overspend_usd=cluster.cache_reuse_recoverable_usd or None,
            past_overspend_tokens=cluster.cache_reuse_recoverable_tokens or None,
            estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
            **_persona_gated_write_fields(persona, advise, rung=1, scope="project"),
        ))
    return proposals


def _verbosity_to_proposals(finding: Any, persona: str = "unknown") -> list[CostProposal]:
    """One proposal for the whole verbosity finding (unlike ``script``/
    ``reuse``, this is a single window-wide signal, not per-cluster).

    ALWAYS advise-only, regardless of ``persona`` — no rung-1 CLAUDE.md write
    is ever offered here, unlike ``script``/``reuse`` which share the same
    write machinery. A cohort-scoped flag (this task shape ran long vs its
    OWN like-shaped peers) written as a global CLAUDE.md note would apply
    blanket terseness pressure to every future task, including legitimately
    verbose ones — the analyzer's own honesty caveat says output length is
    not waste, so an enforced-looking file note fails that gate outright.
    The recommendation is still carried as a copy-pasteable ``suggestion``
    for every persona (never silently dropped), it just never becomes a
    write. ``persona`` is accepted (and kept in the dispatcher's uniform
    ``(f, persona=persona)`` call shape) but intentionally unused — see
    above.
    """
    if finding is None:
        return []
    total_candidates = int(getattr(finding, "total_candidates", 0) or 0)
    if total_candidates <= 0:
        return []
    remedy = str(getattr(finding, "remedy_snippet", "") or "")
    max_tokens = getattr(finding, "suggested_max_tokens", None)
    caveat = str(getattr(finding, "caveat", "") or "")
    evidence = (
        f"{total_candidates} session(s) across "
        f"{int(getattr(finding, 'cohorts_examined', 0) or 0)} task-shape "
        f"cohort(s) ran output well above their cohort's median."
    )
    advise = remedy
    if max_tokens:
        # A DATA POINT, not an enforceable cap — "keep responses under N
        # tokens" reads as a rule once pasted anywhere; this states what
        # similar sessions happened to do instead.
        advise += (
            f" For reference, similar sessions in this cohort typically "
            f"finished under about {max_tokens:,} output tokens — a data "
            f"point to weigh, not a limit to enforce."
        )
    advise = (advise + " " + caveat).strip()
    return [CostProposal(
        kind="cost",
        analyzer="verbosity",
        signature="cost:verbosity",
        title="High-verbosity output vs cohort baseline",
        target_key={"total_candidates": total_candidates},
        evidence=evidence,
        baseline={
            "total_candidates": total_candidates,
            "sessions_examined": int(getattr(finding, "sessions_examined", 0) or 0),
            "cohorts_examined": int(getattr(finding, "cohorts_examined", 0) or 0),
            "suggested_max_tokens": max_tokens,
            "apply_sessions": total_candidates,
        },
        advise_text=advise,
        suggestion=advise,
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
    )]


def _rightsize_target(subagent_finding: Any, config: Any) -> dict[str, Any]:
    """The concrete agent file the compound offload card should name, if one
    exists: the most expensive over-powered subagent that has its own
    definition file, plus the cheaper model to pin in it.

    Reuses ``_agent_model_plumbing`` — the same resolver the ``subagent`` card
    already uses — rather than re-deriving a path, so the two cards can never
    name different files for the same agent. Empty dict when the subagent
    analyzer didn't run, nothing was flagged, or no flagged subagent has a
    definition file; the card then states the right-sizing lever generically
    instead of inventing an agent name.
    """
    if subagent_finding is None:
        return {}
    flagged = list(getattr(subagent_finding, "flagged", []) or [])
    over_powered = [r for r in flagged if "over_powered" in (getattr(r, "flags", []) or [])]
    if not over_powered:
        return {}
    try:
        return _agent_model_plumbing(over_powered, config)
    except Exception:
        return {}


def _compound_offload_fix(rightsize: dict[str, Any], fix_offload: str, fix_rightsize: str) -> str:
    """The single rung-1 rule that carries BOTH halves of the compound lever.

    One artifact, not two: the offload directive decides where context-heavy
    work runs, the right-sizing directive decides what it runs on, and they
    compound. Writing them as one block is what keeps this a consolidation of
    the resend / subagent / downsize recommendations rather than a third card
    on top of them.
    """
    parts = [fix_offload, fix_rightsize]
    if rightsize:
        parts.append(
            f"Concretely: {rightsize['agent_name']} already runs oversized for "
            f"the work it does — pin `model: {rightsize['proposed_model']}` in "
            f"{rightsize['target_path']} and set its reasoning effort to match "
            f"the task rather than inheriting the parent's."
        )
    return "\n\n".join(p for p in parts if p)


def _rightsize_frontmatter_snippet(rightsize: dict[str, Any]) -> str:
    """The copyable agent-file frontmatter for the right-sizing half.

    Both keys live in the same ``.claude/agents/<name>.md`` frontmatter block,
    so the second half of the compound fix is one paste, not two.
    """
    name = rightsize.get("agent_name") or "<subagent-name>"
    model = rightsize.get("proposed_model") or "<cheaper-same-family-model>"
    path = rightsize.get("target_path") or f".claude/agents/{name}.md"
    return (
        f"# {path} — frontmatter\n"
        f"---\n"
        f"name: {name}\n"
        f"model: {model}\n"
        f"reasoning_effort: low\n"
        f"---"
    )


def _resend_to_proposals(
    finding: Any, persona: str = "unknown",
    subagent_finding: Any = None, config: Any = None,
) -> list[CostProposal]:
    """One window-wide card for the ``resend`` (context re-send) finding.

    This card is COMPOUND by design: it consolidates what resend and subagent
    right-sizing would otherwise say on separate cards, because the two are one
    behavioural change (offload context-heavy work to a subagent, and size that
    subagent to the work) applied through one rung-1 rule. Consolidating rather
    than adding is deliberate — the Review inbox does not grow.

    Persona-gated like every other lever-bearing adapter. Three levers exist
    and they are NOT interchangeable:

    * a rung-1 CLAUDE.md rule instructing offload of context-heavy sub-tasks
      to subagents (``fix_subagent_offload``) is the DURABLE claude-code
      lever: it persists across sessions and stops the repeated volume from
      accumulating on the main thread in the first place. Apply-capable via
      the same ``_persona_gated_write_fields`` machinery ``script`` /
      ``reuse`` / ``verbosity`` already use, for a ``claude-code``/``mixed``
      persona — see that helper.
    * ``/compact`` / a fresh session is a MANUAL, per-session, transient
      action available to everyone, but it fixes nothing going forward (a
      real CC user who feels a session getting too full typically abandons
      it and starts fresh anyway rather than compacting). It is carried only
      as a secondary, immediate-relief note for an already-full session —
      never the headline fix.
    * an SDK-side ``cache_control`` breakpoint is the SDK/API developer's
      ADDITIONAL lever. A Claude Code (or other agent-harness) window never
      constructs the request itself, so that snippet is not theirs to paste —
      showing it to them is the same wrong-audience defect the cache family
      already avoids via ``_persona_gated_cache_fields``. Suppressed for a
      ``claude-code`` persona, unchanged from before.

    ``ResendFinding.estimate_basis`` documents the discounted derivation; this
    adapter carries it verbatim. The caveat is carried ONCE via ``caveat=`` and
    deliberately NOT folded into ``advise_text`` — doing both printed it twice
    on the card (the description and the caveat line rendered the same sentence).
    """
    if finding is None:
        return []
    repeat_share = getattr(finding, "repeat_share", None)
    if repeat_share is None:
        return []
    sessions_examined = int(getattr(finding, "sessions_examined", 0) or 0)
    repeat_tokens = int(getattr(finding, "repeat_tokens", 0) or 0)
    evidence = (
        f"{float(repeat_share) * 100:.0f}% of prompt tokens across "
        f"{sessions_examined} session(s) were context already sent in an "
        f"earlier turn (conservative lower bound; independent of whether "
        f"caching is enabled)."
    )
    cost_of_waste_usd = getattr(finding, "cost_of_waste_usd", None)
    if cost_of_waste_usd is not None:
        # Deliberately does NOT restate the dollar figure: the card renders it
        # once, on its own COST line (`observed_cost_usd`), immediately
        # followed by the coverage note. This sentence used to carry the number
        # AND end "the rest is what multi-turn work inherently re-sends" —
        # asserting that everything outside the avoidable figure had been shown
        # to be necessary. It had not been: most of it is simply outside the
        # avoidability analysis, which `coverage_note` now spells out.
        evidence += (
            " What that volume cost is reported below as cost, not waste: it is "
            "an observation of what already billed, not a claim about what a "
            "fix returns."
        )
    fix_compaction = str(getattr(finding, "fix_compaction", "") or "")
    fix_cache_control = str(getattr(finding, "fix_cache_control", "") or "")
    fix_subagent_offload = str(getattr(finding, "fix_subagent_offload", "") or "")
    fix_rightsize = str(getattr(finding, "fix_rightsize", "") or "")
    # The cache_control snippet is the SDK lever only; a claude-code window
    # can't paste it, so suppress it there — unchanged from before.
    cache_snippet = "" if persona == "claude-code" else fix_cache_control
    rightsize = _rightsize_target(subagent_finding, config)

    if persona in {"claude-code", "mixed"} and fix_subagent_offload:
        compound_fix = _compound_offload_fix(rightsize, fix_subagent_offload, fix_rightsize)
        advise = compound_fix
        if fix_compaction:
            advise = advise + " Immediate relief in an already-full session: " + fix_compaction
        write_fields = _persona_gated_write_fields(
            persona, compound_fix, rung=1, scope=rightsize.get("scope") or "project",
        )
        # resend's `suggestion` slot is reserved for the SDK cache_control
        # snippet above, not the write-fallback text the helper would add
        # for a "mixed" persona — drop it so the two don't collide.
        write_fields.pop("suggestion", None)
        # The second half of the compound fix: the agent-file frontmatter that
        # pins model AND reasoning effort. Carried as the one-paste artifact
        # because the rung-1 write lands in CLAUDE.md, and the apply machinery
        # writes exactly one target per apply.
        one_paste_fix = _rightsize_frontmatter_snippet(rightsize)
    elif persona == "sdk":
        # `/compact` is a Claude Code interactive command an SDK caller has
        # no access to, so it must never be the advise text here — see
        # RESEND_SDK_TRIM_FIX's docstring in context_resend.py. Lead with the
        # cache_control snippet when a priced example produced one; fall
        # back to a persona-neutral, call-site instruction otherwise.
        from tokenjam.core.optimize.analyzers.context_resend import (
            RESEND_SDK_TRIM_FIX,
        )
        advise = (
            "Adopt a cache_control breakpoint at the call site (snippet "
            "below) so this repeated context bills at the cache rate "
            "instead of full price every turn."
        ) if cache_snippet else RESEND_SDK_TRIM_FIX
        write_fields = {
            "advise_only": True, "apply_capable": False,
            "rung": 0, "scope": "", "proposed_fix": "",
        }
        one_paste_fix = cache_snippet or RESEND_SDK_TRIM_FIX
    else:
        advise = fix_compaction
        write_fields = {
            "advise_only": True, "apply_capable": False,
            "rung": 0, "scope": "", "proposed_fix": "",
        }
        one_paste_fix = cache_snippet or fix_compaction

    return [CostProposal(
        kind="cost",
        analyzer="resend",
        signature="cost:resend",
        title="Repeated context re-sent every turn",
        target_key={"repeat_share": float(repeat_share)},
        evidence=evidence,
        baseline={
            "sessions_examined": sessions_examined,
            "repeat_tokens": repeat_tokens,
            "repeat_share": float(repeat_share),
            "repeat_share_median": getattr(finding, "repeat_share_median", None),
            "repeat_share_p90": getattr(finding, "repeat_share_p90", None),
            "offloadable_share": getattr(finding, "offloadable_share", None),
            "offloadable_share_sessions": getattr(finding, "offloadable_share_sessions", 0),
            "offloadable_share_sessions_total": getattr(finding, "offloadable_share_sessions_total", 0),
            "offloadable_share_median": getattr(finding, "offloadable_share_median", None),
            "offload_recoverable_usd": getattr(finding, "offload_recoverable_usd", None),
            "rightsize_recoverable_usd": getattr(finding, "rightsize_recoverable_usd", None),
            "rightsize_agent_name": rightsize.get("agent_name", ""),
            "rightsize_target_path": rightsize.get("target_path", ""),
            # The two figures' differing populations, carried machine-readable
            # alongside the prose `coverage_note` so the split is auditable.
            "cost_in_scope_usd": getattr(finding, "cost_in_scope_usd", None),
            "cost_driver_role_usd": getattr(finding, "cost_driver_role_usd", None),
            "cost_no_lever_usd": getattr(finding, "cost_no_lever_usd", None),
            "offload_ceiling_usd": getattr(finding, "offload_ceiling_usd", None),
            "sessions_in_scope": getattr(finding, "sessions_in_scope", 0),
            "sessions_no_lever": getattr(finding, "sessions_no_lever", 0),
            "driver_role_sessions": getattr(finding, "driver_role_sessions", 0),
        },
        advise_text=advise,
        suggestion=cache_snippet,
        one_paste_fix=one_paste_fix,
        past_overspend_usd=getattr(finding, "past_overspend_usd", None),
        past_overspend_tokens=getattr(finding, "past_overspend_tokens", None),
        cost_of_waste_usd=cost_of_waste_usd,
        cost_of_waste_tokens=getattr(finding, "cost_of_waste_tokens", None) or None,
        cost_of_waste_basis=str(getattr(finding, "cost_of_waste_basis", "") or ""),
        coverage_note=str(getattr(finding, "coverage_note", "") or ""),
        estimate_basis=str(getattr(finding, "estimate_basis", "") or ""),
        caveat=str(getattr(finding, "caveat", "") or COST_CORRELATIONAL_CAVEAT),
        **write_fields,
    )]


#: Default look-back for the daemon/CLI cost-proposal recompute. Matches the
#: monthly framing the cost analyzers project against.
DEFAULT_COST_WINDOW_DAYS = 30


#: Mirrors ``relearn_store``'s own ``_LOCK``/``_COMPUTING`` pair, kept local to
#: this module since a cost-proposals recompute and a relearn recompute are
#: independent jobs that must each be able to run without waiting on the
#: other — only two cost-proposals recomputes (a scheduled tick racing a
#: manual "Rescan now") should ever serialize against each other.
_COST_LOCK = threading.Lock()
_COST_COMPUTING = threading.Event()


def is_computing_cost_proposals() -> bool:
    return _COST_COMPUTING.is_set()


#: The Review inbox's cross-reference for waste a caller has decided NOT to
#: sum as a peer card. Generic infrastructure (see ``estimated_recoverable_
#: rollup``'s ``excluded`` parameter and the UI's ``ExcludedWasteNote``) with
#: no current occupant: ``summarize`` used to be the one entry here (issue
#: #326) until summarize got a real peer card instead (see
#: ``_summarize_to_proposals`` and ``COST_ANALYZERS``'s docstring) — kept
#: rather than deleted because the shape (state the total, link to where it
#: lives, never silently drop it) is the right move for a FUTURE analyzer
#: whose fix has no representable inbox card, should one appear.


def recompute_cost_proposals(
    db: Any,
    config: Any,
    *,
    window_days: int = DEFAULT_COST_WINDOW_DAYS,
    agent_id: str | None = None,
) -> list[CostProposal]:
    """Build an ``OptimizeReport`` over the last ``window_days``, adapt the
    cost findings into proposals, and write them into the shared proposal
    store. Returns the proposals (``[]`` if a recompute is already in flight,
    or on a build failure — the inbox shows its empty/degraded state, never a
    crash on the refresh path).

    This is the "daemon path produces findings -> same proposal store" entry
    point the Review-inbox refresh calls; ``tj optimize`` can call it too so a
    manual run also refreshes the inbox. Locked against concurrent recomputes
    (the scheduled job and a manual "Rescan now" can otherwise overlap and
    race each other's cache write) — a caller that hits the lock gets ``[]``
    back rather than blocking, same as a build failure.

    A failure is never silent: the exception is recorded via
    ``relearn_store.write_cost_proposals_error`` (behavioral requirement #5)
    so the Review inbox can show a "last refresh failed" warning instead of
    reading a permanently-empty tab as "nothing to report." A SUCCESSFUL
    recompute clears any previously-recorded error.
    """
    from datetime import timedelta

    from tokenjam.core.framing import (
        agent_persona_mix,
        config_declared_plan,
        dominant_persona,
        dominant_plan,
        plan_tier_mix,
        pricing_mode_for,
    )
    from tokenjam.core.optimize import relearn_store
    from tokenjam.core.optimize.runner import build_report
    from tokenjam.utils.time_parse import utcnow

    if not _COST_LOCK.acquire(blocking=False):
        return []
    _COST_COMPUTING.set()
    try:
        try:
            effective_window_days = max(1, window_days)
            until = utcnow()
            since = until - timedelta(days=effective_window_days)
            conn = getattr(db, "conn", None)
            # Persona decides which cost analyzers are worth running at all —
            # one with no fix this persona can apply is dropped BEFORE the
            # report is built, so it never queries and never reaches the
            # inbox. `build_report` applies the same gate internally (it is
            # the choke point); selecting the persona-scoped list here keeps
            # this surface honest on its own terms rather than relying on the
            # callee to undo an over-broad request.
            persona = dominant_persona(
                agent_persona_mix(conn, since, until, agent_id=agent_id) if conn is not None else {},
                declared_plan=config_declared_plan(config),
            )
            report = build_report(
                db, config, since, until, agent_id=agent_id,
                # `summarize` IS a COST_ANALYZER now and would already
                # be selected by `cost_analyzers_for_persona`; this is the
                # PERSONA-SCOPED list, not the raw one — the skip gate still
                # decides which cost analyzers run for this window.
                findings=list(cost_analyzers_for_persona(persona)),
            )
            # Same plan-tier -> pricing-mode resolution `tj optimize` uses, so
            # the web Review inbox suppresses the same dollar figures the CLI
            # does (placement's batch-lever dollars, currently the only card
            # this gates — see `_placement_to_proposals`).
            plan_mix = plan_tier_mix(conn, since, until, agent_id) if conn is not None else {}
            pricing_mode = pricing_mode_for(dominant_plan(plan_mix))
            proposals = cost_proposals_from_report(
                report, config=config, pricing_mode=pricing_mode,
                window_days=float(effective_window_days),
            )
        except Exception as exc:
            try:
                relearn_store.write_cost_proposals_error(str(exc), config=config)
            except Exception:
                pass
            return []

        try:
            relearn_store.write_cost_proposals(
                proposals, config=config,
                window_days=effective_window_days,
            )
            relearn_store.clear_cost_proposals_error(config=config)
        except Exception:
            pass
        return proposals
    finally:
        _COST_COMPUTING.clear()
        _COST_LOCK.release()


def trigger_background_cost_recompute(
    backend_factory: Callable[[], Any],
    *,
    config: Any | None = None,
    window_days: int = DEFAULT_COST_WINDOW_DAYS,
) -> bool:
    """Fire-and-forget a cost-proposals recompute on a daemon thread — the
    Cost-advisories-tab equivalent of ``relearn_store.
    trigger_background_recompute``, so ``tj serve`` can keep this tab warm on
    a schedule (behavioral requirement #5) without blocking startup or a
    request thread on the analyzer run.

    ``backend_factory`` builds a FRESH ``StorageBackend`` (e.g. ``lambda:
    DuckDBBackend(config.storage)``) — never the caller's live request
    connection. The backend is closed when the job finishes. Returns
    ``False`` (no-op) if a recompute is already running.
    """
    if is_computing_cost_proposals():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            recompute_cost_proposals(backend, config, window_days=window_days)
        except Exception:
            # Best-effort background job — never crash the scheduler/thread.
            pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="cost-proposals-recompute", daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# PACING IS GONE FROM THIS MODULE, DELIBERATELY.
#
# `compute_projection_ratio()` used to live here: `r = clamp(30/active_days,
# 1.0, 3.0)`, the one shared multiplier turning each analyzer's window figure
# into an `estimated_monthly_*` forecast. Both are deleted. The product no
# longer makes a forward per-analyzer claim, so the only thing `r` still did
# was produce a THIRD dollar quantity (measured 7.14% larger than the window
# figure on the reference corpus) that a surface could render by mistake —
# which is exactly what happened.
#
# Do not reintroduce a ratio here. The 30-day pace basis still exists for the
# ONE calculation that genuinely needs a per-month horizon —
# `core/optimize/projection.build_projection_basis`, used by
# `core/optimize/write_budget` to price how many future sessions will re-send
# a permanent CLAUDE.md rule. It is reached only through `_write_budget_basis`
# below, it never touches a past-tense figure, and nothing it produces lands
# on a payload. See the field contract in the repo `CLAUDE.md`.
# --------------------------------------------------------------------------- #


def cost_proposals_from_report(
    report: Any, config: Any = None, *, pricing_mode: str = "api",
    window_days: float = 30.0,
) -> list[CostProposal]:
    """Every cost proposal derivable from an already-built ``OptimizeReport``.

    Reads the ``downsize`` finding off the typed ``report.downgrade`` slot and
    the ``cache`` / ``cache-recommend`` / ``trim`` / ``subagent`` /
    ``placement`` / ``deadweight`` / ``script`` / ``reuse`` / ``verbosity`` /
    ``resend`` / ``summarize`` findings off ``report.findings``. Missing
    findings (analyzer not run, no candidates) contribute nothing. Never
    raises — a malformed finding is skipped so one bad analyzer can't sink
    the inbox.

    ``config`` is optional and used for one thing: looking up the local source
    path a user registered for an agent, which decides whether the downsize card
    can offer the gated model-id swap or falls back to its one-paste artifact.
    Without it every card is advise-only.

    ``pricing_mode`` gates the ``placement`` card's dollar figure — the Batch
    API discount is an api-billed lever, so a subscription/local caller gets
    the advise text without a number, same as the CLI. Defaults to ``"api"``
    so existing callers that don't know their caller's plan keep today's
    behaviour.

    ``script`` / ``reuse`` read ``report.persona`` (set once by ``runner.
    build_report`` — see ``AnalyzerContext.persona``) to decide whether their
    rung-1/rung-2 workspace write is offered at all — see
    ``_persona_gated_write_fields``. ``verbosity`` reads the same finding
    shape but never offers a write for ANY persona (see ``_verbosity_to_
    proposals``) — a cohort-scoped flag written as a global CLAUDE.md rule
    fails the no-quality-tax gate regardless of who would read it. The whole
    ``cache`` family (``cache`` / ``cache_thrash`` / ``cache-recommend``)
    reads the same ``persona`` to decide whether their cache_control
    instruction is shown at all — see ``_persona_gated_cache_fields``; unlike
    script/reuse there is no write to gate, only the instruction text.
    ``downsize`` reads the same ``persona`` to swap its CTA for the
    claude-code-actionable levers (``tj route export`` / ``tj optimize
    subagent`` / ``/compact``) instead of a raw model-swap instruction that
    persona can't act on — see ``_DOWNSIZE_CC_LEVER``. ``placement`` reads it
    to suppress the card outright for a ``"claude-code"`` window: the Batch
    API is a lever in the user's own application code, which an interactive
    session doesn't have. A report built without a
    ``persona`` field (e.g. hand-constructed in a test) defaults to
    ``"unknown"``, which keeps the script/reuse write off (that helper's
    conservative default) while leaving the cache family's instruction on
    (that helper's conservative default runs the other way — see its own
    docstring) — neither ever assumes ``"claude-code"``.

    ``window_days`` (default 30 — ``DEFAULT_COST_WINDOW_DAYS``, matching the
    daemon's own default look-back) is used for exactly ONE thing: sizing the
    write budget's per-month standing-cost comparison
    (``_write_budget_basis``). It never rescales a proposal's figure. Every
    card's ``past_overspend_usd``/``_tokens`` is its analyzer's own
    observation of THIS window, carried through unscaled — there is no
    projection step here any more (see the block comment above
    ``_write_budget_basis``).
    """
    findings = getattr(report, "findings", {}) or {}
    persona = str(getattr(report, "persona", "") or "unknown")
    proposals: list[CostProposal] = []

    # Second half of the persona skip gate. `build_report` already refuses to
    # RUN an analyzer with no fix for this persona, so on the live path these
    # findings are absent anyway; `_pick` makes the inbox correct by its own
    # construction rather than by trusting how the report was built — a report
    # assembled elsewhere (a cached payload, a wider selection, a test) can
    # still carry the finding, and a card the user cannot apply must not
    # appear in Review either way. A `None` finding contributes nothing.
    disabled = _disabled_analyzers(persona)

    def _pick(name: str) -> Any:
        return None if name in disabled else findings.get(name)

    adapters = (
        (
            lambda f: _downsize_to_proposal(f, config, persona=persona),
            getattr(report, "downgrade", None),
        ),
        (lambda f: _cache_to_proposals(f, persona=persona), _pick("cache")),
        (lambda f: _cache_uncached_to_proposals(f, persona=persona), _pick("cache")),
        (lambda f: _cache_thrash_to_proposals(f, persona=persona), _pick("cache")),
        (lambda f: _cache_lookback_to_proposals(f, persona=persona), _pick("cache")),
        (
            lambda f: _cache_recommend_to_proposals(f, _pick("cache"), persona=persona),
            _pick("cache-recommend"),
        ),
        (_trim_to_proposals, _pick("trim")),
        (lambda f: _subagent_to_proposals(f, config), _pick("subagent")),
        (
            lambda f: _placement_to_proposals(f, pricing_mode=pricing_mode, persona=persona),
            _pick("placement"),
        ),
        (_deadweight_to_proposals, _pick("deadweight")),
        (lambda f: _script_to_proposals(f, persona=persona), _pick("script")),
        (lambda f: _reuse_to_proposals(f, persona=persona), _pick("reuse")),
        (lambda f: _verbosity_to_proposals(f, persona=persona), _pick("verbosity")),
        (
            # The resend card is compound: it names the concrete over-powered
            # subagent (from the `subagent` finding) that the offload rule
            # should also right-size, so the two levers land as one card rather
            # than two. Reading the sibling finding here, not in the analyzer,
            # keeps `context_resend.py` free of a cross-analyzer dependency.
            lambda f: _resend_to_proposals(
                f, persona=persona, subagent_finding=_pick("subagent"), config=config,
            ),
            _pick("resend"),
        ),
        (_summarize_to_proposals, _pick("summarize")),
        (_relearn_to_proposals, _pick("relearn")),
    )
    for adapter, finding in adapters:
        try:
            proposals.extend(adapter(finding))
        except Exception:
            continue
    proposals = _apply_write_budget(proposals, report, window_days)
    # Order matters: the write budget can NET a proposal's figure down against
    # what its rule costs to keep, and the past-overspend basis stamp must
    # describe the netted figure, never the gross.
    return [_with_past_overspend(p) for p in proposals]


def _write_budget_basis(report: Any, window_days: float) -> Any:
    """The 30-day pace basis for this report's window — THE ONLY remaining
    consumer of pacing in the cost pipeline, and deliberately internal.

    ``write_budget`` prices a permanent CLAUDE.md rule against how many future
    sessions will re-send it, so it genuinely needs a per-month horizon: a rule
    written today is charged on every one of those sessions. That is a
    standing-cost comparison, not a claim about the user's spend, and nothing
    it produces reaches a payload — only the NETTED
    ``past_overspend_*`` figure and the ``standing_cost_*`` fields do.

    The per-analyzer forward projection this basis used to ALSO feed
    (``estimated_monthly_*`` via ``compute_projection_ratio``) is gone; do not
    route a past-tense figure back through here. See the field contract in the
    repo ``CLAUDE.md``.

    Reads ``active_days``/``sessions`` off the window summary the runner already
    computed. A hand-constructed report (a test, an older cached one) carries
    neither, which resolves to an unprojected basis with a zero session count:
    the netting then charges a rule nothing and only the quality floor and the
    write cap apply. Degrading toward "claim no standing cost" is the only safe
    direction, since the alternative would invent one.
    """
    from tokenjam.core.optimize.projection import build_projection_basis

    window = getattr(report, "window", None)
    return build_projection_basis(
        float(getattr(window, "days", 0.0) or window_days or 0.0),
        int(getattr(window, "active_days", 0) or 0),
        int(getattr(window, "sessions", 0) or 0),
    )


def _apply_write_budget(
    proposals: list[CostProposal], report: Any, window_days: float,
) -> list[CostProposal]:
    """Net every write-bearing card against what its rule costs to KEEP, and
    bound how many permanent rules the window may offer.

    Only cards that actually write something enter the budget: ``apply_capable``
    with a rung-1/rung-2 ``proposed_fix``. Everything else (the advise-only
    majority, the model-id swaps, the MCP-server removals) writes no standing
    prompt text and passes through with its figures untouched.

    Candidates are grouped by ``(analyzer, rung)`` because that is genuinely one
    block: every ``reuse`` cluster writes the same skeleton note, every
    ``script`` cluster the same script note. Nine clusters used to mean nine
    identical appended blocks; now the family's largest carries the write and
    its siblings say they are covered by it. A suppressed write degrades the
    same way the persona gate already degrades one: advise-only, with the
    identical text still carried as a copyable ``suggestion``.
    """
    from tokenjam.core.optimize import write_budget as wb

    basis = _write_budget_basis(report, window_days)
    findings = getattr(report, "findings", {}) or {}
    budget = wb.build_write_budget(
        lane_budget_tokens=wb.COST_WRITE_BUDGET_TOKENS,
        lane_max_writes=wb.COST_MAX_OFFERED_WRITES,
        # The summarize analyzer's own measurement of the files these rules
        # append to. `summarize` IS a COST_ANALYZER (see that tuple's own
        # docstring), so a normal recompute carries the finding and this
        # resolves to a real footprint. The absent case is the exception, not
        # the rule: a report built without summarize (a scoped `tj optimize
        # <analyzer>` run, or an older cache) leaves the lane cap standing
        # alone.
        existing_agent_file_tokens=wb.measured_agent_file_tokens(
            findings.get("summarize"),
        ),
    )

    candidates = [
        wb.WriteCandidate(
            key=p.signature,
            family=f"{p.analyzer}:rung{p.rung}",
            rung=p.rung,
            artifact_text=p.proposed_fix,
            gross_tokens=int(p.past_overspend_tokens or 0),
            gross_usd=p.past_overspend_usd,
        )
        for p in proposals
        if p.apply_capable and p.rung >= 1 and p.proposed_fix
    ]
    if not candidates:
        return proposals
    decisions = wb.allocate_writes(candidates, budget, basis)

    out: list[CostProposal] = []
    for p in proposals:
        decision = decisions.get(p.signature)
        if decision is None or not (p.apply_capable and p.rung >= 1 and p.proposed_fix):
            out.append(p)
            continue
        updates: dict[str, Any] = {
            "gross_recoverable_usd": p.past_overspend_usd,
            "gross_recoverable_tokens": p.past_overspend_tokens,
            "past_overspend_tokens": (
                decision.claimed_tokens if p.past_overspend_tokens is not None else None
            ),
            "past_overspend_usd": decision.claimed_usd,
            "standing_cost_tokens_per_session": decision.standing_tokens_per_session,
            "standing_cost_tokens": decision.standing_tokens,
            "standing_cost_usd": decision.standing_usd,
            "standing_cost_basis": decision.basis,
            "payback_ratio": decision.payback_ratio,
            "net_negative": decision.net_negative,
            "write_offered": decision.offered,
            "write_blocked_reason": decision.reason,
        }
        if not decision.offered:
            updates.update(
                apply_capable=False, advise_only=True, rung=0, scope="",
                proposed_fix="", suggestion=p.suggestion or p.proposed_fix,
                apply_blocked_reason=decision.reason,
            )
        out.append(replace(p, **updates))
    return out


#: Suffix appended to every stamped ``past_overspend_basis`` so the figure can
#: never be read off a payload without the tense being stated with it.
PAST_OVERSPEND_OBSERVED_NOTE = (
    "Observed over the analyzed window: what this behaviour already cost, "
    "priced at the rates it actually billed at. Not a projection and not a "
    "claim about what a fix returns."
)

#: Same, for the second (total observed cost) number. Cost wording only — the
#: whole point of separating it from the figure above is that this one includes
#: spend nobody has shown to be avoidable, so calling it waste would be a claim
#: the data does not support.
OBSERVED_COST_NOTE = (
    "Also observed over the same window: what this behaviour cost in TOTAL, "
    "including the part that was never shown to be avoidable. This is cost, "
    "not waste, and not a claim about what a fix returns. It is not summed "
    "with the figure above — the avoidable figure is a subset of it."
)


def _with_past_overspend(proposal: CostProposal) -> CostProposal:
    """Stamp the tense-bearing BASIS strings onto a finished proposal:
    ``past_overspend_basis`` for the canonical figure the adapter already set,
    and the ``observed_cost_*`` trio where a finding also measured the full
    cost of the behaviour.

    ONE place decides the wording for every analyzer: two surfaces reading two
    differently-worded "what did this cost me" figures is how the Dashboard
    hero and the Review inbox headline silently come to disagree.

    It no longer MOVES a number between fields. It used to copy
    ``estimated_recoverable_usd`` onto ``past_overspend_usd``, which is what
    left two names live for one quantity — the ambiguity this collapse removed.
    The adapter sets ``past_overspend_usd``/``_tokens`` directly, from its
    analyzer's field of the same name, and nothing rewrites it here.

    **The headline is the AVOIDABLE figure for every analyzer**, because waste
    is only ever the avoidable portion. A finding that also computes the full
    observed cost of the behaviour (``cost_of_waste_usd`` — today only
    ``resend`` and ``relearn``) carries it on ``observed_cost_*`` as an
    explicitly-cost second number, together with the analyzer's
    ``coverage_note`` stating how the two figures' populations differ. This
    used to be inverted, which made the card assert that the ~94% gap between
    them was unavoidable; it never was, it was merely un-analysed. Every other
    analyzer produces one quantity, so the second number stays ``None`` and the
    card shows a single figure rather than saying the same thing twice.

    Never applies a projection ratio — there is none left to apply. Never
    mutates: returns a new proposal.
    """
    has_avoidable = (
        proposal.past_overspend_usd is not None
        or proposal.past_overspend_tokens is not None
    )
    cost_usd = proposal.cost_of_waste_usd
    cost_tokens = proposal.cost_of_waste_tokens
    has_cost = cost_usd is not None or bool(cost_tokens)
    if not has_avoidable and not has_cost:
        # Nothing observed at all — e.g. relearn's window produced no priced
        # figure of either kind. Neither field would have anything to carry.
        return proposal

    stamped = proposal
    if has_avoidable:
        stamped = replace(
            stamped,
            past_overspend_basis=" ".join(
                x for x in (proposal.estimate_basis, PAST_OVERSPEND_OBSERVED_NOTE) if x
            ),
        )
    if has_cost:
        # A cost-only card (no avoidable claim at all — e.g. relearn, whose
        # forward claim is deliberately omitted per CLAUDE.md rule 27) must
        # still get its observed cost stamped here; it cannot ride along on
        # the `has_avoidable` branch above, which it never enters.
        stamped = replace(
            stamped,
            observed_cost_usd=cost_usd,
            observed_cost_tokens=cost_tokens or None,
            observed_cost_basis=" ".join(
                x for x in (proposal.cost_of_waste_basis, OBSERVED_COST_NOTE) if x
            ),
        )
    return stamped


#: The pre-collapse names a cached cost-proposal dict may still carry, mapped
#: onto the one canonical field. ``estimated_recoverable_*`` was the SAME
#: quantity under a forward-framed name, so it migrates straight across.
#: ``estimated_monthly_*`` deliberately does NOT appear here: it was that
#: quantity multiplied by a pace ratio, and reviving a paced figure as the
#: past-tense headline is exactly the mistake this collapse exists to prevent —
#: a legacy entry carrying only the monthly key renders no dollar figure until
#: the next recompute (at most 6h), which is the honest degradation.
_LEGACY_PROPOSAL_FIELD_ALIASES = (
    ("past_overspend_usd", "estimated_recoverable_usd"),
    ("past_overspend_tokens", "estimated_recoverable_tokens"),
)


def backfill_legacy_past_overspend_fields(proposal: dict[str, Any]) -> dict[str, Any]:
    """Read-time backward compat for a cost-proposal dict cached before the
    per-analyzer dollar fields were collapsed onto ``past_overspend_*``.

    A cache written by the CURRENT build always carries them (the adapters set
    them and ``_with_past_overspend`` stamps the basis), so this only fires for
    an entry that predates the collapse — which would otherwise render an em
    dash where the page's headline number belongs until the next scheduled
    recompute, up to 6h away. Renames the legacy keys, applies the SAME basis
    derivation ``_with_past_overspend`` does, and never invents a figure: a
    legacy entry with no observed figure at all stays empty.
    """
    if "past_overspend_basis" in proposal:
        return proposal
    stamped = {**proposal}
    for canonical, legacy in _LEGACY_PROPOSAL_FIELD_ALIASES:
        # Always PRESENT, even when there is nothing to carry: a renderer that
        # indexes the canonical key must not blow up on a legacy entry, and an
        # explicit `None` reads as "not measured" rather than "worth nothing".
        stamped[canonical] = (
            stamped.get(canonical)
            if stamped.get(canonical) is not None
            else proposal.get(legacy)
        )
        stamped.pop(legacy, None)
    stamped["past_overspend_basis"] = " ".join(
        x for x in (proposal.get("estimate_basis") or "", PAST_OVERSPEND_OBSERVED_NOTE) if x
    )
    # The retired paced fields never render again, whatever a stale cache says.
    stamped.pop("estimated_monthly_usd", None)
    stamped.pop("estimated_monthly_tokens", None)
    cost_usd = proposal.get("cost_of_waste_usd")
    cost_tokens = proposal.get("cost_of_waste_tokens")
    if cost_usd is None and not cost_tokens:
        return stamped
    return {
        **stamped,
        "observed_cost_usd": cost_usd,
        "observed_cost_tokens": cost_tokens or None,
        "observed_cost_basis": " ".join(
            x for x in (proposal.get("cost_of_waste_basis") or "", OBSERVED_COST_NOTE) if x
        ),
    }


# --------------------------------------------------------------------------- #
# THE rollup — the one aggregate every surface reads. There is no second one.
#
# There used to be two (`estimated_recoverable_rollup` beside this), summing
# two per-analyzer fields that had become the same quantity under two names,
# plus a third paced figure derived from them. A driver session comparing them
# reported "identical for 6 of 7 analyzers" while the Review inbox rendered the
# paced third value, 7.14% larger. Both extra fields and the second rollup are
# gone; if you find yourself adding a rollup here, you are re-creating that bug.
# --------------------------------------------------------------------------- #

def past_overspend_rollup(
    proposals: list[Any],
    *,
    window_days: int = DEFAULT_COST_WINDOW_DAYS,
    excluded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sum ``past_overspend_usd``/``past_overspend_tokens`` across
    ``proposals``, deduplicated by ``signature`` (a proposal's stable identity —
    see the ``CostProposal`` docstring) so a stale or duplicate cache entry is
    never double-counted.

    Generic over ``analyzer``: reads only the shared ``CostProposal`` fields, so
    a new analyzer's cards are picked up automatically with no change here (the
    dedup rule for overlapping CLAIMS lives one layer down, in each analyzer's
    adapter — see CLAUDE.md rule 27).

    This is the AVOIDABLE portion of what the flagged behaviours already cost
    over the analyzed window, every token priced at the rate it genuinely
    billed at. Waste is only ever what could have been avoided, so this — not
    the larger total-cost figure — is the number any surface using
    waste/overspend wording may show. Nothing here is projected, discounted, or
    claimed:

    * **No pacing.** There is no ratio in this module any more, and this
      function does not accept ``active_days``/``n_sessions``, so there is
      nothing to project FROM. The whole point of the figure is that it is an
      observation of a window that already happened.
    * **Only a proposal carrying a numeric figure contributes** — to the sum
      AND to ``proposal_count``. A card with no figure still renders
      individually in the inbox; counting it here without a contribution would
      silently understate the average the headline implies.
    * The token sum is counted INDEPENDENTLY of the dollar sum (a different,
      often overlapping, never identical set of proposals carries each). A
      renderer leading with the token figure must quote
      ``token_proposal_count`` against ``deduplicated_proposal_count`` and say
      the figure is a floor, not a total.

    ``observed_cost_usd`` (the second, LARGER number ``resend`` and ``relearn``
    carry: the full cost of the behaviour including the part nobody has shown
    to be avoidable) IS summed, into its own separate ``observed_cost_usd`` key
    and per-analyzer entry — never into ``past_overspend_usd``. The two totals
    overlap by construction (the avoidable figure is a subset of the cost), so
    adding them would double-count, and swapping them would relabel unanalysed
    cost as waste. A renderer shows the avoidable total as the headline and the
    cost total as an explicitly-cost second line.

    ``excluded`` is a passthrough for waste a caller decided NOT to sum in as a
    peer proposal (an analyzer with its own review surface and no representable
    inbox card). Never summed into any total here; carried on the result
    unchanged so a caller can render "$X more available in <analyzer> → review
    it" instead of silently omitting a real figure. ``None`` becomes ``{}`` —
    "nothing known to be excluded", not "excluded total is zero".
    """
    seen: dict[str, dict[str, Any]] = {}
    for p in proposals:
        row = asdict(p) if is_dataclass(p) and not isinstance(p, type) else dict(p)
        sig = str(row.get("signature") or "")
        if not sig or sig in seen:
            continue
        seen[sig] = row

    total_usd = 0.0
    total_tokens = 0
    total_observed_cost_usd = 0.0
    observed_cost_count = 0
    usd_count = 0
    token_count = 0
    by_analyzer: dict[str, dict[str, Any]] = {}
    for row in seen.values():
        analyzer = str(row.get("analyzer") or "unknown")
        # An analyzer earns a breakdown entry only by CONTRIBUTING a figure.
        # A card carrying nothing still renders individually in the inbox, but
        # listing it here (and in the "contributing analyzers:" basis string)
        # would name it as a contributor to a total it added nothing to.
        if (
            row.get("past_overspend_usd") is None
            and row.get("past_overspend_tokens") is None
            and row.get("observed_cost_usd") is None
        ):
            continue
        entry = by_analyzer.setdefault(
            analyzer,
            {"analyzer": analyzer, "count": 0, "usd": 0.0, "tokens": 0,
             "observed_cost_usd": None},
        )
        entry["count"] += 1

        tokens = row.get("past_overspend_tokens")
        if tokens is not None:
            total_tokens += int(tokens)
            token_count += 1
            entry["tokens"] = int(entry["tokens"]) + int(tokens)

        observed_cost = row.get("observed_cost_usd")
        if observed_cost is not None:
            total_observed_cost_usd += float(observed_cost)
            observed_cost_count += 1
            entry["observed_cost_usd"] = round(
                (entry["observed_cost_usd"] or 0.0) + float(observed_cost), 6
            )

        usd = row.get("past_overspend_usd")
        if usd is None:
            continue
        usd = float(usd)
        total_usd += usd
        usd_count += 1
        entry["usd"] = round(float(entry["usd"]) + usd, 6)

    deduplicated_proposal_count = len(seen)
    if usd_count == 0:
        basis = (
            f"no open (not yet applied) cost proposal currently carries an "
            f"observed dollar figure for the last {window_days} days."
        )
    else:
        breakdown = "; ".join(
            f"{a['analyzer']} ({a['count']})"
            for a in sorted(by_analyzer.values(), key=lambda x: x["analyzer"])
        )
        basis = (
            f"sum of the AVOIDABLE window figure (past_overspend_usd) across "
            f"{usd_count} of {deduplicated_proposal_count} open (not yet "
            f"applied), deduplicated-by-signature cost proposal(s), observed "
            f"over the last {window_days} days; contributing analyzers: "
            f"{breakdown}."
        )
    if token_count:
        basis += (
            f" Token figure: sum of past_overspend_tokens across "
            f"{token_count} of {deduplicated_proposal_count} proposal(s); the "
            f"rest carry no token figure, so it is a floor, not a total."
        )
    if observed_cost_count:
        basis += (
            f" Separately, {observed_cost_count} proposal(s) also report a "
            f"total observed COST of the behaviour they flag, summed into "
            f"observed_cost_usd. That total includes spend that was never "
            f"shown to be avoidable, so it is never added to the figure above "
            f"and is never labelled waste or overspend."
        )
    return {
        "past_overspend_usd": round(total_usd, 6),
        "past_overspend_tokens": total_tokens,
        # The total observed cost of the same behaviours, kept as a distinct
        # key so no renderer can confuse it for the avoidable headline and no
        # caller can accidentally sum the two overlapping quantities.
        "observed_cost_usd": (
            round(total_observed_cost_usd, 6) if observed_cost_count else None
        ),
        "observed_cost_proposal_count": observed_cost_count,
        "proposal_count": usd_count,
        "token_proposal_count": token_count,
        "deduplicated_proposal_count": deduplicated_proposal_count,
        "window_days": window_days,
        "by_analyzer": sorted(by_analyzer.values(), key=lambda x: x["analyzer"]),
        "excluded": excluded or {},
        "basis": basis,
        "disclosure": PAST_OVERSPEND_OBSERVED_NOTE,
        "cost_disclosure": OBSERVED_COST_NOTE if observed_cost_count else "",
    }
