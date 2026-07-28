"""The catalogued fixes themselves.

Split from ``catalog.py`` so the data sits apart from the machinery, and
imported for its side effect the way ``core/optimize/analyzers`` is. Every
entry names the observation it answers and the shapes it may not excuse, which
is what lets ``lint.py`` check the property that matters: applying the fix must
be able to ERASE the number its analyzer found.
"""
from __future__ import annotations

from tokenjam.core.fixes.catalog import (
    LEVER_AWARENESS,
    LEVER_EFFORT,
    LEVER_MODEL,
    LEVER_OFFLOAD,
    LEVER_ROUTING,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    register,
)

#: The shapes the subagent/resend family must never give a pass to. These are
#: exactly what the analyzer bills for: the `output_tokens < 2000` /
#: `tool_calls <= 5` gate was DELETED from `subagent_rightsizing` because on a
#: real Claude Code corpus it excluded the expensive dispatches, so a fix that
#: re-imposes it in prose undoes the gate fix in the one place the user reads.
_SIZING_RELICENSE = frozenset({
    "little tool work",
    "few tool calls",
    "short result rarely needs",
    "short conclusion rarely needs",
    "rarely needs the premium tier",
})

SUBAGENT_RUBRIC = register(FixRecord(
    key="subagent.sizing_rubric",
    text=(
        "Right-size Task-dispatched subagents: default every subagent to the "
        "cheapest same-family model that fits its shape, and treat that "
        "default as the answer unless the dispatch itself states one of these "
        "conditions: the subtask IS the architecture or design decision this "
        "session exists to make; it must reconcile sources that disagree into "
        "one judgement a later step cannot re-derive; or a cheaper model "
        "already attempted it in this session and its output was rejected. "
        "How much tool work a subagent does and how long its result runs are "
        "not on that list: a broad, tool-heavy, long-output dispatch is the "
        "expensive one, not the hard one, and it is the one this rule exists "
        "to route down."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="premium-tier models running Task dispatches that a cheaper same-family model fits",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
))

RIGHTSIZE_TEMPLATE = register(FixRecord(
    key="resend.rightsize_worker",
    text=(
        "Then right-size what you offload to. Default the worker to the "
        "cheapest same-family model that fits its shape, and pin both that "
        "model and its reasoning effort in its own definition file so every "
        "future dispatch inherits them instead of defaulting to whatever the "
        "parent runs on. How much it reads and how long its conclusion runs "
        "are not reasons to keep the premium tier; they are what makes the "
        "dispatch expensive in the first place."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"resend"}),
    answers="context-heavy work run inline on a premium model instead of in a right-sized worker",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
))

#: The CREATE fix. A Claude Code dispatch is almost always a BUILT-IN
#: (`general-purpose`, `Explore`, `Plan`, `fork`), which has no definition file
#: — and the product used to read "no file" as "no fix", falling through to
#: prose. The docs are explicit that a user or project subagent of the same
#: name OVERRIDES the built-in and keeps its own `model` field, so the file can
#: be created. The waste's origin belongs in the same breath: `model` defaults
#: to `inherit`, so an opus-driven session dispatches opus workers unless
#: pinned. Nobody decided that; it is an unset default.
SUBAGENT_DEFINE_BUILTIN = register(FixRecord(
    key="subagent.define_builtin_override",
    text=(
        "Pin the model for the built-in subagents this session dispatches. A "
        "subagent's `model` defaults to `inherit`, so an Opus-driven session "
        "hands every Task dispatch an Opus worker unless something says "
        "otherwise — that is an unset default, not a decision. A user or "
        "project subagent defined with the same name as a built-in overrides "
        "it and keeps its own `model`, so creating the definition file is the "
        "fix: give it the cheapest same-family model that fits the work it "
        "actually does."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="built-in subagents inheriting the driver's premium model because no definition file pins one",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
))

SUBAGENT_EFFORT = register(FixRecord(
    key="subagent.pin_effort",
    text=(
        "Pin a subagent's effort alongside its model. They answer different "
        "diagnoses and neither substitutes for the other: a worker that did "
        "not know enough needs a larger model, a worker that did not try hard "
        "enough needs more effort. Both are frontmatter fields on the "
        "subagent's own definition file, so a dispatch inherits them instead "
        "of taking the parent's."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="subagents inheriting the parent's effort setting rather than one sized to their task",
    lever=LEVER_EFFORT,
))

OFFLOAD_RULE = register(FixRecord(
    key="resend.offload_to_subagent",
    text=(
        "Offload context-heavy sub-tasks (broad file reads, multi-file "
        "search, long tool-output loops, exploratory investigation) to a "
        "subagent instead of running them inline in the main thread. A "
        "subagent's own tool logs and intermediate output stay in its own "
        "context; only its short conclusion returns to the caller, so the "
        "material that keeps getting re-sent turn over turn never accumulates "
        "on the main thread to begin with."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"resend", "downsize"}),
    answers="context re-sent every turn because the work that produced it stayed in the main thread",
    lever=LEVER_OFFLOAD,
))

#: The SDK counterpart of the same observation. A Claude Code user cannot
#: paste this and an SDK caller cannot use the offload rule above; handing
#: either the other's fix names an action they cannot take.
SDK_CACHE_CONTROL = register(FixRecord(
    key="resend.sdk_cache_breakpoint",
    text=(
        "Adopt a cache_control breakpoint at the call site so the repeated "
        "prefix bills at the cache rate instead of full price on every turn, "
        "and trim the request you build rather than relying on the harness to "
        "do it — an SDK caller constructs the whole prompt, so the repeated "
        "context is yours to bound."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    analyzers=frozenset({"resend"}),
    answers="repeated prompt prefix billed at full input rate on every turn",
    lever=LEVER_ROUTING,
))

#: An OBSERVATION, not an offer. The harness already errors clearly and agents
#: self-correct next turn, so there is no action to sell — but the recurrence
#: genuinely cost something, and that figure stands (Critical Rule 32).
EDIT_BEFORE_READ = register(FixRecord(
    key="relearn.edit_before_read",
    text=(
        "The harness already blocks an Edit/Write before a Read with a clear "
        "error and agents reliably self-correct by reading on the next turn, "
        "so no rule or hook is needed here. This is reported as awareness "
        "only: what the retries already cost is real and is shown, but there "
        "is no change to make."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Edit/Write attempted before Read, retried after the harness error",
    lever=LEVER_AWARENESS,
    advisory_only=True,
))


__all__ = [
    "EDIT_BEFORE_READ",
    "OFFLOAD_RULE",
    "RIGHTSIZE_TEMPLATE",
    "SDK_CACHE_CONTROL",
    "SUBAGENT_DEFINE_BUILTIN",
    "SUBAGENT_EFFORT",
    "SUBAGENT_RUBRIC",
]
