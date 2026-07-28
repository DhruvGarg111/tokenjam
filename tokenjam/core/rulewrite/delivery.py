"""HOW a rule reaches the agent — the delivery mechanism, as a seam.

Appending markdown to a ``CLAUDE.md`` is one way to put guidance in front of an
agent. It is not the only one, and for several of these analyzers it is not the
best one: a rule at the top of a long context competes with everything after
it, while a Claude Code hook can put the same guidance in view **at the moment
of the decision** it is meant to change.

So delivery is a dimension of the fix, not a hardcoded behaviour. A rule
carries WHAT to say (``artifact_text``), WHERE it lands (its destinations) and
HOW it gets there (this module). Adding a second mechanism should be a new
:class:`DeliveryKind` registered here plus its renderer — not a refactor of the
staging, diff, apply, undo or budget machinery, none of which knows what a
CLAUDE.md is.

**The pricing trap this seam exists to keep visible.** ``write_budget``'s rung
ladder charges rung 3+ (hooks, wrappers, config) ZERO standing cost, on the
reasoning that a settings file is *executed*, never sent as prompt text. That
is correct for an executing hook — a formatter, a lint gate, a guard that
blocks a command — and it is **wrong for a context-injecting one**. A
``UserPromptSubmit`` re-injection and a ``PreToolUse`` ``additionalContext``
nudge are prompt text; they merely arrive on a different schedule than a file
read at session start. Worse, once injected they land in the conversation and
are re-sent with every subsequent turn, so their cost is not even bounded by
the number of injections.

That is why :attr:`DeliveryKind.carries_prompt_text` is a property of the
DELIVERY and not of the rung. A mechanism that puts tokens in front of the
model pays a standing cost whatever ladder rung it happens to occupy, and a
mechanism that does not, does not. Read the flag; do not infer the answer from
the rung.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tokenjam.core.rulewrite.types import RUNG_SKILL, RuleWrite, RuleWriteRefused

#: The mechanism shipped today: a markdown block appended to a ``CLAUDE.md``
#: (or a ``SKILL.md`` at rung 2), loaded by the harness at session start.
DELIVERY_CLAUDE_MD_RULE = "claude_md_rule"

#: The default for a rule that names no mechanism — every rule written before
#: delivery was a field, and every rule the four current analyzers produce.
DEFAULT_DELIVERY = DELIVERY_CLAUDE_MD_RULE


@dataclass(frozen=True)
class DeliveryKind:
    """One way of getting a rule in front of the agent.

    ``render`` takes ``(rule, existing_content)`` and returns the FULL new
    content of the target. Returning the whole artifact rather than a patch is
    what lets staging persist output instead of a recipe, so what a reviewer
    approved in the diff is byte-for-byte what apply writes — a property every
    delivery kind has to preserve.

    ``standing_tokens`` takes ``(rule, rendered, existing)`` and returns the
    per-session tokens this delivery adds. It exists so a mechanism can price
    itself rather than inherit the rung ladder's answer; see the module
    docstring for why that ladder cannot be trusted for an injecting hook.
    """

    name: str
    #: Short human label, for a CLI column and a UI badge.
    label: str
    #: Whether this mechanism puts tokens in front of the model at all. The
    #: budget's netting is meaningless for a mechanism that does not, and
    #: dangerously wrong for one that does but is assumed not to.
    carries_prompt_text: bool
    render: Callable[[RuleWrite, str], str]
    standing_tokens: Callable[[RuleWrite, str, str], int]


def _render_claude_md(rule: RuleWrite, existing: str) -> str:
    """Delegates to ``relearn_apply``'s renderers so the marker comments — the
    thing that makes a re-apply replace rather than duplicate, and makes the
    existing Revert path able to find the block — are produced in exactly one
    place."""
    from tokenjam.core.optimize.relearn_apply import (
        render_note_content,
        render_skill_content,
        slugify,
    )

    cluster = {
        "title": rule.title or rule.signature,
        "proposed_fix": rule.artifact_text,
        "rung": rule.rung,
        "sessions": sum(d.sessions for d in rule.destinations),
        "repos": [d.path for d in rule.destinations],
    }
    if rule.rung == RUNG_SKILL:
        return render_skill_content(cluster, rule.signature, slugify(rule.title))
    return render_note_content(existing, cluster, rule.signature)


def _standing_tokens_claude_md(rule: RuleWrite, rendered: str, existing: str) -> int:
    """Priced on the DELTA this write introduces, through
    ``write_budget.standing_tokens_per_session`` so the rung semantics are
    applied by the module that owns them rather than restated here.

    A re-apply that replaces an existing block adds nothing, and pricing it as
    if it added a whole block would overstate the cost of keeping a rule
    current.
    """
    from tokenjam.core.optimize.write_budget import standing_tokens_per_session

    added = max(0, len(rendered) - len(existing))
    return standing_tokens_per_session(rule.rung, rendered[:added] if added else "")


#: Every delivery mechanism this product can offer, by name. One today.
#:
#: Adding one means adding an entry here plus its two functions. It must NOT
#: mean touching ``apply.py``, ``store.py``, ``plan.py`` or ``write_budget`` —
#: if it does, the seam has been welded shut again and that is the thing to fix
#: rather than to work around.
DELIVERY_KINDS: dict[str, DeliveryKind] = {
    DELIVERY_CLAUDE_MD_RULE: DeliveryKind(
        name=DELIVERY_CLAUDE_MD_RULE,
        label="CLAUDE.md rule",
        carries_prompt_text=True,
        render=_render_claude_md,
        standing_tokens=_standing_tokens_claude_md,
    ),
}


def resolve_delivery(name: str | None) -> DeliveryKind:
    """The mechanism named by ``name``, defaulting to the markdown rule.

    An UNKNOWN name is refused rather than silently falling back: a staged
    entry written by a later build that names a mechanism this build cannot
    render must not be quietly rendered as a markdown block into whatever file
    it named. That is a wrong write to a real file, which is the one failure
    this package exists to make impossible.
    """
    key = str(name or DEFAULT_DELIVERY)
    kind = DELIVERY_KINDS.get(key)
    if kind is None:
        raise RuleWriteRefused(
            f"unknown delivery mechanism {key!r} — this build knows "
            f"{', '.join(sorted(DELIVERY_KINDS))}. Refusing to guess how to "
            "write it.",
        )
    return kind


__all__ = [
    "DEFAULT_DELIVERY",
    "DELIVERY_CLAUDE_MD_RULE",
    "DELIVERY_KINDS",
    "DeliveryKind",
    "resolve_delivery",
]
