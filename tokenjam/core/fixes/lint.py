"""Properties every fix text must hold, checked mechanically.

The reason this is a lint and not a review checklist: the defects it catches
already shipped, twice each, past readers who were looking. The identical
sizing contradiction lived in two constants at once; a rubric shipped an escape
hatch the agent grades itself against; a card shipped a fix whose own text said
to do nothing. None of those are subtle once stated — they are just invisible
when the policy has no single home to be checked in.

Four of the five properties come from what Anthropic publishes about
instruction text (keep it short, keep it concrete, do not contradict yourself,
escalate to a hook when it must run at a fixed point). The fifth is ours and is
the one that matters most here:

    **A fix may not re-license the behaviour its own analyzer bills for.**

`past_overspend_usd` is the maximum its analyzer knows: applying the fix should
ERASE the number it found. A fix that gives a pass to the exact shape the
analyzer flagged leaves the number where it was, which is a correctness bug
dressed as copy. That is the whole D-class, and it is mechanical: the record
names the shapes it must not excuse, and this checks the text against them.
"""
from __future__ import annotations

import re

from tokenjam.core.fixes.catalog import FIX_CATALOG, FixRecord

#: Anthropic's published guidance for instruction files. A rule longer than
#: this competes with everything after it for attention.
MAX_FIX_LINES = 200

#: Below this a "fix" is a label, not an instruction an agent can follow. Same
#: floor the write budget's quality gate applies.
MIN_FIX_CHARS = 40

#: An escape hatch the agent grades ITSELF against. "Unless the subtask
#: genuinely needs deep reasoning" asks the dispatching agent to rate its own
#: task's difficulty, and an agent asked that question answers yes — so the
#: exception swallows the rule. An exception has to be stated in terms an
#: outside reader could check (a condition present in the dispatch, a stated
#: fact), never in terms of the agent's own judgement of difficulty or need.
_SELF_GRADED = re.compile(
    r"\b(?:unless|except\s+when|only\s+when|if)\b[^.]{0,80}\b"
    r"(?:genuinely|truly|really|actually)\s+(?:needs?|requires?|warrants?)",
    re.IGNORECASE,
)
#: The same trap without an intensifier: "when the task needs deep reasoning".
_SELF_GRADED_JUDGEMENT = re.compile(
    r"\b(?:unless|except\s+when|only\s+when)\b[^.]{0,80}\b"
    r"(?:needs?|requires?)\s+(?:deep|serious|real|careful)\s+"
    r"(?:reasoning|thought|thinking|judgement|judgment)",
    re.IGNORECASE,
)

#: A fix whose own text says no action is required. Legitimate as an
#: OBSERVATION, never as an offered write — a card whose fix says to do nothing
#: must not occupy an apply slot. The record has to admit it via
#: ``advisory_only`` so every surface can gate on a field instead of re-reading
#: the prose.
#:
#: The alternation tolerates a LIST of nouns ("no rule or hook is needed"),
#: which is how the sentence is naturally written. The first draft matched only
#: a single noun and reported a correctly-marked advisory record as
#: mis-marked — a false positive that would have taught the next author to
#: reach for an exception rather than to fix the text.
_NO_ACTION_NOUN = r"(?:hook|rule|action|fix|change|edit)"
_SAYS_NO_ACTION = re.compile(
    rf"no\s+{_NO_ACTION_NOUN}(?:\s*(?:,|/|or|and)\s*{_NO_ACTION_NOUN})*\s+"
    r"(?:is\s+|are\s+)?(?:needed|required)"
    r"|advisory\s+awareness\s+only"
    r"|there\s+is\s+no\s+change\s+to\s+make"
    r"|nothing\s+to\s+do\s+here",
    re.IGNORECASE,
)

#: Text that must run at a fixed point in the loop — a checkpoint, every time,
#: before a specific tool — is a hook's job. Delivered as a rule it is a
#: request the agent may or may not honour at that instant.
_FIXED_POINT = re.compile(
    r"\b(?:before every|after every|on every (?:tool|command|bash)"
    r"|block the (?:tool|command)|refuse to run)\b",
    re.IGNORECASE,
)


def lint_fix(record: FixRecord) -> list[str]:
    """Every property ``record`` violates, as human-readable strings.

    An empty list is a pass. Returns all violations rather than the first, so a
    fix is corrected in one pass instead of being re-run per complaint.
    """
    text = record.text or ""
    problems: list[str] = []

    if len(text.strip()) < MIN_FIX_CHARS:
        problems.append(
            f"too short to be an instruction ({len(text.strip())} chars, "
            f"minimum {MIN_FIX_CHARS}) — this is a label, not a fix",
        )
    lines = text.splitlines()
    if len(lines) > MAX_FIX_LINES:
        problems.append(
            f"{len(lines)} lines exceeds the {MAX_FIX_LINES}-line ceiling for "
            "instruction text — a rule this long competes with everything "
            "after it",
        )
    if _SELF_GRADED.search(text) or _SELF_GRADED_JUDGEMENT.search(text):
        problems.append(
            "contains an escape hatch the agent grades itself against "
            "(\"unless it genuinely needs ...\"): an agent asked to rate its "
            "own task's difficulty answers yes, so the exception swallows the "
            "rule. State the exception as a condition an outside reader can "
            "check.",
        )
    # THE load-bearing check. See the module docstring.
    for shape in sorted(record.must_not_relicense):
        if shape and shape.lower() in text.lower():
            problems.append(
                f"re-licenses the behaviour its analyzer bills for: the text "
                f"contains {shape!r}, which gives a pass to the shape "
                f"{record.key!r} is written to route away from. Applying this "
                "fix would leave the number that produced it where it was.",
            )
    says_no_action = bool(_SAYS_NO_ACTION.search(text))
    if says_no_action and not record.advisory_only:
        problems.append(
            "the text says no action is needed, but the record is not marked "
            "advisory_only — a fix that says to do nothing must never be "
            "OFFERED as a write, though its observation still stands.",
        )
    if record.advisory_only and not says_no_action:
        problems.append(
            "marked advisory_only but the text reads as an actionable "
            "instruction — say plainly that no action is required, or offer "
            "the action.",
        )
    if _FIXED_POINT.search(text) and record.delivery == "claude_md_rule":
        problems.append(
            "asks for behaviour at a FIXED POINT in the loop but is delivered "
            "as an instruction-file rule, which is read once at session start "
            "and honoured at the agent's discretion. Escalate to a hook.",
        )
    return problems


def lint_catalog() -> dict[str, list[str]]:
    """Every catalogued fix's violations, keyed by fix key. Empty dict = clean."""
    out = {}
    for key, record in FIX_CATALOG.items():
        problems = lint_fix(record)
        if problems:
            out[key] = problems
    return out


__all__ = ["MAX_FIX_LINES", "MIN_FIX_CHARS", "lint_catalog", "lint_fix"]
