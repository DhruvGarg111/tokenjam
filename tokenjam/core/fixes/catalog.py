"""One record per fix type — text, delivery, persona, lever, and who uses it.

Fix text used to live as hardcoded constants scattered across analyzer modules.
That is not a tidiness complaint; it is the structural cause of a whole class of
shipped defect, and the proof is on the record: the identical sizing-rule
contradiction shipped in BOTH the subagent rubric and the resend right-sizing
template, in different words, so fixing the reported one left the other live and
a user still got the wrong instruction from the other card. Two copies of a
policy is two policies. A catalog plus a lint makes that mechanically
detectable instead of review-detectable.

Three dimensions a fix carries that a bare string cannot:

* **delivery** — HOW it reaches the agent (``core/rulewrite/delivery``). A
  markdown rule and a hook deliver the same words on different schedules and
  at different costs.
* **persona** — WHICH user it is for, because what a user can edit differs
  completely. A Claude Code user edits instruction files, ``.claude/rules/``,
  agent files and hooks. An SDK service edits routing code and redeploys. The
  same observation therefore has two different fixes, and handing either one
  the other's is worse than saying nothing: it names an action they cannot
  take, which reads as the product not understanding their setup.
* **lever** — WHAT it changes. ``model`` and ``effort`` are separate,
  independently pinnable levers, and they answer different diagnoses: a worker
  that did not know enough needs a bigger model, a worker that did not try
  hard enough needs more effort. Collapsing them loses half the remedy.

The catalog is deliberately data, not behaviour. Rendering stays with the
delivery kind, gating stays with the persona map in ``runner``; this module
answers "what do we say, to whom, through what, about which observation" and
nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Personas a fix can target. Mirrors the vocabulary ``core/framing`` and
#: ``runner.PERSONA_DISABLED_ANALYZERS`` already use; ``ANY`` is for a fix that
#: genuinely applies to everyone rather than one nobody classified.
PERSONA_CLAUDE_CODE = "claude-code"
PERSONA_SDK = "sdk"
PERSONA_ANY = "any"

#: Levers a fix pulls. ``model`` and ``effort`` are BOTH pinnable per agent and
#: are not substitutes for each other (see the module docstring).
LEVER_MODEL = "model"
LEVER_EFFORT = "effort"
LEVER_OFFLOAD = "offload"
LEVER_ROUTING = "routing"
LEVER_AWARENESS = "awareness"


@dataclass(frozen=True)
class FixRecord:
    """One fix type, stated once.

    ``answers`` names the observation this fix is a remedy for. It is not
    decoration: the lint's load-bearing check is that a fix does not re-license
    the very behaviour its analyzer bills for, and that check needs to know
    what the analyzer is billing for.
    """

    key: str
    #: The instruction text, verbatim as it reaches the user.
    text: str
    #: Delivery kind name (see ``core/rulewrite/delivery.DELIVERY_KINDS``).
    delivery: str
    #: Personas this fix is FOR. A fix offered to a persona outside this set is
    #: naming an action that user cannot take.
    personas: frozenset[str]
    #: Analyzers that hand this fix out.
    analyzers: frozenset[str]
    #: The observation it answers, in one line.
    answers: str
    lever: str = LEVER_AWARENESS
    #: True when the fix's own text says no action is needed. Such a record may
    #: be LISTED (the observation is real) but never OFFERED as a write — a
    #: card whose fix text says to do nothing must not occupy an apply slot.
    advisory_only: bool = False
    #: Shapes this fix must never give a pass to, as lowercase substrings.
    #: The analyzer bills for these; a fix that excuses them cannot erase the
    #: number that produced it. Checked by the lint.
    must_not_relicense: frozenset[str] = field(default_factory=frozenset)

    def applies_to(self, persona: str) -> bool:
        return PERSONA_ANY in self.personas or persona in self.personas


def _record(**kw: object) -> FixRecord:
    return FixRecord(**kw)  # type: ignore[arg-type]


#: THE catalog. One entry per fix type.
#:
#: Entries here are the source of truth for their text. An analyzer that wants
#: to hand out a fix reads it from here rather than defining a constant of its
#: own — that is the whole point, and a second definition of the same policy is
#: the defect this module exists to prevent.
FIX_CATALOG: dict[str, FixRecord] = {}


def register(record: FixRecord) -> FixRecord:
    """Add a record to the catalog, refusing a duplicate key.

    A silent overwrite would let two modules disagree about one fix and have
    import order decide the winner — the same failure as two constants, with a
    catalog wrapped around it.
    """
    if record.key in FIX_CATALOG:
        raise ValueError(
            f"duplicate fix key {record.key!r}: a fix is defined once, and a "
            "second definition is how two surfaces come to state one policy "
            "two ways.",
        )
    FIX_CATALOG[record.key] = record
    return record


def fix_for(key: str) -> FixRecord | None:
    return FIX_CATALOG.get(key)


def fix_text(key: str) -> str:
    """The catalogued text for ``key``, raising if it is absent.

    Strict on purpose. A caller asking for a fix by key has that key written
    into it; an absent entry is a programming error at import time, not a
    runtime condition to degrade around. Silently returning ``""`` would ship a
    card with an empty fix, which is the same class of quiet failure as the
    ignored frontmatter key this catalog exists to stop.
    """
    record = FIX_CATALOG.get(key)
    if record is None:
        raise KeyError(
            f"no catalogued fix {key!r} — known keys: "
            f"{', '.join(sorted(FIX_CATALOG))}",
        )
    return record.text


def fixes_for(analyzer: str, persona: str = PERSONA_ANY) -> tuple[FixRecord, ...]:
    """Every catalogued fix ``analyzer`` hands out that ``persona`` can act on.

    An empty result is a real answer: this analyzer has nothing this user can
    do. That is a reason to withhold the OFFER — never to alter what the
    behaviour already cost (Critical Rule 32).
    """
    return tuple(
        record for record in FIX_CATALOG.values()
        if analyzer in record.analyzers
        and (persona == PERSONA_ANY or record.applies_to(persona))
    )


__all__ = [
    "FIX_CATALOG",
    "LEVER_AWARENESS",
    "LEVER_EFFORT",
    "LEVER_MODEL",
    "LEVER_OFFLOAD",
    "LEVER_ROUTING",
    "PERSONA_ANY",
    "PERSONA_CLAUDE_CODE",
    "PERSONA_SDK",
    "FixRecord",
    "fix_for",
    "fix_text",
    "fixes_for",
    "register",
]
