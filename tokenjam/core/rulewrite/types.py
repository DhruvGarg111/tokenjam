"""The shapes the rule-write lifecycle passes around.

One rule, N destinations. Everything downstream — staging, the diff, apply,
undo — is per DESTINATION, because that is the granularity a user reviews at:
"this rule, into these 4 of your 11 projects, here is each diff, apply
selectively" is a sentence a Review-inbox card structurally cannot say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Rungs this lifecycle writes. Rung 3+ (hooks, config) is executed rather than
#: sent as prompt text, so it is not a "rule" in this sense and is not handled
#: here — ``relearn_apply`` keeps owning those.
RUNG_NOTE = 1
RUNG_SKILL = 2


@dataclass(frozen=True)
class RuleDestination:
    """One file a rule would land in, and the evidence that put it there."""

    path: str
    #: ``project`` or ``user-global``.
    scope: str = "user-global"
    #: How many of the window's sessions load this file. The standing cost is
    #: charged against exactly this count, which is the whole point of placing
    #: a rule rather than defaulting it to the user-global file.
    sessions: int = 0
    #: This destination's share of the rule's finding, split by session weight.
    #: ``usd`` is ``None`` — never 0.0 — when the finding carried no dollars.
    tokens: int = 0
    usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "scope": self.scope, "sessions": self.sessions,
            "tokens": self.tokens, "usd": self.usd,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuleDestination:
        usd = raw.get("usd")
        return cls(
            path=str(raw.get("path", "")),
            scope=str(raw.get("scope", "user-global")),
            sessions=int(raw.get("sessions", 0) or 0),
            tokens=int(raw.get("tokens", 0) or 0),
            usd=None if usd is None else float(usd),
        )


@dataclass(frozen=True)
class RuleWrite:
    """One permanent rule an analyzer is offering, with where it would go.

    ``signature`` is the proposal's own stable identity, so a rule listed here,
    a card in the Review inbox and a staged diff on disk are the same thing
    under one name.
    """

    signature: str
    analyzer: str
    title: str
    rung: int
    artifact_text: str
    destinations: tuple[RuleDestination, ...] = ()
    #: Mirrors the proposal's own verdict. A rule the write budget did not
    #: offer is still LISTED — its text is copyable and its reason is stated —
    #: it simply cannot be staged. Hiding it would make a deliberate deferral
    #: indistinguishable from an analyzer that found nothing.
    offered: bool = True
    blocked_reason: str = ""
    #: Why the rule is going where it is going, and what the placement could not
    #: cover. Carried verbatim from ``core/optimize/rule_placement``; never
    #: re-derived here, so the CLI, the UI and the payload cannot disagree.
    placement_basis: str = ""
    placement_coverage_note: str = ""
    #: The finding's own figures, for ordering and for the card. Not recomputed.
    past_overspend_tokens: int | None = None
    past_overspend_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "analyzer": self.analyzer,
            "title": self.title,
            "rung": self.rung,
            "artifact_text": self.artifact_text,
            "destinations": [d.to_dict() for d in self.destinations],
            "offered": self.offered,
            "blocked_reason": self.blocked_reason,
            "placement_basis": self.placement_basis,
            "placement_coverage_note": self.placement_coverage_note,
            "past_overspend_tokens": self.past_overspend_tokens,
            "past_overspend_usd": self.past_overspend_usd,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuleWrite:
        tokens = raw.get("past_overspend_tokens")
        usd = raw.get("past_overspend_usd")
        return cls(
            signature=str(raw.get("signature", "")),
            analyzer=str(raw.get("analyzer", "")),
            title=str(raw.get("title", "")),
            rung=int(raw.get("rung", 0) or 0),
            artifact_text=str(raw.get("artifact_text", "")),
            destinations=tuple(
                RuleDestination.from_dict(d) for d in (raw.get("destinations") or [])
            ),
            offered=bool(raw.get("offered", True)),
            blocked_reason=str(raw.get("blocked_reason", "") or ""),
            placement_basis=str(raw.get("placement_basis", "") or ""),
            placement_coverage_note=str(raw.get("placement_coverage_note", "") or ""),
            past_overspend_tokens=None if tokens is None else int(tokens),
            past_overspend_usd=None if usd is None else float(usd),
        )


@dataclass
class StagedRuleWrite:
    """One rule's staged result for ONE destination: what would be written,
    the diff a reviewer reads, and the hash apply re-checks the file against.

    Staging persists the rendered OUTPUT rather than a recipe, so what a user
    approved in the diff is byte-for-byte what apply writes. The hash is the
    guard that makes that promise keepable: a file edited between staging and
    applying is skipped and reported, never merged into.
    """

    signature: str
    path: str
    scope: str
    rung: int
    title: str
    analyzer: str
    #: sha256 of the file as it stood when this was staged.
    source_sha256: str
    #: The full file content to write.
    rendered: str
    diff: str
    standing_tokens_per_session: int = 0
    sessions: int = 0
    #: True when the file did not exist at staging time and would be created.
    creates_file: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature, "path": self.path, "scope": self.scope,
            "rung": self.rung, "title": self.title, "analyzer": self.analyzer,
            "source_sha256": self.source_sha256, "rendered": self.rendered,
            "diff": self.diff,
            "standing_tokens_per_session": self.standing_tokens_per_session,
            "sessions": self.sessions, "creates_file": self.creates_file,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StagedRuleWrite:
        return cls(
            signature=str(raw.get("signature", "")),
            path=str(raw.get("path", "")),
            scope=str(raw.get("scope", "")),
            rung=int(raw.get("rung", 0) or 0),
            title=str(raw.get("title", "")),
            analyzer=str(raw.get("analyzer", "")),
            source_sha256=str(raw.get("source_sha256", "")),
            rendered=str(raw.get("rendered", "")),
            diff=str(raw.get("diff", "")),
            standing_tokens_per_session=int(
                raw.get("standing_tokens_per_session", 0) or 0
            ),
            sessions=int(raw.get("sessions", 0) or 0),
            creates_file=bool(raw.get("creates_file", False)),
            notes=list(raw.get("notes") or []),
        )


class RuleWriteRefused(Exception):
    """Refusing to stage, apply or undo a rule write.

    Carries the house-voice text; callers surface it as a 409 / a re-stage
    prompt. Every guard in this package raises this rather than returning a
    falsy value, so no caller can accidentally treat a refusal as a no-op.
    """
