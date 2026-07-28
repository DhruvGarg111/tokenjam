"""``list`` — every permanent rule the last analyzer run is offering, and where.

Four analyzers write rules (``downsize``, ``resend``, ``subagent``, ``relearn``).
Before this module each hand-rolled its own text and its own single-destination
write, and the only surface that showed them was the Review inbox, one card per
analyzer. A card can express one write. It cannot express "this rule, into these
4 of your 11 projects, here is each diff, apply selectively" — which is what a
placed rule actually is, and why this lifecycle exists beside the inbox rather
than inside it.

Reads only CONFIG and the already-computed proposal cache (``relearn_store``),
never the DB: the analyzers do the expensive work on their own schedule
(``core/optimize/runner``), and re-running a sweep to answer "what rules are on
offer" would put an analyzer pass on a user-facing command.
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.core.rulewrite.types import RuleDestination, RuleWrite

#: The analyzers whose fix is a permanent rule. Not a copy of a gating map —
#: membership here is decided by fix SHAPE, and a name absent from the stored
#: proposals simply contributes nothing, so a persona gate upstream
#: (``runner.PERSONA_DISABLED_ANALYZERS``) removes an analyzer from this
#: surface without any edit here.
RULE_WRITING_ANALYZERS = ("downsize", "resend", "subagent", "relearn")


def _destinations_from_proposal(raw: dict[str, Any]) -> tuple[RuleDestination, ...]:
    """The files this proposal's rule lands in.

    A proposal computed before placement existed carries no ``placement_paths``;
    it degrades to the historical single user-global destination rather than to
    no destination at all, so an older cache still lists and still applies.
    """
    paths = [str(p) for p in (raw.get("placement_paths") or []) if p]
    scope = str(raw.get("placement_scope", "") or "user-global")
    if not paths:
        return ()
    # The per-destination token/dollar split is not carried on the proposal —
    # the payload holds the netted totals, and re-deriving a split here from a
    # different input than the one the netting used is exactly how two surfaces
    # come to disagree. Sessions and figures are therefore left at their
    # not-measured defaults; what a destination is FOR is its path.
    return tuple(RuleDestination(path=path, scope=scope) for path in paths)


def _rule_from_cost_proposal(raw: dict[str, Any]) -> RuleWrite | None:
    """One stored cost proposal as a rule write, or ``None`` if it writes none."""
    analyzer = str(raw.get("analyzer", "") or "")
    if analyzer not in RULE_WRITING_ANALYZERS:
        return None
    rung = int(raw.get("rung", 0) or 0)
    # A suppressed write has its `proposed_fix` cleared and its text moved to
    # `suggestion` by the budget pass, so the artifact has to be read from
    # whichever slot survived. Losing it would make a deferred rule
    # indistinguishable from one that was never derived.
    artifact = str(raw.get("proposed_fix", "") or raw.get("suggestion", "") or "")
    offered = bool(raw.get("write_offered", False))
    if not artifact:
        return None
    if rung < 1 and not raw.get("write_blocked_reason"):
        return None
    return RuleWrite(
        signature=str(raw.get("signature", "") or ""),
        analyzer=analyzer,
        title=str(raw.get("title", "") or ""),
        rung=max(rung, 1),
        artifact_text=artifact,
        destinations=_destinations_from_proposal(raw),
        offered=offered,
        blocked_reason=str(raw.get("write_blocked_reason", "") or ""),
        placement_basis=str(raw.get("placement_basis", "") or ""),
        placement_coverage_note=str(raw.get("placement_coverage_note", "") or ""),
        past_overspend_tokens=raw.get("past_overspend_tokens"),
        past_overspend_usd=raw.get("past_overspend_usd"),
    )


def _rule_from_relearn_cluster(raw: dict[str, Any]) -> RuleWrite | None:
    """One stored relearn cluster as a rule write.

    relearn's clusters already carry a resolved ``suggested_target`` and their
    own per-repo exposure accounting (``relearn._write_exposure_sessions``), so
    the destination is read off the cluster rather than re-derived. The two
    lanes stay separate upstream — different budgets, different populations,
    Critical Rule 27 — and meet only here, at the fix surface.
    """
    target = str(raw.get("suggested_target", "") or "")
    artifact = str(raw.get("proposed_fix", "") or "")
    if not artifact:
        return None
    repos = [str(r) for r in (raw.get("repos") or [])]
    scope = str(raw.get("scope", "") or "user-global")
    destinations = (
        (RuleDestination(
            path=target, scope=scope, sessions=int(raw.get("sessions", 0) or 0),
        ),) if target else ()
    )
    return RuleWrite(
        signature=str(raw.get("signature", "") or ""),
        analyzer="relearn",
        title=str(raw.get("title", "") or ""),
        rung=max(int(raw.get("rung", 1) or 1), 1),
        artifact_text=artifact,
        destinations=destinations,
        offered=bool(raw.get("write_offered", False)) and bool(target),
        blocked_reason=str(raw.get("write_blocked_reason", "") or ""),
        placement_basis=(
            f"scoped to {len(repos)} repo(s) by the recurrence's own sessions"
            if scope == "project" and repos else ""
        ),
        past_overspend_tokens=raw.get("past_overspend_tokens"),
        past_overspend_usd=raw.get("past_overspend_usd"),
    )


def list_rule_writes(config: TjConfig) -> list[RuleWrite]:
    """Every permanent rule currently on offer, ranked by what it addresses.

    Includes rules the write budget did NOT offer, flagged with the reason: the
    text is still copyable and a deferral is a different statement from a
    finding that does not exist. Never raises — an unreadable cache reads as an
    empty list, which is what a fresh install genuinely is.
    """
    from tokenjam.core.optimize import relearn_proposals, relearn_store

    out: list[RuleWrite] = []
    try:
        block = relearn_store.read_cost_proposals(config=config) or {}
    except Exception:
        block = {}
    for raw in (block.get("cost_proposals") or []):
        if not isinstance(raw, dict):
            continue
        rule = _rule_from_cost_proposal(raw)
        if rule is not None and rule.signature:
            out.append(rule)
    try:
        clusters = relearn_proposals.list_proposals(config)
    except Exception:
        clusters = []
    for cluster in clusters:
        raw = cluster if isinstance(cluster, dict) else getattr(cluster, "__dict__", {})
        rule = _rule_from_relearn_cluster(dict(raw))
        if rule is not None and rule.signature:
            out.append(rule)
    # Ranked by what the rule addresses, unpriced last. `None` sorts last
    # rather than as zero: "not measured" is not "worth nothing".
    out.sort(key=lambda r: (r.past_overspend_usd is None, -(r.past_overspend_usd or 0.0)))
    return out


def find_rule(config: TjConfig, signature: str) -> RuleWrite | None:
    """One rule by signature, or ``None``."""
    for rule in list_rule_writes(config):
        if rule.signature == signature:
            return rule
    return None
