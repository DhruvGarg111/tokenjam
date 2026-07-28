"""/api/v1/rules/* — the placed rule-write surface.

Four analyzers end in the same artifact, a rule appended to a CLAUDE.md, and
``core/optimize/rule_placement`` now decides WHICH files each one lands in from
the working directories the sessions recorded. A Review-inbox card can express
one write; it cannot express "this rule, into these 4 of your 11 projects, here
is each diff, apply selectively". These routes back the screen that can.

Everything calls ``core/rulewrite`` in-process — the same shape
``/summarize/*`` uses — so the owner / hash-drift / symlink / gzip-backup
guards are the core module's, never re-implemented here or in JS. Writes are
DRY-RUN by default; the UI sends ``go=true`` only on an explicit Apply.

Reads only config and the proposal cache: listing rules must never trigger an
analyzer sweep (analyzers run at daemon boot, on a schedule, or on a user
rescan — never inline on a route).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from tokenjam.api.deps import require_api_key

router = APIRouter()


def _config(request: Request) -> Any:
    config = request.app.state.config
    if config is None:
        raise HTTPException(
            status_code=503, detail="Server not fully initialised (config missing).",
        )
    return config


@router.get("/rules", dependencies=[Depends(require_api_key)])
def get_rules(request: Request) -> dict[str, Any]:
    """Every permanent rule on offer, with the files it would be written into.

    Includes rules the write budget deferred, each carrying its reason: a
    deferral is a different statement from "no finding", and a surface that
    showed only the offered ones would report the second when the truth is the
    first.
    """
    from tokenjam.core.rulewrite import list_rule_writes

    rules = [r.to_dict() for r in list_rule_writes(_config(request))]
    return {
        "rules": rules,
        "offered_count": sum(1 for r in rules if r["offered"]),
    }


@router.get("/rules/staged", dependencies=[Depends(require_api_key)])
def get_rules_staged(request: Request) -> dict[str, Any]:
    """Staged writes, re-checked against the files as they stand now.

    Each row carries ``applyable`` plus the reason when false — computed by the
    same code the apply path enforces, so the UI can grey out a destination
    that drifted instead of offering an Apply that fails when clicked.
    """
    from tokenjam.core.rulewrite import check_staged

    return {"staged": check_staged(_config(request))}


@router.get("/rules/applied", dependencies=[Depends(require_api_key)])
def get_rules_applied(request: Request) -> dict[str, Any]:
    """Applied rule writes that still have a gzip backup — the Undo surface."""
    from tokenjam.core.rulewrite.store import list_backups

    return {"applied": list_backups(_config(request))}


class StageRequest(BaseModel):
    signature: str


class ApplyRequest(BaseModel):
    #: One rule's destinations, or every staged write when omitted.
    signature: str | None = None
    go: bool = False


class UndoRequest(BaseModel):
    signature: str
    #: One destination, or every destination this rule was applied to.
    path: str | None = None
    go: bool = False


@router.post("/rules/stage", dependencies=[Depends(require_api_key)])
def post_rules_stage(request: Request, body: StageRequest) -> dict[str, Any]:
    """Render one diff per destination and stage them. Writes nothing to disk."""
    from tokenjam.core.rulewrite import RuleWriteRefused, find_rule, stage_rule

    config = _config(request)
    rule = find_rule(config, body.signature)
    if rule is None:
        raise HTTPException(
            status_code=404, detail=f"no rule with signature {body.signature!r}",
        )
    try:
        staged = stage_rule(config, rule)
    except RuleWriteRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"staged": [entry.to_dict() for entry in staged]}


@router.post("/rules/apply", dependencies=[Depends(require_api_key)])
def post_rules_apply(request: Request, body: ApplyRequest) -> dict[str, Any]:
    """Apply staged writes. Dry-run by default; returns
    ``{applied, skipped, dry_run}``.

    A destination that drifted, is a symlink, or is owned by another user is
    skipped WITH its reason — a partial result is the honest one, and a caller
    that only counted successes would report a rule as fully applied when one
    project's file was left untouched.
    """
    from tokenjam.core.rulewrite import RuleWriteRefused, apply_staged

    try:
        return apply_staged(_config(request), body.signature, go=body.go)
    except RuleWriteRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/rules/undo", dependencies=[Depends(require_api_key)])
def post_rules_undo(request: Request, body: UndoRequest) -> dict[str, Any]:
    """Revert an applied rule write. Refuses (409) on drift or a missing backup."""
    from tokenjam.core.rulewrite import RuleWriteRefused, undo

    try:
        return undo(_config(request), body.signature, body.path, go=body.go)
    except RuleWriteRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
