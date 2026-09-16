"""GET /api/v1/shipped: what the window's sessions shipped, and what the
rest cost (shipped-value ledger, contracts §4).

Pure SQL over the ledger tables `core/shipped.match_sessions_to_commits`
keeps current on the daemon pass; no analyzer and no git runs here. The
figures are the same `shipped_summary` the `shipped` analyzer stores, so the
Dashboard card (which asks for its own window), `tj status` in serve mode
and the Optimize card cannot disagree on a definition. Every dollar is
MEASURED and travels with the `framing` block (contracts §1).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.framing import PERSONAS, WindowSummary, compute_framing, plan_determination_mix
from tokenjam.core.shipped import shipped_summary
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/shipped", dependencies=[Depends(require_api_key)])
def get_shipped(
    request: Request,
    since: str = Query("30d", description="Lookback window (e.g. 30d, 7d, 24h)."),
    agent_id: str | None = Query(None, alias="agent_id"),
    persona: str | None = Query(None),
) -> dict[str, Any]:
    db = request.app.state.db
    config = request.app.state.config
    conn = getattr(db, "conn", None)
    if db is None or config is None or conn is None:
        raise HTTPException(status_code=503, detail="Server has no direct database connection.")
    if persona is not None and persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona {persona!r}. Expected one of {sorted(PERSONAS)}.",
        )
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid since: {exc}") from exc
    until_dt = utcnow()
    summary = shipped_summary(
        conn, since_dt, until_dt, agent_id=agent_id, persona_scope=persona,
    )
    payload = summary.to_dict()
    payload["since"] = since
    payload["window_start"] = since_dt.isoformat()
    payload["window_end"] = until_dt.isoformat()
    mix = plan_determination_mix(conn, agent_id)
    framing = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=summary.cost_shipped_usd + summary.cost_unshipped_usd
            + summary.cost_committed_usd,
            sessions=summary.sessions_total,
            plan_tier_mix=mix,
        ),
    )
    payload["framing"] = framing.to_dict()
    return payload
