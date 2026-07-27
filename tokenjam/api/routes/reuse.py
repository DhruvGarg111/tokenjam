"""
GET /api/v1/reuse/clusters — server-side Reuse analyzer + skeleton-ready data.

`tj report --reuse` renders a per-cluster planning skeleton, which needs both
the Reuse finding AND each cluster's planning-call completion text. Both come
from direct `spans` queries that DuckDB blocks while `tj serve` holds the write
lock, so the report errored out whenever the daemon was up (#154).

This is a *dedicated* endpoint (issue #154 Option B) rather than bolting the
skeleton text onto `/api/v1/optimize`: the per-cluster planning text can be many
KB, and the Overview polls `/optimize` — we don't make every poll pay for
report-only data. This endpoint is hit only when a report is generated.

Like every other analyzer-consuming route, it does NOT run the analyzer: the
Reuse finding comes out of the stored report `core.optimize.report_store` keeps
warm (daemon boot / scheduled interval / user-pressed rescan). The only live
work here is `gather_planning_texts`, which is a plain span lookup for clusters
the stored finding already named — no analyzer, no full-corpus scan.

Returns `report_to_dict(report)` (so the CLI reconstructs the finding via the
existing `report_from_dict`) plus the freshness envelope and two report-only
extras: `planning_texts` ({session_id: completion text or null}) and
`pricing_mode`.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.export.reuse_report import gather_planning_texts
from tokenjam.core.framing import dominant_plan, plan_tier_mix, pricing_mode_for
from tokenjam.core.optimize import report_store, report_to_dict
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()


@router.get("/reuse/clusters", dependencies=[Depends(require_api_key)])
def get_reuse_clusters(
    request: Request,
    since: str = Query(
        "30d",
        description="Echoed back as requested_since. The finding comes from the "
                    "stored report; see window_days / scan_since / scan_until.",
    ),
    agent_id: str | None = Query(None, alias="agent_id"),
) -> dict[str, Any]:
    """Serve the stored Reuse finding + its skeleton text."""
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    try:
        parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    envelope = report_store.stored_report_block(config)
    envelope["requested_since"] = since

    report = report_store.stored_report(config)
    if report is None:
        # Cold store: no finding, and deliberately no empty-looking one. The
        # caller must say "not computed yet", not "no reuse clusters found".
        return {**envelope, "report_available": False,
                "planning_texts": {}, "pricing_mode": "unknown"}

    payload = report_to_dict(report)
    payload.update(envelope)
    payload["report_available"] = True

    finding = report.findings.get("reuse")
    conn = getattr(db, "conn", None)
    # Skeleton text + pricing mode both need the DB; the daemon owns it here.
    # Neither is an analyzer — the clusters were already chosen by the stored
    # finding; this only fetches the text for the sessions it named.
    if finding is not None and finding.clusters and conn is not None:
        since_dt = report.window.since
        until_dt = report.window.until or utcnow()
        payload["planning_texts"] = gather_planning_texts(conn, finding)
        payload["pricing_mode"] = pricing_mode_for(
            dominant_plan(plan_tier_mix(conn, since_dt, until_dt, agent_id))
        )
    else:
        payload["planning_texts"] = {}
        payload["pricing_mode"] = "unknown"

    return payload
