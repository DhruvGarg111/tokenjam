"""
GET /api/v1/optimize — the STORED analyzer report. Never a live analyzer run.

This route used to call `build_report(...)` on the request thread, which
dispatched every registered analyzer over the corpus — including `relearn`,
whose own store exists precisely because it is "far too slow to compute per
HTTP request". `fast=true` did not save it: it dropped only `trim`, applied no
timeout, and still ran the full-corpus scans. Measured on a real corpus the
Review inbox (which reads a stored block) painted instantly while the
Dashboard's recoverable-waste panel and Budget-at-risk card took tens of
seconds to minutes.

So no request path runs an analyzer any more. `core.optimize.report_store`
holds the report; the `tj serve` daemon computes it at boot, on the configured
interval, and when a user presses Rescan (`POST /optimize/rescan`, which starts
a BACKGROUND pass and returns immediately). This route reads that store and
returns the stored body plus the freshness envelope (`status`, `computed_at`,
the window it was observed over).

Ingestion is untouched: traces, spans, sessions, maps, approach and timeline
still update continuously. Only the analyzer layer moved off the request path.

**A cold store is not an empty result.** `status: "never_run"` comes back with
NO report body — never a zeroed one. A zero here would read as "no waste
found", which is a reassurance the data does not support.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from tokenjam.api.deps import require_api_key, require_relearn_write_auth
from tokenjam.cli.cmd_optimize import _rank_findings
from tokenjam.core.framing import (
    WindowSummary,
    agent_persona_mix,
    compute_framing,
    plan_tier_mix,
)
from tokenjam.core.optimize import disabled_analyzers_for_persona, report_to_dict
from tokenjam.core.optimize import report_store
from tokenjam.utils.time_parse import parse_since, utcnow

router = APIRouter()

# Rescan is a write-shaped action (it starts a full-corpus pass), so it takes
# the same local write-token gate the relearn write endpoints use — an
# unauthenticated caller must not be able to make the daemon scan on demand.
_WRITE_AUTH = [Depends(require_api_key), Depends(require_relearn_write_auth)]


@router.get("/optimize", dependencies=[Depends(require_api_key)])
def get_optimize(
    request: Request,
    since: str = Query(
        "30d",
        description="Accepted for backwards compatibility and echoed back as "
                    "requested_since. The response is the STORED report, which "
                    "was computed over [optimize] scan_window_days — see "
                    "window_days / scan_since / scan_until.",
    ),
    agent_id: str | None = Query(None, alias="agent_id"),
    finding: list[str] | None = Query(
        None, description="Accepted and echoed back; the stored report always "
                          "contains every analyzer the persona gate allowed.",
    ),
    budget_provider: str | None = Query(None),
    budget_usd: float | None = Query(None),
    fast: bool = Query(
        False,
        description="Accepted for backwards compatibility and ignored: no "
                    "analyzer runs on this request at any speed.",
    ),
) -> dict[str, Any]:
    """Serve the stored analyzer report plus its freshness envelope.

    When a report exists, the stored body is returned at the TOP LEVEL (so
    `report_from_dict(payload)` keeps working for the CLI) with the envelope
    keys merged alongside it. When the store is cold or has only ever failed,
    the envelope comes back on its own with `report_available: false` — the
    caller renders "not yet computed", never a zero.
    """
    db = request.app.state.db
    config = request.app.state.config
    if db is None or config is None:
        raise HTTPException(
            status_code=503,
            detail="Server not fully initialised (db or config missing).",
        )

    # `since` is still validated so a malformed window is a 400 rather than
    # being silently ignored, even though it no longer selects the data.
    try:
        parse_since(since)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc

    envelope = report_store.stored_report_block(config)
    envelope["requested_since"] = since
    envelope["requested_findings"] = list(finding) if finding else None
    envelope["scan_interval_hours"] = getattr(config.optimize, "scan_interval_hours", None)
    envelope["scan_enabled"] = getattr(config.optimize, "scan_enabled", True)
    envelope["ui_poll_seconds"] = getattr(config.optimize, "scan_ui_poll_seconds", 0)

    report = report_store.stored_report(config)
    if report is None:
        # COLD (or error-only). No report body, no finding list, no rollups —
        # emphatically no zeros. Everything downstream must render this as
        # "not computed yet", which is a different claim from "found nothing".
        envelope["report_available"] = False
        return envelope

    payload: dict[str, Any] = report_to_dict(report)
    payload.update(envelope)
    payload["report_available"] = True

    # `fast` no longer skips anything (nothing runs here), so nothing is
    # "skipped for speed". The key stays for wire compatibility.
    persona_disabled = disabled_analyzers_for_persona(report.persona)
    payload["skipped_analyzers"] = []
    # The names the persona gate dropped, so the UI can tell "ran, found
    # nothing" (render the empty state) from "not run for this persona"
    # (render nothing at all).
    payload["persona_disabled_analyzers"] = sorted(persona_disabled)

    # Biggest-waste-first ranking — the same `_rank_findings` the CLI's text
    # view ranks by, so the web doesn't fall back to Object.keys() insertion
    # order. `share` of None means "no quantified estimate" (unranked), which
    # is NOT zero — the UI must not sort those away as de-minimis.
    payload["finding_rank"] = [
        {"name": name, "share": share} for name, share in _rank_findings(report, None)
    ]

    # Plan-tier / persona mix are cheap direct queries (no analyzer), so they
    # stay live: they describe the corpus as it is right now, and a stale mix
    # would frame a fresh figure under the wrong pricing mode.
    conn = getattr(db, "conn", None)
    since_dt = report.window.since
    until_dt = report.window.until or utcnow()
    payload["plan_tier_mix"] = _mix(plan_tier_mix, conn, since_dt, until_dt, agent_id)
    payload["agent_persona_mix"] = _mix(agent_persona_mix, conn, since_dt, until_dt, agent_id)

    w = report.window
    payload["framing"] = compute_framing(
        config,
        WindowSummary(
            total_cost_usd=float(getattr(w, "total_cost_usd", 0.0) or 0.0),
            total_tokens=int(getattr(w, "total_tokens", 0) or 0),
            sessions=int(getattr(w, "sessions", 0) or 0),
            plan_tier_mix=payload["plan_tier_mix"],
        ),
    ).to_dict()

    return payload


def _mix(fn: Any, conn: Any, since_dt: Any, until_dt: Any, agent_id: str | None) -> dict:
    """Best-effort mix query; `{}` when the storage layer exposes no connection
    (e.g. a proxy backend) rather than failing the whole read."""
    if conn is None:
        return {}
    try:
        return dict(fn(conn, since_dt, until_dt, agent_id))
    except Exception:
        return {}


@router.post("/optimize/rescan", dependencies=_WRITE_AUTH)
def rescan_optimize(request: Request) -> dict[str, Any]:
    """Start a background analyzer scan and return immediately.

    Three rails, all always-on rather than staged:

    * **Overlap guard** — `report_store.trigger_background_recompute` no-ops
      when a scan is already in flight, so pressing Rescan twice (or pressing
      it while the scheduled job runs) costs one pass, not two.
    * **Rate limit** — a request inside `[optimize] scan_min_rescan_seconds`
      is answered `throttled` with the stored result untouched, so a user
      cannot hammer full-corpus passes.
    * **Own connection** — the scan runs on a daemon thread against a FRESH
      `DuckDBBackend`, never this request's connection, so it never contends
      with the live writer and never blocks this response.
    """
    config = request.app.state.config
    db = getattr(request.app.state, "db", None)
    if config is None or db is None or getattr(db, "conn", None) is None:
        return {"status": "unavailable", "reason": "no direct database connection"}

    if report_store.is_computing():
        return {**report_store.stored_report_block(config), "started": False,
                "reason": "a scan is already running"}
    if report_store.rescan_throttled(config):
        return {**report_store.stored_report_block(config), "started": False,
                "throttled": True,
                "reason": "rescanned too recently; showing the stored result"}

    from tokenjam.core.db import DuckDBBackend

    started = report_store.trigger_background_recompute(
        lambda: DuckDBBackend(config.storage), config,
    )
    return {**report_store.stored_report_block(config), "started": started}
