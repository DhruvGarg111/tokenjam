"""GET /api/v1/traces — trace listing and detail."""
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Request

from tokenjam.api.deps import require_api_key
from tokenjam.core.data_span import available_data_span
from tokenjam.core.framing import (
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.models import TraceCostStats, TraceFilters
from tokenjam.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])


def _traces_framing(request: Request, agent_id: str | None) -> dict:
    """Plan-tier framing block for trace cost figures (#187).

    Traces and trace-detail are not window-scoped views, so the plan is derived
    from a window-INDEPENDENT session mix (`plan_determination_mix`) — the same
    helper `/cost` uses. The web UI consumes this block to suppress / reframe raw
    dollar costs for subscription / local users (honesty discipline, Rule 14)
    instead of re-deriving the suppression rules in JS (single compute path).
    """
    db = request.app.state.db
    config = request.app.state.config
    conn = getattr(db, "conn", None)
    mix = plan_determination_mix(conn, agent_id) if conn is not None else {}
    framing = compute_framing(
        config,
        WindowSummary(plan_tier_mix=mix, sessions=sum(mix.values())),
    )
    return framing.to_dict()


# Valid values for the `sort` query param. Anything else falls back to
# "recent" rather than erroring — same forgiving-default philosophy as the
# web UI's readParam() (bad/old bookmarked URLs degrade gracefully).
_VALID_TRACE_SORTS = ("recent", "cost")


@router.get("/traces")
async def list_traces(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 200,
    offset: int = 0,
    status: str | None = None,
    span_name: str | None = None,
    sort: str | None = None,
    min_cost_usd: float | None = None,
) -> dict:
    db = request.app.state.db
    try:
        since_dt = parse_since(since) if since else None
        until_dt = parse_since(until) if until else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid --since: {exc}") from exc
    filters = TraceFilters(
        agent_id=agent_id,
        since=since_dt,
        until=until_dt,
        limit=limit,
        offset=offset,
        status=status,
        span_name=span_name,
        sort=sort if sort in _VALID_TRACE_SORTS else "recent",
        min_cost_usd=min_cost_usd,
    )
    traces = db.get_traces(filters)
    total_count = db.count_traces(filters) if hasattr(db, "count_traces") else len(traces)
    stats = db.get_trace_cost_stats(filters) if hasattr(db, "get_trace_cost_stats") else None
    conn = getattr(db, "conn", None)
    return {
        "traces": [
            {
                "trace_id": t.trace_id,
                "agent_id": t.agent_id,
                "name": t.name,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "duration_ms": t.duration_ms,
                "cost_usd": t.cost_usd,
                "status_code": t.status_code,
                "span_count": t.span_count,
                # Per-trace token totals — the UI renders per-row cost as TOKENS
                # for subscription/local users (#249), never "% of cycle".
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                # Statistical cost-outlier flag for this window; see
                # `outlier_rule` below for the plain-language explanation and
                # the numbers behind it.
                "is_outlier": t.is_outlier,
            }
            for t in traces
        ],
        "count": len(traces),
        "total_count": total_count,
        "framing": _traces_framing(request, agent_id),
        "outlier_rule": _outlier_rule_dict(stats),
        # `available_days` (core/data_span.py) so the Traces window selector
        # can derive its options from what the store actually holds, the same
        # way the Dashboard's does — instead of a fixed 24h/7d/30d/90d list.
        "data_span": available_data_span(conn).to_dict(),
    }


def _outlier_rule_dict(stats: TraceCostStats | None) -> dict | None:
    """Plain-language + numeric explanation of the `is_outlier` flag.

    The UI renders this verbatim so a user never has to read source to know
    what "outlier" means. `threshold_usd` is None when there weren't enough
    priced traces in the window to compute a reliable range — in that case no
    trace in the response is flagged, and the UI should say so rather than
    imply a rule ran and found nothing.
    """
    if stats is None:
        return None
    return {
        "method": stats.method,
        "sample_size": stats.sample_size,
        "min_sample": stats.min_sample,
        "q1_usd": stats.q1_usd,
        "q3_usd": stats.q3_usd,
        "threshold_usd": stats.threshold_usd,
    }


# How many of a trace's own spans to surface as "the costliest spans in this
# trace" — a runaway retry loop or a single blown-up tool call should be
# findable without scanning the whole waterfall by eye.
TOP_COST_SPAN_LIMIT = 5


@router.get("/traces/{trace_id}")
async def get_trace(request: Request, trace_id: str) -> dict:
    db = request.app.state.db
    spans = db.get_trace_spans(trace_id)
    # Scope the plan determination to this trace's agent when known (falls back
    # to the whole install) so subscription / local cost suppression matches the
    # Traces list and Cost screen.
    agent_id = next((s.agent_id for s in spans if getattr(s, "agent_id", None)), None)
    # Rank spans WITHIN this trace by cost. Computed from the span list already
    # fetched for the waterfall (bounded by one trace's span count — never a
    # separate DB scan), since the waterfall needs every span regardless of
    # rank to draw the tree; only the ranking is new.
    priced = [s for s in spans if (getattr(s, "cost_usd", None) or 0) > 0]
    top_cost_spans = sorted(priced, key=lambda s: s.cost_usd or 0, reverse=True)[:TOP_COST_SPAN_LIMIT]
    return {
        "trace_id": trace_id,
        "spans": [_span_to_dict(s) for s in spans],
        "span_count": len(spans),
        "top_cost_span_ids": [s.span_id for s in top_cost_spans],
        "framing": _traces_framing(request, agent_id),
    }


def _span_to_dict(span: object) -> dict:
    """Serialise a NormalizedSpan to a JSON-safe dict."""
    from tokenjam.core.models import NormalizedSpan
    assert isinstance(span, NormalizedSpan)
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "kind": span.kind.value,
        "status_code": span.status_code.value,
        "status_message": span.status_message,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "duration_ms": span.duration_ms,
        "agent_id": span.agent_id,
        "session_id": span.session_id,
        "provider": span.provider,
        "model": span.model,
        "tool_name": span.tool_name,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cache_tokens": span.cache_tokens,            # cache-READ tokens
        "cache_write_tokens": span.cache_write_tokens,  # cache-CREATE tokens (#17)
        "cost_usd": span.cost_usd,
        "request_type": span.request_type,
        "conversation_id": span.conversation_id,
        "attributes": span.attributes,
    }
