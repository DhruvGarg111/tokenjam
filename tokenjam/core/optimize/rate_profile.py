"""Blended input rate + cache-read ratio, measured off observed spans.

Two analyzers need to price the SAME shape of thing: a block of tokens that is
sent once at the input rate and then re-read on later calls at the cache-read
rate. ``relearn`` prices a failure's re-read tail that way; ``summarize``
prices an always-on prompt file's per-session re-reads that way. Both need the
two rates blended over whichever models the user actually ran, and neither may
invent one when the data cannot supply it.

Deliberately NOT derived from observed ``cost_usd``. An all-in blended $/token
cannot tell an input token from a cache read, and the whole point here is to
price those two classes differently. The rates come from
``pricing/models.toml`` via :func:`tokenjam.core.pricing.get_rates`, weighted
by the token volume each model actually carried.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RateProfile:
    """The two rates a re-read block is billed at, blended over observed models.

    ``input_rate_per_token`` prices the first send; ``cache_read_ratio``
    (cache-read rate / input rate — exactly 0.100 for every Anthropic model in
    ``pricing/models.toml``) prices every re-read of it. ``basis`` names the
    models the blend came from so the derivation is never a black box.
    """

    input_rate_per_token: float
    cache_read_ratio: float
    basis: str

    def cost_of(self, tokens: float, rereads: int) -> float:
        """What ``tokens`` cost when sent once and re-read ``rereads`` times."""
        return (
            tokens
            * self.input_rate_per_token
            * (1.0 + self.cache_read_ratio * max(rereads, 0))
        )


def blended_rate_profile(
    conn: Any,
    *,
    session_ids: set[str] | None = None,
    since: Any = None,
    until: Any = None,
    agent_id: str | None = None,
) -> RateProfile | None:
    """Blend input + cache-read rates over the spans the filters select.

    Pass ``session_ids`` to scope to an explicit set of sessions, or
    ``since``/``until`` (plus an optional ``agent_id``) to scope to a window.
    Returns ``None`` — never a default rate — when there is no connection, no
    matching spans, or no model with pricing data; the caller then reports a
    token figure with no dollars rather than a number borrowed from a model
    the user never ran (CLAUDE.md anti-pattern #22).
    """
    if conn is None:
        return None
    clauses = ["model IS NOT NULL"]
    params: list[Any] = []
    if session_ids is not None:
        if not session_ids:
            return None
        ids = sorted(session_ids)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(ids)))
        clauses.append(f"session_id IN ({placeholders})")
        params.extend(ids)
    if since is not None:
        clauses.append(f"start_time >= ${len(params) + 1}")
        params.append(since)
    if until is not None:
        clauses.append(f"start_time < ${len(params) + 1}")
        params.append(until)
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)

    try:
        rows = conn.execute(
            "SELECT provider, model, "
            "COALESCE(SUM(input_tokens + output_tokens + cache_tokens "
            "+ cache_write_tokens), 0) "
            "FROM spans WHERE " + " AND ".join(clauses) + " GROUP BY provider, model",
            params,
        ).fetchall()
    except Exception:
        return None

    from tokenjam.core.pricing import get_rates

    weighted_input = 0.0
    weighted_cache_read = 0.0
    total_tokens = 0
    models: list[str] = []
    for provider, model, tokens in rows:
        tokens = int(tokens or 0)
        if tokens <= 0:
            continue
        rates = get_rates(str(provider or "unknown"), str(model))
        if rates is None or rates.input_per_mtok <= 0:
            continue
        weighted_input += rates.input_per_mtok * tokens
        weighted_cache_read += rates.cache_read_per_mtok * tokens
        total_tokens += tokens
        models.append(f"{provider}/{model}")
    if total_tokens <= 0:
        return None
    input_per_mtok = weighted_input / total_tokens
    if input_per_mtok <= 0:
        return None
    cache_read_ratio = min(weighted_cache_read / total_tokens / input_per_mtok, 1.0)
    return RateProfile(
        input_rate_per_token=input_per_mtok / 1_000_000,
        cache_read_ratio=cache_read_ratio,
        basis=(
            f"${input_per_mtok:.2f}/MTok input, re-reads at "
            f"{cache_read_ratio:.3f}x that, blended over the models actually "
            f"observed: {', '.join(sorted(set(models)))}"
        ),
    )
