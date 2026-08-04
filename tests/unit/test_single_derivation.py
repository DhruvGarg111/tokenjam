"""The generic guard: one test PER SHAPE, driven by the registry, not one
test per value.

See ``core/optimize/single_derivation.py`` for the registry itself and why
this module exists. Three shapes are tested here:

1. :data:`SEAMS` — parametrized: every symbol-reachable-from-one-module
   invariant. Adding a value here is a new line in the registry, never a new
   test function.
2. :data:`BESPOKE_SEAMS` — parametrized: seams real enough to pin but not
   expressible as a reachability check. This does not re-verify the
   invariant (the named test already does that); it verifies the NAMED TEST
   STILL EXISTS, so deleting or renaming it out from under the registry
   fails loudly instead of silently losing coverage.
3. Aggregate-versus-parts — one passing pin (the `cache` family, which holds
   the invariant by construction) and one ``xfail(strict=True)`` pin (the
   `downsize` per-agent path, which does not) — see
   ``core/optimize/single_derivation.py``'s docstring on ``KNOWN_GAPS``.
"""
from __future__ import annotations

import pytest

from tokenjam.core.optimize.single_derivation import (
    BESPOKE_SEAMS,
    SEAMS,
    BespokeSeam,
    SingleSeam,
    check_bespoke_seam,
    offenders_for,
)


@pytest.mark.parametrize("seam", SEAMS, ids=[s.name for s in SEAMS])
def test_no_seam_gains_a_second_derivation(seam: SingleSeam) -> None:
    offenders = offenders_for(seam)
    assert not offenders, (
        f"{seam.name!r} is meant to have exactly one derivation, in "
        f"{sorted(seam.allowed_modules)}. Found {seam.symbol!r} reached "
        f"from outside that module at:\n  " + "\n  ".join(offenders) +
        f"\n\nWhy this must have one seam: {seam.reason}"
    )


@pytest.mark.parametrize("seam", BESPOKE_SEAMS, ids=[s.name for s in BESPOKE_SEAMS])
def test_every_bespoke_seam_still_has_a_live_test(seam: BespokeSeam) -> None:
    problem = check_bespoke_seam(seam)
    assert problem is None, (
        f"the registry names {seam.test_module}.{seam.test_name} as the "
        f"only guard for {seam.name!r}, but {problem}. This seam was not "
        "mechanized because: " + seam.reason_not_mechanized
    )


def test_the_registry_has_no_duplicate_seam_names() -> None:
    names = [s.name for s in SEAMS] + [s.name for s in BESPOKE_SEAMS]
    assert len(names) == len(set(names)), (
        "two registry entries share a name — pick a distinct name for each; "
        f"names were: {names}"
    )


def test_a_seam_reports_its_own_symbol_when_violated() -> None:
    """The walker actually WALKS, rather than trivially passing every entry.

    Only the shipped package is in scope (tests are always exempt — the same
    exemption the rollup and window-anchor tests rely on, since a test
    legitimately constructs the raw guarded symbol to pin its own
    behaviour). So the fixture here has to be a real call site INSIDE
    ``tokenjam/``: pointing at ``RateProfile`` with an empty allow-list
    must catch its own defining module using it.
    """
    fake = SingleSeam(
        name="test-only: RateProfile with nothing allowed",
        description="sanity check on the walker itself",
        symbol="RateProfile",
        kind="call",
        allowed_modules=frozenset(),
        reason="exercises the mechanism, not a real product invariant",
    )
    offenders = offenders_for(fake)
    assert any("rate_profile.py" in o for o in offenders)


# --------------------------------------------------------------------- #
# Aggregate versus parts
# --------------------------------------------------------------------- #
def test_the_cache_family_sums_exactly_to_the_findings_own_total() -> None:
    """The invariant :data:`KNOWN_GAPS` says `downsize` lacks, HOLDS for
    `cache` — proving the shape is enforceable, not just aspirational.

    `_cache_to_proposals` nets each generic row against whatever the more
    specific per-agent cards already claimed for the same (provider, model)
    (`_per_agent_cache_recoverable_by_model`), so the family's cards partition
    the finding's own total rather than double-claiming or dropping any of
    it.
    """
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
        UncachedAgentCandidate,
        estimate_cache_recoverable,
    )
    from tokenjam.core.optimize.cost_proposals import (
        _cache_to_proposals,
        _cache_uncached_to_proposals,
    )

    row = CacheEfficacyRow(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=100_000, cache_tokens=1_000, efficacy=0.01,
        support="full", flagged=True,
    )
    row_usd, row_tokens = estimate_cache_recoverable([row])
    assert row_usd is not None and row_tokens is not None

    agent = UncachedAgentCandidate(
        agent_id="worker-a", provider="anthropic", model="claude-opus-4-7",
        calls=5, sessions=2, assumed_prefix_tokens=1_000,
        past_overspend_usd=round(row_usd * 0.4, 6),
        past_overspend_tokens=int(row_tokens * 0.4),
        estimate_basis="a1 basis",
    )
    finding = CacheEfficacyFinding(
        rows=[row], flagged=[row],
        past_overspend_usd=row_usd, past_overspend_tokens=row_tokens,
        estimate_basis="cache basis", uncached_agents=[agent],
    )

    proposals = (
        _cache_to_proposals(finding, persona="unknown")
        + _cache_uncached_to_proposals(finding, persona="unknown")
    )
    total = sum(p.past_overspend_usd or 0.0 for p in proposals)
    assert total == pytest.approx(finding.past_overspend_usd, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap, not yet closed — see KNOWN_GAPS in "
        "core/optimize/single_derivation.py. An unexpected pass here means "
        "the gap was closed: delete this xfail AND the matching account in "
        "KNOWN_GAPS together."
    ),
)
def test_the_downsize_per_agent_path_can_undercount_the_findings_own_total() -> None:
    """The gap `KNOWN_GAPS` names, reproduced directly.

    A finding whose aggregate `past_overspend_usd` includes a candidate on a
    model with NO pricing data (so `build_agent_price_rows` legitimately
    drops that group rather than guessing a rate) surfaces almost none of
    that money once the per-agent cards replace the window-wide card — and
    nothing on any card says so.

    Measured here: an aggregate of $998.00 surfaces as roughly $0.11 across
    the cards the Review inbox actually renders.
    """
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
    from tokenjam.core.optimize.analyzers.model_downgrade import DowngradeFinding
    from tokenjam.core.optimize.cost_proposals import _downsize_to_proposal
    from tokenjam.utils.time_parse import utcnow
    from datetime import timedelta

    priced_candidates = [{
        "session_id": f"s{i}", "agent_id": "worker-a", "provider": "anthropic",
        "model": "claude-opus-4-7", "alt_model": "claude-sonnet-5",
        "input_tokens": 1_000, "output_tokens": 500,
        "cache_tokens": 2_000, "cache_write_tokens": 100,
        "started_at": utcnow() - timedelta(days=1),
    } for i in range(10)]
    # An entire agent's worth of spend on a model `build_agent_price_rows`
    # cannot price, so its group is DROPPED rather than guessed at — that
    # drop is correct in isolation; the gap is that nothing downstream
    # discloses it.
    unpriceable_candidate = [{
        "session_id": "s-unpriced", "agent_id": "worker-b", "provider": "anthropic",
        "model": "totally-unpriced-model-xyz", "alt_model": "claude-sonnet-5",
        "input_tokens": 5_000_000, "output_tokens": 100,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "started_at": utcnow() - timedelta(days=1),
    }]
    rows = build_agent_price_rows(
        priced_candidates + unpriceable_candidate, window_days=30.0,
    )
    assert {"worker-a", "worker-b"} - {r.agent_id for r in rows}, (
        "fixture assumption broken: expected build_agent_price_rows to drop "
        "the unpriceable agent's group"
    )

    finding = DowngradeFinding(
        candidate_sessions=11, total_sessions=20,
        actual_cost_usd=999.0, alternative_cost_usd=1.0,
        monthly_savings_usd=0.0, percent_of_sessions=55.0,
        examples=[], suggestions={"claude-opus-4-7": "claude-sonnet-5"},
        past_overspend_usd=998.0, percent_of_tokens=90.0,
        estimate_basis="downsize basis", per_agent=rows,
    )
    proposals = _downsize_to_proposal(finding, config=None, persona="unknown")
    offered = sum(p.past_overspend_usd or 0.0 for p in proposals)

    # The invariant this file wants: the inbox's cards should sum to (or at
    # least disclose falling short of) the aggregate the Dashboard publishes
    # for the same finding. It currently does neither.
    assert offered == pytest.approx(finding.past_overspend_usd, abs=1.0)
