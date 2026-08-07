"""THE allocation point for every permanent write a report may offer.

**The defect this exists for.** Two producers propose permanent writes into the
user's agent files: ``analyzers/relearn`` (recurring-failure clusters) and
``cost_proposals`` (the write-bearing cost cards). Each used to build its OWN
:class:`~tokenjam.core.optimize.write_budget.WriteBudget` from its own pair of
constants and spend it in its own pass, inside itself, before the other's
candidates existed. Three consequences, all of them real and none of them
visible from either side:

* **The bound was the SUM of two caps.** A user reading "at most N permanent
  rules this window" was subject to N + M, and one window could grow the same
  ``CLAUDE.md`` by both allowances at once.
* **The headroom was spent twice.** Both passes sized their share against the
  SAME ``summarize``-measured always-loaded footprint, each as though it were
  the only writer. The growth share that is supposed to cap what a file takes
  on per window was therefore applied twice to one file.
* **Ranking was per-producer.** A high-net cost write and a low-net relearn
  write never entered one ``ranked`` list, so the comparison that decides which
  of them deserves a permanent block never happened. Whichever producer ran
  first spent its slots on its own best candidates regardless of what the other
  one was holding.

**The shape of the fix.** Not a second guard, and not a reconciliation pass:
allocation is REMOVED from both producers and happens exactly once, here, over
the union of their candidates. Each producer keeps the half only it can do —
deciding what its candidates ARE, what they cost, and what exposure a rule for
them would be charged against — and neither may size or spend a budget. That is
the same lesson Critical Rule 39 draws one level down: when two passes write the
same field, the fix is to leave one decision, not to referee two.

**Why the report is the unit.** A report is what a surface publishes, and every
producer's candidates are derivable from one. So allocation is a pure function
of a report, memoized onto it: whatever findings a report carries, both lanes'
candidates come out of it and are ranked together, once. A narrower report
(``tj optimize <analyzer>``) simply has fewer candidates — the invariant is not
"both lanes are always present", it is "no write is offered until every
candidate this report can produce has been seen together".

**What each lane keeps.** The one thing that genuinely differs between them is
the session count a rule's standing cost is charged against: relearn measures
its own unbounded corpus pace, the cost lane the report window's. Both resolve
that per candidate BEFORE they get here (relearn on
``RelearnCluster.write_exposure_sessions``, the cost lane through placement), so
one shared allocation cannot silently re-price either producer's rules on the
other's horizon.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from tokenjam.core.optimize import write_budget as wb

#: Lane names. They namespace both the decision KEY and the family key, so two
#: producers cannot collide on a signature and — more importantly — a relearn
#: family and a cost family are never merged into one block. They write
#: different artifacts; only same-lane candidates share a block.
LANE_RELEARN = "relearn"
LANE_COST = "cost"


def _namespaced(lane: str, candidate: wb.WriteCandidate) -> wb.WriteCandidate:
    return replace(
        candidate,
        key=f"{lane}:{candidate.key}",
        family=f"{lane}:{candidate.family}",
    )


def _decision_from_stored(payload: dict[str, Any]) -> wb.WriteDecision:
    """Rebuild a decision from its serialized form.

    A report crosses a process boundary (``report_store`` writes JSON, the scan
    cycle reads it back) between the pass that ALLOCATES and the pass that
    APPLIES the cost half, so the decisions have to survive the round trip.
    ``destinations`` comes back as a list and is re-tupled; unknown keys are
    dropped so a payload written by a newer build cannot crash an older reader.
    """
    fields = {f for f in wb.WriteDecision.__dataclass_fields__}
    data = {k: v for k, v in payload.items() if k in fields}
    if "destinations" in data:
        data["destinations"] = tuple(data["destinations"] or ())
    return wb.WriteDecision(**data)


def allocate_report_writes(
    report: Any, *, config: Any = None, window_days: float = 30.0,
    pricing_mode: str = "api",
) -> dict[str, dict[str, Any]]:
    """Rank every permanent write this report proposes, from EITHER producer,
    against ONE budget. Returns the serialized decisions keyed by
    ``"<lane>:<signature>"``, and memoizes them onto ``report.write_decisions``.

    Idempotent: a report that already carries decisions is returned untouched.
    That is what lets the allocation run at the end of the report build and be
    consumed later by the cost pass — including after the report has been
    stored and rehydrated — without a second, differently-sized allocation ever
    happening.

    relearn's half is applied HERE, to ``report.findings["relearn"]``, because
    its clusters are the finding a surface reads directly; the cost half is
    returned for :func:`decisions_for` to hand to ``cost_proposals``, whose
    proposals are built downstream of this call.

    Never raises. A report this cannot allocate over (a hand-built one, a
    producer that blew up) leaves the decisions empty, which withdraws offers
    rather than inventing them — the safe direction for a gate.
    """
    existing = getattr(report, "write_decisions", None)
    if existing:
        return existing

    from tokenjam.core.optimize import cost_proposals
    from tokenjam.core.optimize.analyzers import relearn as relearn_mod

    findings = getattr(report, "findings", {}) or {}

    relearn_finding = findings.get("relearn")
    relearn_candidates: list[wb.WriteCandidate] = []
    if relearn_finding is not None:
        # The SAME window the Review inbox headline matches its relearn rows
        # against (see `inbox_contribution.exact_window_label` — no
        # nearest-match fallback there either). Resolved off the finding's
        # OWN precomputed vocabulary, not off the fixed label set, since
        # `run(ctx)` widens that vocabulary to include the report's actual
        # window (`relearn.run` -> `window_labels_including`). Ranking
        # candidates on different horizons is exactly the defect this module
        # exists to fix, so relearn's candidates must never rank on anything
        # but a bucket matching this report's own window — see
        # `relearn.write_candidates`'s `window_label` for what happens when no
        # such bucket exists.
        from tokenjam.core.optimize import inbox_contribution

        relearn_window_label = inbox_contribution.exact_window_label(
            window_days,
            list(getattr(relearn_finding, "past_overspend_windows", None) or {}),
        )
        try:
            relearn_candidates = relearn_mod.write_candidates(
                list(getattr(relearn_finding, "clusters", None) or []),
                window_label=relearn_window_label,
            )
        except Exception:  # noqa: BLE001 - one producer must not sink the pass
            relearn_candidates = []

    try:
        cost_candidates, _placements = cost_proposals.write_candidates_from_report(
            report, config=config, pricing_mode=pricing_mode,
        )
    except Exception:  # noqa: BLE001 - see above
        cost_candidates = []

    # ONE budget, sized ONCE off the `summarize` scan both producers used to
    # read separately. Sizing it here rather than in either producer is what
    # stops the same headroom being spent twice.
    # THIS REPORT'S OWN PERSONA decides whether `summarize` may size the budget,
    # even when the pass that built the report dispatched a wider analyzer set
    # (the daemon computes the union so one artifact answers for either side of
    # the persona picker — see `runner.build_report`'s `personas`). `summarize`
    # is gated for `sdk` because that persona has no agent instruction files to
    # scan; letting a union pass's summarize finding through here would size an
    # SDK window's write budget off a footprint the matrix records as
    # deliberately unmeasured for it (ANALYZER-PERSONA-MATRIX.md §10, the
    # documented `None` path that leaves the lane cap intact).
    from tokenjam.core.optimize.runner import disabled_analyzers_for_persona

    _report_persona = str(getattr(report, "persona", "") or "unknown")
    summarize_finding = (
        None if "summarize" in disabled_analyzers_for_persona(_report_persona)
        else findings.get("summarize")
    )
    budget = wb.build_write_budget(
        existing_agent_file_tokens=wb.measured_agent_file_tokens(summarize_finding),
        existing_by_path=wb.measured_agent_file_tokens_by_path(summarize_finding),
    )
    # The shared basis is only ever a FALLBACK exposure, for a candidate that
    # named none. relearn's candidates always name one, so this is the cost
    # lane's own report-window basis and nothing else reads it.
    basis = cost_proposals._write_budget_basis(report, window_days)

    ranked_input = (
        [_namespaced(LANE_RELEARN, c) for c in relearn_candidates]
        + [_namespaced(LANE_COST, c) for c in cost_candidates]
    )
    decisions = wb.allocate_writes(ranked_input, budget, basis)

    stored: dict[str, dict[str, Any]] = {
        key: asdict(decision) for key, decision in decisions.items()
    }
    try:
        report.write_decisions = stored
    except Exception:  # noqa: BLE001 - a report type without the field
        pass

    if relearn_finding is not None:
        relearn_decisions = _lane_decisions(stored, LANE_RELEARN)
        try:
            relearn_finding.clusters = relearn_mod.apply_write_decisions(
                list(getattr(relearn_finding, "clusters", None) or []),
                relearn_decisions,
            )
        except Exception:  # noqa: BLE001 - see above
            pass
    return stored


def _lane_decisions(
    stored: dict[str, dict[str, Any]], lane: str,
) -> dict[str, wb.WriteDecision]:
    """One lane's decisions, un-namespaced back to its own keys."""
    prefix = f"{lane}:"
    out: dict[str, wb.WriteDecision] = {}
    for key, payload in (stored or {}).items():
        if not key.startswith(prefix):
            continue
        try:
            decision = _decision_from_stored(payload)
        except Exception:  # noqa: BLE001 - a malformed entry is not a verdict
            continue
        own_key = key[len(prefix):]
        out[own_key] = replace(decision, key=own_key)
    return out


def decisions_for(
    report: Any, lane: str, *, config: Any = None, window_days: float = 30.0,
    pricing_mode: str = "api",
) -> dict[str, wb.WriteDecision]:
    """One lane's verdicts for this report, allocating first if nobody has yet.

    The allocating call normally happens at the end of the report build, so
    this is a read. It falls back to allocating because a report can reach a
    consumer without ever having been through that path — a hand-built one in a
    test, or a cache written by an older build that predates the field. Doing
    the allocation late is still ONE allocation over both lanes; what it cannot
    recover is a relearn cluster whose target the OLD build already cleared,
    which drops out of the candidate set. That can only shrink the offered set,
    never grow it, which is the direction a gate is allowed to be wrong in.
    """
    stored = getattr(report, "write_decisions", None)
    if not stored:
        stored = allocate_report_writes(
            report, config=config, window_days=window_days,
            pricing_mode=pricing_mode,
        )
    return _lane_decisions(stored, lane)
