"""Helpers for driving THE write allocation from a test.

Allocation is no longer something a producer does to itself, so a test that
wants to see a budget verdict has to go through the one pass that makes them
(``core/optimize/write_allocation``). These helpers build the smallest report
that pass can allocate over — deliberately the SAME entry point production
uses, never a private shortcut, so a test cannot pass against an allocation
shape no user reaches (Critical Rule 39's second half).
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.optimize import write_allocation
from tokenjam.core.optimize.analyzers.relearn import RelearnCluster, RelearnFinding
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
from tokenjam.utils.time_parse import utcnow


def make_report(
    *, findings: dict[str, Any] | None = None, persona: str = "claude-code",
    sessions: int = 0, days: float = 30.0, active_days: int = 0,
) -> OptimizeReport:
    """A report shell carrying the given findings and nothing else."""
    now = utcnow()
    return OptimizeReport(
        window=WindowSummary(
            since=now, until=now, days=days, sessions=sessions, spans=0,
            total_tokens=0, total_cost_usd=0.0, thin_data=False,
            active_days=active_days,
        ),
        persona=persona,
        findings=dict(findings or {}),
    )


def allocate_relearn(
    clusters: list[RelearnCluster], *, summarize_finding: Any = None,
    persona: str = "claude-code",
) -> list[RelearnCluster]:
    """Allocate over a report whose only candidates are these clusters.

    Returns the clusters carrying their write verdicts. The cost lane is empty
    because the report has no cost findings — which is a real production shape
    (a scoped ``tj optimize relearn`` run), not a test-only bypass.
    """
    findings: dict[str, Any] = {"relearn": RelearnFinding(clusters=list(clusters))}
    if summarize_finding is not None:
        findings["summarize"] = summarize_finding
    report = make_report(findings=findings, persona=persona)
    write_allocation.allocate_report_writes(report)
    return list(report.findings["relearn"].clusters)
