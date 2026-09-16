"""What the window's sessions shipped, and what the ones that shipped nothing
cost: the `shipped` analyzer (shipped-value ledger, contracts §1 and §4).

A MEASURE ROI finding, not a cost-saving one. It publishes no
`past_overspend_*` field and is absent from `cost_proposals.COST_ANALYZERS`
on purpose: the unshipped figure is measured spend on sessions that left no
commit, which is a fact about output, not recoverable waste. Pulling it into
the recoverable rollup would price research, review and ops sessions as
money to claw back (Critical Rules 27 and 30 in `tokenjam/CLAUDE.md`), and
the caveat carried as a dataclass default says exactly that on every surface
(the `MODEL_DOWNGRADE_CAVEAT` device, Critical Rule 14).

Runs no git: it reads the tables `core/shipped.match_sessions_to_commits`
keeps current (`session_commits`, `repo_commits`, `commit_file_stats`), which
is why it can be part of a stored report and served from it. The matcher
runs earlier in the same daemon pass (`scan_cycle`) and before a direct-DB
`tj optimize`, so the finding is at most one pass behind the repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.shipped import SHIPPED_CAVEAT, shipped_summary


@dataclass
class ShippedFinding:
    """Contracts §4 `ShippedFinding`, plus the two disclosures the OSS form
    needs: `sessions_committed` (a joined commit exists but is not on the
    default branch, its own state rather than "shipped") and
    `sessions_no_repo` (never analysed; Critical Rule 30's "what was NOT
    analysed" in a number rather than a sentence)."""
    window_days:         float = 0.0
    sessions_total:      int = 0
    sessions_shipped:    int = 0
    sessions_unshipped:  int = 0
    sessions_committed:  int = 0
    sessions_no_repo:    int = 0
    cost_shipped_usd:    float = 0.0        # measured
    cost_unshipped_usd:  float = 0.0        # measured
    cost_committed_usd:  float = 0.0        # measured
    cost_rework_usd:     float | None = None   # measured; None = no evidence yet
    cost_loop_usd:       float | None = None   # measured; None = no tool spans
    coverage:            float | None = None   # contracts §1; None = nothing to cover
    commits_joined:      int = 0
    commits_on_default:  int = 0
    top_unshipped:       list[dict[str, Any]] = field(default_factory=list)
    top_reworked:        list[dict[str, Any]] = field(default_factory=list)
    rework_basis:        str = ""
    loop_basis:          str = ""
    caveat:              str = SHIPPED_CAVEAT


@register("shipped")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ShippedFinding to ctx.report.findings."""
    summary = shipped_summary(
        ctx.conn, ctx.since, ctx.until,
        agent_id=ctx.agent_id, persona_scope=ctx.persona_scope,
    )
    ctx.report.findings["shipped"] = ShippedFinding(**summary.to_dict())
