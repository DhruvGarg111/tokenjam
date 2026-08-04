"""THE registry of values that must have exactly one derivation.

**The defect class.** A published number, window or classification gets
computed independently in two places, nothing forces the two to agree, and
they drift. It has shipped roughly ten times in this product, always fixed by
hand with a bespoke test written after the fact: the rollup population, the
write budget, the report window, the persona gate, the write-apply target.
Each fix was correct and each one left the general shape of the defect
unguarded, because nothing forced the NEXT shared value through the same
discipline.

**The fix is one registry, not one test per value.** :data:`SEAMS` names every
value whose derivation is pinned to a SINGLE module, by the symbol a second
derivation would necessarily touch (a function call, or an attribute read).
``tests/unit/test_single_derivation.py`` walks the whole package once per
entry and fails if that symbol is reached from anywhere outside the module
that owns it. Adding the next shared value is a new :class:`SingleSeam` line
here — never a new AST walk, never a new test file. That is the entire point:
a design that needs a new test per value has rebuilt the problem it exists to
retire.

**Not every seam fits a symbol guard**, and forcing one is worse than not
having one — see the module docstring on :data:`BESPOKE_SEAMS` below for why
persona classification, the write-apply target and the scan-cycle anchor stay
as hand-written tests instead. This registry still names them, so a reviewer
scanning ONE file sees every pinned seam, mechanized or not, and
:func:`check_bespoke_seam` fails loudly if the test guarding one of them is
ever deleted.

**Aggregate-versus-parts is a different shape of the same defect** — see the
module docstring further down, near :data:`KNOWN_GAPS`.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
from dataclasses import dataclass

import tokenjam as _pkg

#: Every module under here is in scope for a seam's offender walk. Test files
#: are exempt everywhere in this module — they legitimately construct or pin
#: the raw guarded symbol's own behaviour (see any seam's own unit test), and
#: this walk only ever inspects the SHIPPED package.
PACKAGE_ROOT = pathlib.Path(_pkg.__file__).parent


@dataclass(frozen=True)
class SingleSeam:
    """One value whose derivation may live in exactly one module.

    ``symbol`` is the bare identifier a second derivation would have to
    touch — a function name for ``kind="call"``, or an attribute name for
    ``kind="attr"`` (matched both as ``obj.symbol`` and as the string literal
    key of ``getattr(obj, "symbol", ...)``, since config reads use both
    forms in this codebase).

    ``allowed_modules`` are paths relative to :data:`PACKAGE_ROOT`, POSIX
    style (``"core/optimize/report_window.py"``). The symbol's OWN defining
    module always needs to be listed if it also USES the symbol internally
    (e.g. a factory function calling its own class constructor).
    """

    name: str
    description: str
    symbol: str
    kind: str  # "call" | "attr"
    allowed_modules: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in ("call", "attr"):
            raise ValueError(f"unknown SingleSeam.kind {self.kind!r} for {self.name!r}")


def _calls_symbol(node: ast.AST, symbol: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    return name == symbol


def _reads_attr(node: ast.AST, symbol: str) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr == symbol
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
        args = node.args
        if len(args) >= 2 and isinstance(args[1], ast.Constant):
            return args[1].value == symbol
    return False


def offenders_for(seam: SingleSeam) -> list[str]:
    """Every ``path:lineno`` outside ``seam.allowed_modules`` that reaches
    ``seam.symbol``. Empty means the seam holds.

    Walks every ``.py`` under the shipped package, in file order, so the
    result is deterministic and reviewable — a caller can paste it straight
    into a failure message.
    """
    check = _calls_symbol if seam.kind == "call" else _reads_attr
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in seam.allowed_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if check(node, seam.symbol):
                offenders.append(f"{rel}:{node.lineno}")
    return offenders


#: --------------------------------------------------------------------- #
#: THE REGISTRY. Add a shared value here, never as a new test file.
#: --------------------------------------------------------------------- #
SEAMS: tuple[SingleSeam, ...] = (
    SingleSeam(
        name="rollup population",
        description=(
            "the Review inbox headline: every open cost proposal PLUS "
            "relearn's open clusters, summed once."
        ),
        symbol="past_overspend_rollup",
        kind="call",
        allowed_modules=frozenset({"core/optimize/inbox_contribution.py"}),
        reason=(
            "completeness used to be a caller CONVENTION — whoever built the "
            "input list had to remember to concatenate relearn's clusters in. "
            "cmd_quickstart's first-run screen forgot, and its own comment "
            "asserted the two totals could never disagree. "
            "gather_rollup_population() in inbox_contribution.py is now the "
            "only path that assembles both feeds before calling the raw "
            "rollup function."
        ),
    ),
    SingleSeam(
        name="write allocation point",
        description=(
            "the permanent-write budget every report may spend, across "
            "BOTH producers (relearn clusters, cost-proposal write cards)."
        ),
        symbol="build_write_budget",
        kind="call",
        allowed_modules=frozenset({"core/optimize/write_allocation.py"}),
        reason=(
            "relearn and cost_proposals used to each build their OWN "
            "WriteBudget from their own constants and spend it inside "
            "themselves — the bound a user actually faced was the SUM of "
            "both caps, and ranking was per-producer so a high-net cost "
            "write and a low-net relearn write never entered one ranked "
            "list. write_allocation.allocate_report_writes() is now the "
            "one place a budget is built and spent, over the union of both "
            "producers' candidates."
        ),
    ),
    SingleSeam(
        name="window length",
        description=(
            "the trailing look-back, in days, EVERY past-overspend surface "
            "observes over — the Dashboard tiles and the Review inbox "
            "headline must resolve the same number."
        ),
        symbol="scan_window_days",
        kind="attr",
        allowed_modules=frozenset({
            "core/optimize/report_window.py",
            "core/config.py",
        }),
        reason=(
            "the Dashboard read `[optimize] scan_window_days` directly (a "
            "fixed config int) while the Review inbox read the resolved "
            "analysis span, bounded by measured history — 30 vs 69 days on "
            "a real corpus, so the six Dashboard tiles summed to roughly "
            "half the inbox headline. report_window.report_window_days() "
            "is now the only reader of the raw config field; core/config.py "
            "stays allowed because it OWNS the field's definition and "
            "default."
        ),
    ),
    SingleSeam(
        name="rate profile",
        description=(
            "the blended $/token rate an analyzer prices its findings "
            "against."
        ),
        symbol="RateProfile",
        kind="call",
        allowed_modules=frozenset({"core/optimize/rate_profile.py"}),
        reason=(
            "RateProfile is a plain dataclass — nothing stops a second "
            "constructor call from hand-rolling a rate outside "
            "blended_rate_profile()'s own weighting. Every analyzer that "
            "prices tokens (relearn, summarize, downsize) calls the shared "
            "function; only rate_profile.py itself is allowed to construct "
            "the record it returns."
        ),
    ),
)


@dataclass(frozen=True)
class BespokeSeam:
    """A single-derivation invariant real enough to pin, but NOT expressible
    as a symbol-reachable-from-one-module check.

    Each of these was tried against :class:`SingleSeam` first and rejected
    for a concrete, stated reason — never "harder to write". Listing them
    here (rather than leaving them as orphaned tests nobody indexes) means a
    reviewer scanning this one file sees every pinned single-derivation
    invariant in the product, not just the mechanized ones, and
    :func:`check_bespoke_seam` fails loudly — not silently — if the test
    naming a seam here is ever deleted or renamed out from under it.
    """

    name: str
    description: str
    reason_not_mechanized: str
    test_module: str
    test_name: str


BESPOKE_SEAMS: tuple[BespokeSeam, ...] = (
    BespokeSeam(
        name="persona classification",
        description=(
            "a persona gate (which analyzers run, whether relearn may "
            "write) may never be resolved over ALL history."
        ),
        reason_not_mechanized=(
            "the defect is in the CALL ARGUMENTS (an unwindowed "
            "agent_persona_mix() reaching dominant_persona()), not in which "
            "module makes the call — a symbol-reachability guard can't see "
            "into a call's own arguments, only whether the call exists."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_no_surface_classifies_persona_over_all_history",
    ),
    BespokeSeam(
        name="write-apply target",
        description=(
            "the path relearn suggests as a write target, and the path the "
            "API's write guard authorizes against, resolve through the "
            "SAME call: resolve_write_scope(scope=scope).suggest_root."
        ),
        reason_not_mechanized=(
            "scope.claude_home has a legitimate SECOND, unrelated purpose "
            "(deadweight's own read-only MCP-config scope) — a symbol guard "
            "on `.claude_home` would false-positive on that call. The pin "
            "has to be scoped to the two write-target call sites "
            "specifically, which only a source-text match on the exact "
            "call shape can do."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_the_apply_target_and_the_write_guard_share_one_derivation",
    ),
    BespokeSeam(
        name="scan-cycle anchor",
        description=(
            "one `utcnow()` timestamp per scan cycle, threaded into BOTH "
            "the report pass and the cost-proposal pass, so they measure "
            "the same instant instead of two instants seconds apart."
        ),
        reason_not_mechanized=(
            "report_store.recompute_now and cost_proposals."
            "recompute_cost_proposals BOTH legitimately fall back to their "
            "own utcnow() when called standalone outside a cycle (`until` "
            "is optional by design) — the invariant is that scan_cycle "
            "threads ONE value through both calls in the same cycle, which "
            "is a data-flow property a reachability guard cannot express."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_the_report_and_cost_stores_come_from_ONE_analyzer_pass",
    ),
)


def check_bespoke_seam(seam: BespokeSeam) -> str | None:
    """``None`` if the test pinning ``seam`` still exists and is callable;
    otherwise a string explaining what went missing."""
    try:
        module = importlib.import_module(seam.test_module)
    except ImportError as exc:
        return f"{seam.test_module} failed to import: {exc}"
    fn = getattr(module, seam.test_name, None)
    if fn is None or not callable(fn):
        return f"{seam.test_module}.{seam.test_name} no longer exists"
    return None


#: --------------------------------------------------------------------- #
#: AGGREGATE VERSUS PARTS.
#:
#: A DIFFERENT shape of the same defect class: not one value derived twice,
#: but one FINDING fanned out into several proposals whose figures a surface
#: publishes as parts, next to (or instead of) an aggregate figure the
#: finding itself carries — with nothing forcing the parts to sum to the
#: whole, or disclosing it when they don't.
#:
#: The `cache` family holds this invariant BY CONSTRUCTION today:
#: `_cache_to_proposals` subtracts whatever the per-agent root-cause cards
#: (`_per_agent_cache_recoverable_by_model`) already claimed for the same
#: (provider, model) before it surfaces the generic row, so the family's
#: cards sum EXACTLY to the finding's own `past_overspend_usd` — pinned in
#: `tests/unit/test_single_derivation.py::
#: test_the_cache_family_sums_exactly_to_the_findings_own_total`.
#:
#: `downsize` does NOT hold it. When a finding carries `per_agent` rows,
#: `_downsize_to_proposal` drops the window-wide card and emits the
#: driver-role card plus one card per agent — but the per-agent rows come
#: from `build_agent_price_rows`, a SEPARATE per-(agent, provider, model)
#: repricing that silently DROPS any group whose model has no pricing data
#: or no dated candidate. `finding.past_overspend_usd` is unaffected by that
#: drop (it is `savings_window + driver_savings`, computed window-wide over
#: EVERY candidate session, priced or not). So the Dashboard tile
#: (`api/routes/cost.py::_collect_recoverable`, which reads
#: `report.downgrade.past_overspend_usd` straight off the finding) and the
#: Review inbox (the driver card plus the per-agent cards) can name
#: arbitrarily different totals for the SAME analyzer, with no disclosure
#: that the per-agent cards are a partial accounting.
#:
#: Measured directly (see the module docstring on
#: `test_the_downsize_per_agent_path_can_undercount_the_findings_own_total`):
#: an aggregate of $998.00 (one candidate's model carries no pricing data)
#: surfaced as $0.11 across the per-agent cards the inbox actually shows,
#: with nothing on either card naming the other $997.89.
#:
#: THIS IS NOT FIXED HERE. Choosing how to close it — fall back to the
#: window-wide card when the per-agent total falls materially short,
#: disclose the gap on each per-agent card the way the cache family's
#: `estimate_basis` does, or reprice every dropped group at a default rate —
#: is a product decision about what downsize/`build_agent_price_rows` should
#: do, not a mechanical guard. `test_the_downsize_per_agent_path_can_
#: undercount_the_findings_own_total` pins the CURRENT behaviour as an
#: xfail(strict=True): it exists to fail LOUDLY the moment someone closes
#: this gap (an unexpected pass forces the xfail's removal, which is the
#: signal to also delete this docstring's account of it) and to make the gap
#: impossible to silently reintroduce a different way in the meantime.
#: --------------------------------------------------------------------- #
KNOWN_GAPS = (
    "downsize: the per-agent card path can undercount the finding's own "
    "past_overspend_usd with no disclosure — see the module docstring "
    "above this constant, and "
    "test_the_downsize_per_agent_path_can_undercount_the_findings_own_total "
    "in tests/unit/test_single_derivation.py.",
)


__all__ = [
    "PACKAGE_ROOT",
    "SEAMS",
    "BESPOKE_SEAMS",
    "KNOWN_GAPS",
    "SingleSeam",
    "BespokeSeam",
    "check_bespoke_seam",
    "offenders_for",
]
