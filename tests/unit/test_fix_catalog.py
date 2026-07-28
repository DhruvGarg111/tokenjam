"""The fix catalog and the lint that keeps its texts honest.

Every property here corresponds to a defect that ALREADY SHIPPED, twice each,
past readers who were looking. That is the argument for a lint rather than a
review checklist: none of these are subtle once stated, they are just invisible
when a policy has no single home to be checked in.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tokenjam.core.fixes import (
    FIX_CATALOG,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    fix_for,
    fix_text,
    fixes_for,
    lint_catalog,
    lint_fix,
)


def _record(**kw) -> FixRecord:
    base = dict(
        key="test.fix",
        text="Default every worker to the cheapest model that fits its shape.",
        delivery="claude_md_rule",
        personas=frozenset({PERSONA_CLAUDE_CODE}),
        analyzers=frozenset({"subagent"}),
        answers="an observation",
    )
    base.update(kw)
    return FixRecord(**base)   # type: ignore[arg-type]


# --- the catalog holds --------------------------------------------------------#

def test_every_catalogued_fix_passes_its_own_lint():
    """The gate. A shipped fix that violates a published property is exactly
    the class of defect this catalog was built after."""
    assert lint_catalog() == {}


def test_a_fix_is_defined_once_and_a_duplicate_key_is_refused():
    """Two copies of a policy is two policies — which is how the identical
    sizing contradiction shipped in the subagent rubric AND the resend
    template in different words, and was fixed in only one of them."""
    from tokenjam.core.fixes.catalog import register

    with pytest.raises(ValueError, match="duplicate fix key"):
        register(_record(key="subagent.sizing_rubric"))


def test_the_analyzer_constants_read_from_the_catalog_rather_than_redefining_it():
    """The two constants that drifted now resolve THROUGH the catalog, so a
    correction lands in both places or neither."""
    from tokenjam.core.optimize.analyzers.context_resend import RIGHTSIZE_FIX_TEMPLATE
    from tokenjam.core.optimize.cost_proposals import SUBAGENT_RUBRIC_INTRO

    assert SUBAGENT_RUBRIC_INTRO == fix_text("subagent.sizing_rubric")
    assert RIGHTSIZE_FIX_TEMPLATE == fix_text("resend.rightsize_worker")


def test_a_missing_key_raises_rather_than_returning_empty_text():
    """A card with an empty fix is the same quiet failure as a frontmatter key
    the harness ignores: it looks fine and does nothing."""
    with pytest.raises(KeyError):
        fix_text("nope.not.a.fix")


# --- the persona dimension ----------------------------------------------------#

def test_a_fix_is_only_offered_to_a_persona_that_can_act_on_it():
    """What a user can edit differs completely by persona. A Claude Code user
    edits instruction files, agent files and hooks; an SDK service edits
    routing code and redeploys. Handing either the other's fix names an action
    they cannot take."""
    cc = {r.key for r in fixes_for("resend", PERSONA_CLAUDE_CODE)}
    sdk = {r.key for r in fixes_for("resend", PERSONA_SDK)}
    assert "resend.offload_to_subagent" in cc
    assert "resend.offload_to_subagent" not in sdk
    assert "resend.sdk_cache_breakpoint" in sdk
    assert "resend.sdk_cache_breakpoint" not in cc


def test_model_and_effort_are_separate_levers():
    """They answer different diagnoses and neither substitutes for the other:
    a worker that did not know enough needs a bigger model, one that did not
    try hard enough needs more effort. Collapsing them loses half the remedy."""
    from tokenjam.core.fixes import LEVER_EFFORT, LEVER_MODEL

    levers = {r.key: r.lever for r in fixes_for("subagent")}
    assert levers["subagent.sizing_rubric"] == LEVER_MODEL
    assert levers["subagent.pin_effort"] == LEVER_EFFORT


# --- the lint's own properties ------------------------------------------------#

def test_the_lint_catches_a_fix_that_relicenses_what_its_analyzer_bills_for():
    """THE load-bearing check. `past_overspend_usd` is the maximum its analyzer
    knows: applying the fix should ERASE it. A fix that gives a pass to the
    exact shape the analyzer flagged leaves the number where it was."""
    rubric = fix_for("subagent.sizing_rubric")
    assert rubric is not None
    reopened = replace(
        rubric,
        text=rubric.text + " A subagent that does little tool work rarely "
                           "needs the premium tier.",
    )
    problems = lint_fix(reopened)
    assert any("re-licenses the behaviour" in p for p in problems)
    assert any("little tool work" in p for p in problems)


def test_the_lint_catches_a_self_graded_escape_hatch():
    """"Unless the subtask genuinely needs deep reasoning" asks the agent to
    rate its own task's difficulty, and an agent asked that answers yes — so
    the exception swallows the rule."""
    for hatch in (
        "Use the cheap model unless the task genuinely needs deep reasoning.",
        "Route down, except when the subtask truly requires a larger model.",
        "Prefer the small model unless the work needs deep thinking.",
    ):
        problems = lint_fix(_record(text=hatch))
        assert any("grades itself" in p for p in problems), hatch


def test_the_lint_catches_a_fix_that_says_to_do_nothing_but_is_offered():
    """A card whose fix text says there is nothing to do must not occupy an
    apply slot. The observation still stands — only the offer is withdrawn."""
    problems = lint_fix(_record(
        text="The harness already handles this, so no rule or hook is needed.",
        advisory_only=False,
    ))
    assert any("advisory_only" in p for p in problems)
    # And marked correctly, it passes.
    assert lint_fix(_record(
        text="The harness already handles this, so no rule or hook is needed.",
        advisory_only=True,
    )) == []


def test_the_lint_catches_a_label_masquerading_as_a_fix():
    assert any("too short" in p for p in lint_fix(_record(text="Use haiku.")))


def test_the_lint_catches_instruction_text_past_the_published_ceiling():
    from tokenjam.core.fixes import MAX_FIX_LINES

    long_text = "\n".join(["Route context-heavy work to a worker."] * (MAX_FIX_LINES + 1))
    assert any("exceeds the" in p for p in lint_fix(_record(text=long_text)))


def test_the_lint_says_escalate_to_a_hook_for_fixed_point_behaviour():
    """Text that must run at a fixed point is a hook's job. Delivered as a
    rule it is a request the agent may or may not honour at that instant."""
    problems = lint_fix(_record(
        text="Before every Bash call, check the working directory is the repo root.",
    ))
    assert any("Escalate to a hook" in p for p in problems)


def test_the_no_action_pattern_tolerates_a_list_of_nouns():
    """Regression on the lint itself. Its first draft matched only a single
    noun, so "no rule or hook is needed" — the natural phrasing — reported a
    correctly-marked advisory record as mis-marked. A false positive teaches
    the next author to reach for an exception rather than fix the text."""
    assert lint_fix(_record(
        text="The harness already blocks this, so no rule or hook is needed here.",
        advisory_only=True,
    )) == []


def test_the_catalog_covers_the_analyzers_that_write_rules():
    for analyzer in ("subagent", "resend", "relearn"):
        assert fixes_for(analyzer), analyzer
    assert len(FIX_CATALOG) >= 7
