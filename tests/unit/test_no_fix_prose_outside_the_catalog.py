"""No analyzer module may author its own user-facing fix prose.

A standing guard, not a one-off cleanup — the same shape as
``test_rulewrite_legacy``'s guard against the re-introduced ladder rung, and
for the same reason: the cleanup is worthless if the next author can undo it
without anyone noticing.

The defect this prevents is not untidiness. Every one of these shipped, past
readers who were looking, and every one of them was possible only because the
policy had no single home to be checked in:

* the identical sizing-rule contradiction lived in BOTH the subagent rubric and
  the resend right-size template, in different words, so fixing the reported one
  left the other live;
* three analyzers each authored their own wording of "delegate context-heavy
  work to a subagent", any combination of which could land in one file — and
  length plus redundancy REDUCE adherence, so writing a rule three times makes
  it less likely to be followed than writing it once;
* four call sites each wrote their own wording of the cache_control
  instruction;
* a card shipped whose fix text said no action was needed while still occupying
  an inbox slot.

The lint in ``core/fixes/lint`` catches all four classes — but only over what
is CATALOGUED. A green lint over eight records while thirty texts live
elsewhere is worse than no lint, because it reads as "the fix text is checked".
This test is what makes the lint's coverage total.

**Where the line falls.** The catalog owns DURABLE POLICY: the instruction, as
it would read for any user who hit this finding. It does not own GROUNDING —
the sentence naming this row's server, this window's session count, this
model's id. That distinction is mechanical here rather than a matter of taste:
a static string literal says the same thing to everyone and therefore belongs
in one place, while an f-string interpolating the finding is by construction
per-row.

**Three ways prose escaped this guard, all of them silent.** The guard read
green over a ~280-character multi-sentence advisory paragraph hardcoded in
``cost_proposals._summarize_to_proposals``, and it took all three to let that
through:

1. ``advise_text`` — the slot that paragraph is assigned to — was not in
   ``_FIX_SLOTS`` at all. A guard scoped by slot name sees nothing outside the
   slots it lists, and a missing slot is indistinguishable from a clean one.
2. The constant-name check matched the token ``ADVICE`` while the variable was
   named ``advise``. ADVISE vs ADVICE, one letter, and the substring test
   returned False forever. That is why this file no longer GUESSES at names:
   it traces what actually flows into a fix slot, so a near-miss spelling
   cannot get anything past it.
3. The length floor was borrowed from ``MIN_FIX_CHARS``, which answers a
   different question (is a CATALOGUED text long enough to be an instruction).
   Prose written as short interpolated fragments was therefore automatically
   exempt however much of it there was.

**Why the floor could not simply be lowered.** Length alone cannot separate the
two cases: the grounded MCP-removal sentence's longest literal run and the
shortest rule fragment are the same size, so any single threshold either misses
the rule or condemns the grounding. The second detector asks a different
question instead — total literal volume paired with SENTENCE COUNT. Grounding
is one sentence about one row; a policy runs to several, however many holes are
punched in it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tokenjam.core.fixes.lint import (
    MIN_GUARD_SENTENCES,
    MIN_GUARD_STATIC_RUN,
    MIN_GUARD_STATIC_TOTAL,
)

ROOT = Path(__file__).resolve().parents[2] / "tokenjam"
ANALYZERS = ROOT / "core" / "optimize" / "analyzers"

#: Every module that BUILDS a card, not only the analyzers. Scoping this to
#: `analyzers/` alone would have left the largest holder of fix prose out: the
#: card builders are where an analyzer's finding acquires the words a user
#: reads, and two of the three defects this migration found were there rather
#: than in an analyzer — a fourth copy of the offload rule in a one-paste
#: block, and a sentence restating the subagent rubric a paragraph below it.
_CARD_BUILDERS = (
    ROOT / "core" / "optimize" / "cost_proposals.py",
    ROOT / "core" / "optimize" / "relearn_apply.py",
    ROOT / "core" / "optimize" / "relearn_proposals.py",
    ROOT / "cli" / "cmd_optimize.py",
)

#: `core/summarize/` was entirely outside the scanned roots, which is a whole
#: package of user-facing route advice nothing checked. It is scanned as a tree
#: rather than as a file list so a module added there tomorrow is covered
#: without anyone remembering to extend anything — a hand-listed root set is how
#: this gap existed in the first place.
_SCANNED_TREES = (
    ROOT / "core" / "summarize",
)

#: Names whose value IS the fix a user is shown or writes into a file. Slot
#: names rather than a guess at prose: what makes a string a fix is where it
#: goes, not how it reads.
#:
#: ``advise_text`` is the one whose absence let a multi-sentence advisory
#: paragraph live outside the catalog while this file reported 23 passes.
_FIX_SLOTS = frozenset({
    "fix", "proposed_fix", "one_paste_fix", "artifact_text",
    "remedy_snippet", "fix_template", "suggestion",
    "advise_text", "advice_text", "recommendation", "guidance",
})

#: The only calls that may produce fix text. Reaching the catalog through any
#: of these means the text is linted; anything else means it is not.
_CATALOG_CALLS = frozenset({"fix_text", "fix_text_for", "ground", "compound_offload_fix"})

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _names_a_slot(name: str) -> bool:
    """Whether an identifier IS one of the fix slots, compared exactly.

    THE fix for the second escape. The old check asked whether an identifier
    CONTAINED any of ``("FIX", "RUBRIC", "REMEDY", "LEVER", "RULE_TEXT",
    "SNIPPET", "ADVICE")``, and a variable called ``advise`` defeated it
    permanently — ADVISE against ADVICE, one letter, no error, no warning, and
    the paragraph behind it shipped past a guard reporting 23 passes.

    Substring guessing fails in the one direction that cannot be noticed: a name
    the guess does not cover is invisible, and invisible is indistinguishable
    from clean. So there is no guessing left. A name matches only by BEING a
    slot (case- and underscore-insensitively, so ``FIX_TEMPLATE`` and
    ``fix_template`` are one name), and everything else is caught by
    :func:`_fix_carrying_locals` tracing where its value actually goes — which
    holds however the name is spelled.
    """
    return name.lower().strip("_") in _FIX_SLOTS


def _fix_carrying_locals(tree: ast.AST) -> set[str]:
    """Local names whose value ends up in a fix slot, found by TRACING.

    This replaces a substring test over constant names, which is the check that
    missed ``advise`` because it was looking for ``ADVICE``. Guessing at names
    fails silently and in the one direction that matters: a name the guess does
    not cover is invisible, and invisible is indistinguishable from clean.

    So the assignment TARGET SET is derived from where values actually go. Any
    bare ``Name`` handed to a fix slot — as a keyword argument, a dict value, or
    assigned into one — makes that name fix-carrying, and its own definition is
    then checked. Transitive, because one rename of an intermediate would
    otherwise reopen the hole.
    """
    carriers: set[str] = set()
    for _ in range(4):  # transitive closure; the real depth is 1-2
        before = len(carriers)

        def _note(value: ast.AST) -> None:
            if isinstance(value, ast.Name):
                carriers.add(value.id)
            elif isinstance(value, (ast.BinOp, ast.JoinedStr, ast.IfExp)):
                for child in ast.walk(value):
                    if isinstance(child, ast.Name):
                        carriers.add(child.id)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in _FIX_SLOTS:
                        _note(kw.value)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in _FIX_SLOTS
                    ):
                        _note(value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id in _FIX_SLOTS or target.id in carriers
                    ):
                        _note(node.value)
        if len(carriers) == before:
            break
    return carriers


def _static_text(node: ast.AST) -> tuple[int, str]:
    """``(longest contiguous literal run, all literal text joined)``.

    Both are needed. The run catches a rule quoted whole; the joined text is
    what the second detector counts sentences in, and it is what makes prose
    stitched around interpolated evidence visible at all.
    """
    longest = 0
    parts: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            stripped = child.value.strip()
            longest = max(longest, len(stripped))
            parts.append(child.value)
    return longest, "".join(parts)


def _sentences(text: str) -> int:
    return len([s for s in _SENTENCE_END.split(text) if s.strip()])


def _verdict(node: ast.AST) -> str:
    """Why this value is fix prose, or ``""`` when it is grounding.

    Two independent detectors, because one threshold cannot answer both. See
    the module docstring.
    """
    run, joined = _static_text(node)
    if run >= MIN_GUARD_STATIC_RUN:
        return f"{run} characters of hardcoded fix prose in one contiguous run"
    total = len(joined.strip())
    if total >= MIN_GUARD_STATIC_TOTAL:
        sentences = _sentences(joined)
        if sentences >= MIN_GUARD_SENTENCES:
            return (
                f"{total} characters of hardcoded fix prose across "
                f"{sentences} sentences, stitched around interpolated evidence"
            )
    return ""


def _resolves_through_the_catalog(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _CATALOG_CALLS:
                return True
    return False


def _offenders_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    carriers = _fix_carrying_locals(tree)
    out: list[str] = []

    def _is_fix_slot(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value in _FIX_SLOTS                   # {"fix": ...}
        if isinstance(node, ast.Name):
            return _names_a_slot(node.id) or node.id in carriers
        return False

    def check(slot: str, value: ast.AST) -> None:
        if _resolves_through_the_catalog(value):
            return
        verdict = _verdict(value)
        if not verdict:
            return
        out.append(
            f"{path.name}:{getattr(value, 'lineno', '?')}: {slot} carries {verdict}",
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            # AugAssign (`fix += "..."`) was never inspected at all, so a rule
            # appended to a grounded sentence was invisible however long it ran.
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_fix_slot(target):
                    check(getattr(target, "id", "?"), node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None and _is_fix_slot(key):
                    label = getattr(key, "value", None) or getattr(key, "id", "?")
                    check(f'"{label}"', value)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in _FIX_SLOTS:
                    check(f"{kw.arg}=", kw.value)
    return out


@pytest.mark.parametrize(
    "module",
    sorted(p.name for p in ANALYZERS.glob("*.py") if p.name != "__init__.py"),
)
def test_no_analyzer_defines_its_own_fix_prose(module):
    """THE guard. An analyzer names the fix it hands out; it does not author it.

    Failing this means a policy has acquired a second home, and a second
    definition of one policy is two policies that will disagree — which is not
    a prediction, it is what happened four times over. Move the text to
    ``core/fixes/registry.py`` and reference it with ``fix_text``; the record is
    then linted for every property the loose constant was never checked
    against.
    """
    offenders = _offenders_in(ANALYZERS / module)
    assert not offenders, (
        "fix prose defined outside the catalog — move it to "
        "core/fixes/registry.py and read it back with fix_text():\n  "
        + "\n  ".join(offenders)
    )


#: HOW MUCH uncatalogued prose each card builder still holds. A RATCHET, not an
#: allowlist, and the difference is the whole point.
#:
#: Closing the three escapes above made 18 per-card advisory texts in
#: ``cost_proposals.py`` visible for the first time. They were always there; the
#: guard simply could not see them, which is the worst possible state — a check
#: reporting a clean bill of health over prose it was structurally incapable of
#: reading. Migrating all of them is a real change to what a dozen cards say and
#: belongs in its own review, so what is recorded here is the SIZE of the
#: remaining gap.
#:
#: An allowlist names what is excused and quietly grows. This is asserted as an
#: EXACT count in both directions: adding one more uncatalogued text fails
#: immediately, and moving one into the catalog also fails until the number here
#: comes down with it. It can only be driven to zero, and the failure message
#: lists exactly which texts are left.
_UNCATALOGUED_RATCHET = {
    "cost_proposals.py": 18,
    "relearn_apply.py": 0,
    "relearn_proposals.py": 0,
    "cmd_optimize.py": 0,
}


@pytest.mark.parametrize("module", [p.name for p in _CARD_BUILDERS])
def test_no_card_builder_defines_its_own_fix_prose(module):
    """The same rule where the cards are actually assembled.

    An analyzer produces a finding; these modules turn it into the words a user
    reads and the block a user pastes. A rule authored here is exactly as
    unlinted as one authored in an analyzer, and harder to notice, because the
    surrounding code is legitimately full of per-row evidence prose.

    Ratcheted rather than absolute for ``cost_proposals.py`` — see
    :data:`_UNCATALOGUED_RATCHET` for why a recorded, shrinking number beats
    both a green lie and an allowlist.
    """
    path = next(p for p in _CARD_BUILDERS if p.name == module)
    offenders = _offenders_in(path)
    allowed = _UNCATALOGUED_RATCHET[module]
    assert len(offenders) == allowed, (
        f"{module} holds {len(offenders)} uncatalogued fix texts, and this "
        f"guard is pinned at {allowed}. If you ADDED one: move it to "
        f"core/fixes/registry.py and read it back with fix_text(). If you "
        f"MOVED one into the catalog: lower the number in "
        f"_UNCATALOGUED_RATCHET so the gap that is left stays honest.\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "module",
    sorted(
        str(p.relative_to(ROOT))
        for tree in _SCANNED_TREES for p in tree.rglob("*.py")
        if p.name != "__init__.py"
    ),
)
def test_no_summarize_module_defines_its_own_fix_prose(module):
    """`core/summarize/` was outside the scanned roots entirely.

    A whole package of user-facing route advice — prune, path-scope, hook,
    expire — with nothing checking any of it against the catalog. The gap was
    not that the checks were weak there; it was that they never ran there, and
    a root set nobody re-derives is exactly how that happens. This walks the
    tree, so a module added tomorrow is covered without anyone extending a list.
    """
    offenders = _offenders_in(ROOT / module)
    assert not offenders, (
        "fix prose defined outside the catalog — move it to "
        "core/fixes/registry.py and read it back with fix_text():\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_catches_prose_assigned_to_an_advise_slot():
    """THE regression test for the escape this guard was green over.

    Reproduces the exact shape: a multi-sentence advisory paragraph built in a
    local named ``advise``, then handed to ``advise_text=``. Before the fix all
    three escapes had to hold at once — the slot was unlisted, the constant-name
    check looked for ADVICE while the name was ADVISE, and the floor was
    borrowed from a different question. Any one of them being closed catches
    this, which is why all three were closed rather than the cheapest one.
    """
    source = '''
def build(files, plural):
    advise = (
        f"Review {files} oversized file{plural} in the summarize curate -> "
        "diff -> apply surface (`tj summarize list` / `tj summarize check` / "
        "`tj summarize apply`, or the Summarize screen in the web UI). This "
        "card links there instead of applying inline: the fix is a reviewed "
        "rewrite - structure kept verbatim, prose compressed, one file at a "
        "time - not a one-click removal."
    )
    return CostProposal(advise_text=advise)
'''
    tmp = ANALYZERS.parent / "_guard_probe_advise.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        offenders = _offenders_in(tmp)
        assert offenders, "the guard still cannot see prose behind an advise slot"
        assert "advise" in offenders[0]
    finally:
        tmp.unlink()


def test_the_guard_catches_a_rule_stitched_out_of_short_fragments():
    """The length floor's own regression test.

    No fragment here reaches the single-run floor, so the old guard exempted it
    automatically. It is still a policy: several sentences of durable
    instruction with a couple of interpolations dropped in.
    """
    source = '''
def build(name, n):
    fix = (
        f"Remove `{name}` first. "
        f"Then re-run the scan, {n} times if needed. "
        "Do not re-add it without a reason. "
        "Record that reason in the file itself. "
        "Prefer scoping over deleting when unsure."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_fragments.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "short fragments are still automatically exempt"
    finally:
        tmp.unlink()


def test_the_guard_catches_prose_appended_with_augmented_assignment():
    """``fix += "..."`` was never inspected at all."""
    source = '''
def build(name):
    fix = f"Remove `{name}`."
    fix += (
        "Removing only this location stops the tax for the sessions counted "
        "here. The remaining sessions need their own location edited too."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_augassign.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "an appended rule is still invisible"
    finally:
        tmp.unlink()


def test_a_near_miss_constant_spelling_cannot_escape():
    """The ADVISE/ADVICE hole, pinned.

    The old check asked whether a name CONTAINED one of a handful of guessed
    tokens. One letter defeated it, permanently and silently. The replacement
    traces what reaches a fix slot, so the name can be anything at all.
    """
    source = '''
def build():
    wibble = (
        "Right-size the workers you dispatch: default every one of them to "
        "the cheapest same-family model that fits the shape of its task."
    )
    return CostProposal(advise_text=wibble)
'''
    tmp = ANALYZERS.parent / "_guard_probe_wibble.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "a name the guard did not guess still escapes"
    finally:
        tmp.unlink()


def test_the_guard_actually_catches_a_reintroduced_constant():
    """The guard's own regression test.

    A structural check that cannot fail is indistinguishable from one that
    passes, and this one is easy to write in a way that never fires — every
    predicate in it is a judgement about what counts as fix prose. So it is
    pointed at a module that reintroduces the defect, and must report it.
    """
    source = '''
FIX_TEMPLATE = (
    "Right-size the workers you dispatch: default every one of them to the "
    "cheapest same-family model that fits the shape of its task."
)
'''
    tmp = ANALYZERS.parent / "_guard_probe.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp), "the guard cannot see a reintroduced constant"
    finally:
        tmp.unlink()


def test_a_grounded_sentence_is_not_mistaken_for_a_rule():
    """The other half, and the more important one to get right.

    A false positive here is worse than a gap: it teaches the next author to
    reach for an exception rather than to move the text, and an exception in a
    structural guard is permanent. A sentence built around this row's evidence
    is grounding — it belongs at the render site, and it must pass.
    """
    source = '''
def build(server, sessions):
    fix = (
        f"Remove the `{server.name}` MCP server ({server.source}); "
        f"zero tool calls across {sessions} session(s) in this window."
    )
    return fix
'''
    tmp = ANALYZERS.parent / "_guard_probe_grounded.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        assert _offenders_in(tmp) == []
    finally:
        tmp.unlink()
