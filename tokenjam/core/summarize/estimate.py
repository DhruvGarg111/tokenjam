"""Per-candidate savings estimate for the summarize surface.

Honesty discipline: this estimates the **per-call token reduction** of
summarizing a static prompt's prose. It needs no telemetry — tokens are derived
from the file. The saving amortizes across every reuse of the (cached) prompt.

Dollar figures are deliberately NOT fabricated here: a per-call dollar amount at
default rates is noise, and the meaningful *amortized* figure needs a real
call-count (telemetry), which the optional usage-ranked path can add later. The
token figure is the one we can stand behind from the file alone.

**The ratio is an ASK, not a prediction — and the two must not be confused.**
`DEFAULT_TARGET_RATIO` is what `session.prepare` puts in the rewriter's prompt
("summarize this to N words"). There is no retry loop and no gate on hitting it:
whatever the model returns is accepted so long as the structure markers survive,
and rewrites observed while building this delivered materially less than the
target asks for. Using the ask as the estimate therefore overstates, so this
module separates the two:

* :data:`DEFAULT_TARGET_RATIO` — what the rewriter is asked for.
* :func:`observed_prose_ratio` — what rewrites on THIS machine have actually
  delivered, derived from staged results (`session.CheckVerdict`), or ``None``
  when there is no credible sample.

A caller that wants a defensible number asks for the observed ratio first and
falls back to the target only while saying so (see the summarize analyzer's
`estimate_basis`). Nothing here invents a middle number: a ratio that is neither
the documented ask nor a measured outcome would be the least defensible of the
three.

**Waiting for evidence is not the same as having none.** The observed ratio used
to appear only if a user happened to rewrite files on their own; `tj summarize
calibrate` (`core/summarize/calibrate`) now runs the real rewrite over a bounded,
explicitly-consented sample so the measurement can exist on demand. It writes
through the same `session.check` gate into the same staging dir this module
reads, so there is exactly one store and one definition of a sample.

**A ratio is also the wrong SHAPE for the goal on an always-resident file.**
Anthropic publishes a concrete target for that artifact — :data:`PUBLISHED_LINE_TARGET`
— and a huge file halved is still far past the point where adherence degrades.
:func:`line_target_prose_words` turns that target into the word budget the
rewriter is ASKED for. It deliberately does not touch the CLAIM: what a rewrite
is asked for and what it delivers are the two things this module exists to keep
apart, so the estimate stays bounded by :func:`observed_prose_ratio` (or the
target ratio) no matter how aggressive the line goal is.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN, StructureBreakdown, line_breakdown

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

logger = logging.getLogger(__name__)

#: Anthropic's published size target for an always-resident instruction file,
#: quoted below. Ours is the *application* of it to every ALWAYS-class file
#: (`.claude/rules/*.md` and `AGENTS.md` load the same way a `CLAUDE.md` does);
#: the number itself is theirs, which is why every surface that states it also
#: states where it came from rather than presenting it as a tokenjam threshold.
PUBLISHED_LINE_TARGET = 200
PUBLISHED_LINE_TARGET_SOURCE = "https://code.claude.com/docs/en/memory"
PUBLISHED_LINE_TARGET_QUOTE = (
    "target under 200 lines per CLAUDE.md file. Longer files consume more "
    "context and reduce adherence."
)
#: Anthropic's own suggested alternative to compressing an oversized instruction
#: file, quoted so the advice that mentions it cannot drift from what they say.
#: Summarize only ever MENTIONS this; it does not write path-scoped rules.
PATH_SCOPED_RULES_QUOTE = (
    "If your instructions are growing large, use path-scoped rules so "
    "instructions load only when Claude works with matching files."
)

#: What the rewriter is INSTRUCTED to hit — `session.prepare` turns this into
#: the word count in the model's prompt. It is an ask, not a prediction, and
#: **nothing enforces it**: there is no retry and no gate on the achieved word
#: count, since `check` gates on structure surviving and nothing else. So do not
#: tune this as though it were a measurement of what rewrites deliver — changing
#: it changes what the model is ASKED for. The measured counterpart is
#: :func:`observed_prose_ratio`, which supersedes this the moment real staged
#: results exist.
DEFAULT_TARGET_RATIO = 0.5

#: Minimum evidence before an observed ratio may replace the target. Both gates
#: must pass: too few files, or too little prose, and one unusual rewrite would
#: set the number for every file the user owns.
MIN_OBSERVED_SAMPLES = 3
MIN_OBSERVED_PROSE_WORDS = 500


def tokens_saved(breakdown: StructureBreakdown, ratio: float = DEFAULT_TARGET_RATIO) -> int:
    """Estimated tokens removed per call by summarizing the prose to ``ratio``.

    Protected structure (fenced code/JSON, tables, tags, inline code, templates) is preserved
    verbatim and never counted as savings — only prose shrinks. Returns 0 when ``ratio >= 1``.
    """
    if ratio >= 1.0:
        return 0
    prose_tokens = breakdown.prose_chars / CHARS_PER_TOKEN
    return max(0, int(prose_tokens * (1.0 - ratio)))


def observed_prose_ratio(config: "TjConfig | None") -> tuple[float | None, int]:
    """The prose ratio rewrites have ACTUALLY delivered here, and the sample size.

    Returns ``(None, n)`` when the sample is too small to be credible — the
    caller must then fall back to the target ratio and disclose that it is an
    ask rather than a measurement. Never guesses: no staged results means no
    observed ratio, not a zero and not a nudged constant.

    Derivation. `session.check` restores the rewrite with its structure put back
    verbatim, so every word the file lost was a prose word: ``words_before -
    words_after`` is prose removed, and ``prose_words_before`` is what it started
    with. Summing both across the sample before dividing weights the ratio by
    volume, so a handful of tiny files cannot outvote a large one. Only staged
    (structure-passing) results count — a rewrite that failed the structure gate
    was never a usable outcome, and letting it into the sample would measure the
    rewriter's failures as if the user had accepted them.

    Never raises: an unreadable or half-written staged result is skipped.
    """
    if config is None:
        return None, 0
    from tokenjam.core.summarize.session import list_staged

    try:
        staged = list_staged(config)
    except Exception:
        logger.debug("summarize estimate: could not read staged results", exc_info=True)
        return None, 0

    samples = 0
    prose_before_total = 0
    removed_total = 0
    for record in staged:
        if not isinstance(record, dict) or not record.get("staged"):
            continue
        prose_before = int(record.get("prose_words_before") or 0)
        before = int(record.get("words_before") or 0)
        after = int(record.get("words_after") or 0)
        # A result staged before `prose_words_before` existed carries no
        # denominator; it is not evidence, so it is skipped rather than
        # counted as a zero-prose file.
        if prose_before <= 0 or before <= 0:
            continue
        samples += 1
        prose_before_total += prose_before
        removed_total += max(0, before - after)

    if samples < MIN_OBSERVED_SAMPLES or prose_before_total < MIN_OBSERVED_PROSE_WORDS:
        return None, samples
    # Clamped to [0, 1]: a rewrite that somehow grew the file is a 1.0 (no
    # reduction), never a negative saving.
    ratio = 1.0 - (removed_total / prose_before_total)
    return min(1.0, max(0.0, ratio)), samples


def gate_failed_attempts(config: "TjConfig | None") -> int:
    """How many rewrites here were attempted and FAILED the structure gate.

    A failed rewrite is deliberately excluded from :func:`observed_prose_ratio`
    (it was never a usable outcome), but it is not nothing: a file the rewriter
    cannot compress without mangling its structure is a real finding about that
    file, and dropping the attempt silently would let a run that produced only
    failures look identical to a run that never happened. Reported alongside the
    ratio so the basis can say what the sample cost as well as what it showed.

    Never raises: an unreadable attempt record is skipped.
    """
    if config is None:
        return 0
    from tokenjam.core.summarize.session import list_attempts

    try:
        return len(list_attempts(config))
    except Exception:
        logger.debug("summarize estimate: could not read attempt records", exc_info=True)
        return 0


def line_target_prose_words(
    *,
    text: str,
    load_class: str,
    prose_words: int,
    line_target: int = PUBLISHED_LINE_TARGET,
) -> int | None:
    """Prose-word budget that would bring ``text`` under ``line_target`` lines.

    ``None`` when the target does not apply, and the three reasons are all real
    rather than error cases: the file is not always-resident (the published
    guidance is about a file that loads every session), it is already under the
    target, or its protected structure ALONE exceeds the target — in which case
    summarizing cannot get there and pretending otherwise would be an ask the
    fix structurally cannot deliver.

    This is an ASK. It never widens a saving: the caller takes whichever of this
    and the ratio budget is smaller for the rewriter's prompt, and prices the
    result off the ratio regardless (see the module docstring).
    """
    if load_class != load_semantics.ALWAYS or prose_words <= 0:
        return None
    lines = line_breakdown(text)
    if lines.total_lines <= line_target or lines.prose_lines <= 0:
        return None
    allowance = line_target - lines.protected_lines
    if allowance <= 0:
        return None
    return max(8, int(prose_words * (allowance / lines.prose_lines)))
