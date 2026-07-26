"""Failure-family coverage, the not-a-relearn filter, and the relearn/resend line.

Every error string in here is REAL wording taken from the local Claude Code
corpus (2026-07-26), not invented — a family that matches a plausible-looking
paraphrase but not the harness's actual message is the exact failure mode the
`read_offset_malformed` ordering comment already documents.
"""
from __future__ import annotations

import pytest

from tokenjam.core.optimize.analyzers.relearn import (
    _KNOWN_FAMILIES,
    classify_known_family,
    is_user_decline,
)
from tokenjam.core.optimize.analyzers.resend_tail import RELEARN_RESEND_BOUNDARY

# (tool_name, real error text, expected family)
_REAL_CORPUS_SAMPLES = [
    ("gen_ai.llm.call", "prompt is too long: 201070 tokens > 200000 maximum",
     "context_overflow"),
    ("Bash", "Exit code 143\nCommand timed out after 2m 0s", "bash_timeout"),
    ("Bash", "Exit code 128\nfatal: a branch named 'ticket-234' already exists",
     "git_branch_exists"),
    ("Bash", "This Bash command contains multiple operations. The following part "
             "requires approval: git push", "bash_chained_approval"),
    ("Bash", "Exit code 1\nuv not found\n/Users/x/.local/bin/uv", "command_not_found"),
    ("Bash", "Exit code 127\n(eval):1: command not found: aws", "command_not_found"),
    ("Read", "EISDIR: illegal operation on a directory, read '/Users/x/skills'",
     "read_directory"),
    ("Read", "File content (34827 tokens) exceeds maximum allowed tokens (25000). "
             "Use offset and limit parameters to read specific portions.",
     "read_too_large"),
    ("Edit", "Found 2 matches of the string to replace, but replace_all is false. "
             "To replace all occurrences, set replace_all to true.",
     "edit_ambiguous_match"),
    ("gen_ai.llm.call", "The following domains are not accessible to our user "
                        "agent: ['reddit.com']. Read more: https://example",
     "webfetch_domain_blocked"),
    ("WebFetch", "Claude Code is unable to fetch from web.archive.org",
     "webfetch_domain_blocked"),
    ("ExitPlanMode", "Error: No such tool available: ExitPlanMode. ExitPlanMode "
                     "exists but is not enabled in this context.",
     "deferred_tool_cold"),
    # Regressions on the families that already existed, so the new entries'
    # ORDERING cannot silently steal their evidence.
    ("Read", "File does not exist. Note: your current working directory is /tmp",
     "cwd_confusion"),
    ("Edit", "File has not been read yet. Read it first before writing to it.",
     "edit_before_read"),
    ("Edit", "String to replace not found in file.", "edit_string_not_found"),
    ("Edit", "File has been modified since read, either by the user or by a linter.",
     "stale_read_race"),
]


@pytest.mark.parametrize("tool,text,expected", _REAL_CORPUS_SAMPLES)
def test_real_corpus_wording_lands_in_the_intended_family(tool, text, expected):
    assert classify_known_family(tool, text, "") == expected


def test_every_family_ships_an_actionable_fix_not_a_placeholder():
    """A family exists to convert an observation into a claim. One whose fix is
    a placeholder claims nothing, so it would be a family in name only."""
    from tokenjam.core.optimize.write_budget import is_placeholder_fix

    for family in _KNOWN_FAMILIES:
        assert not is_placeholder_fix(family["fix"]), family["key"]
        assert family["rung"] in (1, 2, 3, 4, 5), family["key"]


def test_family_keys_are_unique():
    keys = [f["key"] for f in _KNOWN_FAMILIES]
    assert len(keys) == len(set(keys))


# --- The not-a-relearn filter -------------------------------------------------

def test_a_sibling_cancelled_by_another_calls_failure_is_not_a_relearn():
    """Claude Code marks the SIBLINGS of a parallel tool block as errored when
    one member fails. The sibling never ran, so it taught the agent nothing and
    forced no recovery turn of its own — the one recovery turn belongs to the
    member that actually failed and is already counted. On the local corpus
    this was the 4th-largest cluster (127 sessions) and pure double-count."""
    assert is_user_decline(
        "Cancelled: parallel tool call Bash(cd /Users/x/code && make test) errored"
    )


def test_a_bare_permission_prompt_is_not_a_relearn():
    """The user's own allowlist. The same command succeeds once approved, so no
    rule written into any agent-file surface changes the outcome."""
    assert is_user_decline("This command requires approval")


def test_the_chained_command_variant_IS_a_relearn():
    """The negative lookahead that separates the two: chaining is what forced
    the prompt here, and un-chaining removes it — so it must survive the filter
    and reach its own family."""
    text = ("This Bash command contains multiple operations. The following part "
            "requires approval: git push")
    assert not is_user_decline(text)
    assert classify_known_family("Bash", text, "") == "bash_chained_approval"


def test_a_real_failure_is_never_filtered_out():
    for _tool, text, _fam in _REAL_CORPUS_SAMPLES:
        assert not is_user_decline(text), text[:60]


# --- The relearn / resend boundary --------------------------------------------

def test_the_boundary_is_stated_once_and_quoted_by_both_analyzers():
    """Two analyzers price nearly the same physical tokens (a coding turn is
    ~99.8% re-read context), so an unstated boundary lets both claim them.
    Neither side may paraphrase it — both quote the one constant, so the two
    cards cannot drift into two different accounts of the same line."""
    from tokenjam.core.optimize.analyzers.context_resend import RESEND_ESTIMATE_BASIS
    from tokenjam.core.optimize.analyzers.relearn import ESTIMATE_BASIS

    assert RELEARN_RESEND_BOUNDARY in ESTIMATE_BASIS
    assert RELEARN_RESEND_BOUNDARY in RESEND_ESTIMATE_BASIS


def test_the_boundary_names_the_counterfactual_not_the_token_class():
    """The line is 'did this call have to happen', NOT 'is this token re-sent'.
    If it ever gets rewritten as a token-class split, relearn's claim collapses
    to the couple of fresh tokens a retry turn introduces, which would value
    eliminating a 96k-token rejected call at a fraction of a cent."""
    lowered = RELEARN_RESEND_BOUNDARY.lower()
    assert "should not have happened" in lowered
    assert "size of calls that had to" in lowered
