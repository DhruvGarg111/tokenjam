"""The catalogued fixes themselves.

Split from ``catalog.py`` so the data sits apart from the machinery, and
imported for its side effect the way ``core/optimize/analyzers`` is. Every
entry names the observation it answers and the shapes it may not excuse, which
is what lets ``lint.py`` check the property that matters: applying the fix must
be able to ERASE the number its analyzer found.
"""
from __future__ import annotations

from tokenjam.core.fixes.catalog import (
    Substitution,
    LEVER_AWARENESS,
    LEVER_EFFORT,
    LEVER_MODEL,
    LEVER_OFFLOAD,
    LEVER_ROUTING,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    register,
)

#: The shapes the subagent/resend family must never give a pass to. These are
#: exactly what the analyzer bills for: the `output_tokens < 2000` /
#: `tool_calls <= 5` gate was DELETED from `subagent_rightsizing` because on a
#: real Claude Code corpus it excluded the expensive dispatches, so a fix that
#: re-imposes it in prose undoes the gate fix in the one place the user reads.
_SIZING_RELICENSE = frozenset({
    "little tool work",
    "few tool calls",
    "short result rarely needs",
    "short conclusion rarely needs",
    "rarely needs the premium tier",
})

SUBAGENT_RUBRIC = register(FixRecord(
    key="subagent.sizing_rubric",
    text=(
        "Right-size Task-dispatched subagents: default every subagent to the "
        "cheapest same-family model that fits its shape, and treat that "
        "default as the answer unless the dispatch itself states one of these "
        "conditions: the subtask IS the architecture or design decision this "
        "session exists to make; it must reconcile sources that disagree into "
        "one judgement a later step cannot re-derive; or a cheaper model "
        "already attempted it in this session and its output was rejected. "
        "How much tool work a subagent does and how long its result runs are "
        "not on that list: a broad, tool-heavy, long-output dispatch is the "
        "expensive one, not the hard one, and it is the one this rule exists "
        "to route down."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="premium-tier models running Task dispatches that a cheaper same-family model fits",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
    grounding=(
        Substitution(
            find="Right-size Task-dispatched subagents",
            template="Right-size the {agents} dispatches",
        ),
    ),
))

RIGHTSIZE_TEMPLATE = register(FixRecord(
    key="resend.rightsize_worker",
    text=(
        "Then right-size what you offload to. Default the worker to the "
        "cheapest same-family model that fits its shape, and pin both that "
        "model and its reasoning effort in its own definition file so every "
        "future dispatch inherits them instead of defaulting to whatever the "
        "parent runs on. How much it reads and how long its conclusion runs "
        "are not reasons to keep the premium tier; they are what makes the "
        "dispatch expensive in the first place."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"resend"}),
    answers="context-heavy work run inline on a premium model instead of in a right-sized worker",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
    grounding=(
        Substitution(find="the worker", template="the {agents} workers"),
    ),
))

#: The CREATE fix. A Claude Code dispatch is almost always a BUILT-IN
#: (`general-purpose`, `Explore`, `Plan`, `fork`), which has no definition file
#: — and the product used to read "no file" as "no fix", falling through to
#: prose. The docs are explicit that a user or project subagent of the same
#: name OVERRIDES the built-in and keeps its own `model` field, so the file can
#: be created. The waste's origin belongs in the same breath: `model` defaults
#: to `inherit`, so an opus-driven session dispatches opus workers unless
#: pinned. Nobody decided that; it is an unset default.
SUBAGENT_DEFINE_BUILTIN = register(FixRecord(
    key="subagent.define_builtin_override",
    text=(
        "Pin the model for the built-in subagents this session dispatches. A "
        "subagent's `model` defaults to `inherit`, so an Opus-driven session "
        "hands every Task dispatch an Opus worker unless something says "
        "otherwise — that is an unset default, not a decision. A user or "
        "project subagent defined with the same name as a built-in overrides "
        "it and keeps its own `model`, so creating the definition file is the "
        "fix: give it the cheapest same-family model that fits the work it "
        "actually does."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="built-in subagents inheriting the driver's premium model because no definition file pins one",
    lever=LEVER_MODEL,
    must_not_relicense=_SIZING_RELICENSE,
))

SUBAGENT_EFFORT = register(FixRecord(
    key="subagent.pin_effort",
    text=(
        "Pin a subagent's effort alongside its model. They answer different "
        "diagnoses and neither substitutes for the other: a worker that did "
        "not know enough needs a larger model, a worker that did not try hard "
        "enough needs more effort. Both are frontmatter fields on the "
        "subagent's own definition file, so a dispatch inherits them instead "
        "of taking the parent's."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"subagent"}),
    answers="subagents inheriting the parent's effort setting rather than one sized to their task",
    lever=LEVER_EFFORT,
))

#: THE offload instruction. Three analyzers used to write this into a
#: CLAUDE.md in three separately-authored wordings — `resend`'s
#: `SUBAGENT_OFFLOAD_FIX`, `downsize`'s driver-role advice, and `relearn`'s
#: `context_overflow` family. They price genuinely different span populations
#: (Critical Rule 27 is untouched), but they were all telling the agent the
#: same thing, so three near-identical blocks could land in one file.
#:
#: That is not merely untidy: length and redundancy REDUCE adherence, so
#: writing the rule three times makes it less likely to be followed than
#: writing it once — the opposite of what each analyzer intended. It also
#: reads as broken to anyone who opens the file.
#:
#: The text below is what the three wordings shared, stated once. Per-analyzer
#: framing (why THIS card is showing it) belongs in the card's advise text,
#: never in a second copy of the rule.
#:
#: **This record is about WHERE the work runs and nothing else.** Model pinning
#: belongs to ``resend.rightsize_worker``, which owns "what it runs on", and
#: keeping the two separate is not pedantry: the compound artifact renders both
#: records back to back, so a pinning sentence here lands immediately before
#: the one that owns it and the user reads the same instruction twice. Caught
#: by rendering the composed artifact and reading it — the pairwise catalog
#: lint scored the pair at 42%, under threshold, because each record is only
#: partly redundant. One record, one instruction, is what makes composition
#: safe.
OFFLOAD_RULE = register(FixRecord(
    key="resend.offload_to_subagent",
    text=(
        "Offload context-heavy sub-tasks (broad file reads, log sweeps, "
        "multi-file search, long tool-output loops, exploratory "
        "investigation) to a subagent instead of running them inline in the "
        "main thread, and prefer a targeted search plus a bounded read over "
        "reading a large file end to end. A subagent's own tool logs and "
        "intermediate output stay in its own context; only its short "
        "conclusion returns to the caller, so the material that would "
        "otherwise be re-sent on every later turn never accumulates on the "
        "main thread to begin with. This is not a request to downgrade the "
        "session you are driving: the driver keeps the premium model and "
        "keeps making the decisions. What changes is where the bulk context "
        "lives."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    # One record, three analyzers. Each prices a different population and each
    # references THIS text rather than authoring its own.
    analyzers=frozenset({"resend", "downsize", "relearn"}),
    answers="context-heavy work run inline on the main thread, re-sent on every later turn",
    lever=LEVER_OFFLOAD,
    # `relearn`'s `context_overflow` family shows this record for a reason the
    # other two do not have — the request was REJECTED, not merely expensive —
    # and that sentence used to be written beside the family table. It names
    # what was observed and stops; restating any part of the rule below is the
    # defect this record exists to have already fixed.
    lead_ins=((
        "relearn",
        "This session hit the model's context ceiling and the request was "
        "rejected outright: the tokens were spent and no completion came back.",
    ),),
    # REPLACES the vague enumeration with the observed one; it does not append
    # to it. "the `Read` and `Grep` sweeps you run in `optimize/`" is shorter
    # than the parenthesised list it stands in for, which is the point — the
    # guidance is specific AND concise, and a longer rule is a less-followed
    # rule.
    grounding=(
        Substitution(
            find=(
                "context-heavy sub-tasks (broad file reads, log sweeps, "
                "multi-file search, long tool-output loops, exploratory "
                "investigation)"
            ),
            template="the {tools} sweeps you run in {repos}",
        ),
        # Falls back to naming just the directories when the tool mix was not
        # observed — still concrete, still shorter than the generic span.
        Substitution(
            find=(
                "context-heavy sub-tasks (broad file reads, log sweeps, "
                "multi-file search, long tool-output loops, exploratory "
                "investigation)"
            ),
            template="the context-heavy work you run in {repos}",
        ),
    ),
))

#: The SDK counterpart of the same observation. A Claude Code user cannot
#: paste this and an SDK caller cannot use the offload rule above; handing
#: either the other's fix names an action they cannot take.
SDK_CACHE_CONTROL = register(FixRecord(
    key="resend.sdk_cache_breakpoint",
    text=(
        "Adopt a cache_control breakpoint at the call site so the repeated "
        "prefix bills at the cache rate instead of full price on every turn, "
        "and trim the request you build rather than relying on the harness to "
        "do it — an SDK caller constructs the whole prompt, so the repeated "
        "context is yours to bound."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_SDK}),
    analyzers=frozenset({"resend"}),
    answers="repeated prompt prefix billed at full input rate on every turn",
    lever=LEVER_ROUTING,
))

#: An OBSERVATION, not an offer. The harness already errors clearly and agents
#: self-correct next turn, so there is no action to sell — but the recurrence
#: genuinely cost something, and that figure stands (Critical Rule 32).
EDIT_BEFORE_READ = register(FixRecord(
    key="relearn.edit_before_read",
    text=(
        "The harness already blocks an Edit/Write before a Read with a clear "
        "error and agents reliably self-correct by reading on the next turn, "
        "so no rule or hook is needed here. This is reported as awareness "
        "only: what the retries already cost is real and is shown, but there "
        "is no change to make."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Edit/Write attempted before Read, retried after the harness error",
    lever=LEVER_AWARENESS,
    advisory_only=True,
))

#: An awareness fix that IS confined to identifiable files, so it can carry a
#: glob and cost almost nothing. The observation names the file class directly:
#: the mistake happens when editing a migration without reading it, which is a
#: statement about a kind of file rather than about the shape of the next
#: action. Contrast every model/effort/offload record above, which decide
#: something BEFORE any file is read and so must stay always-resident.
MIGRATION_READ_FIRST = register(FixRecord(
    key="relearn.migration_read_before_edit",
    text=(
        "Read a migration file in full before editing it. Migrations are "
        "append-only and ordered, so an edit written from a remembered shape "
        "rather than the current contents lands in the wrong place or "
        "duplicates an existing step, and the failure surfaces later as a "
        "schema that does not match its own history."
    ),
    delivery="path_scoped_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="edits to migration files written without reading the file first",
    lever=LEVER_AWARENESS,
    path_globs=("**/migrations/**", "**/migrate/**"),
))


# --- relearn: the known failure families -------------------------------------#
#
# One record per family, keyed ``relearn.<family_key>`` so the family table and
# the catalog cannot drift apart on which text belongs to which matcher. These
# were inline strings in ``analyzers/relearn.py``; nothing checked them, so the
# properties the lint enforces held only by whoever last read the file.
#
# Their delivery is declared per family because the FAMILY is what knows:
# `sleep_chain` blocks a command and injects nothing, while the PostToolUseFailure
# families exist precisely to inject text. Those cost opposite amounts and no
# property of the artifact tells them apart.

RELEARN_CWD_CONFUSION = register(FixRecord(
    key="relearn.cwd_confusion",
    text=(
        "PostToolUseFailure hook (Bash/Read): react only after a "
        "'no such file or directory' failure by injecting the real cwd + "
        "a short directory listing as additionalContext, so the agent "
        "recovers in one shot instead of a PreToolUse guess-and-block on "
        "every relative path (which would misfire on normal usage)."
    ),
    delivery="injecting_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="tool calls failing on a relative path resolved against the wrong working directory",
    lever=LEVER_AWARENESS,
))

RELEARN_SLEEP_CHAIN = register(FixRecord(
    key="relearn.sleep_chain",
    text=(
        "PreToolUse hook: block a `sleep N && <check>` Bash chain and point the "
        "agent at the Monitor tool instead of a busy-wait."
    ),
    delivery="executing_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="blocked sleep-chain Bash commands retried after the harness refused them",
    lever=LEVER_AWARENESS,
))

#: ONE record for TWO families. `stale_read_race` ("modified since read") and
#: `edit_string_not_found` ("string to replace not found") are different
#: matchers over different failures, and they stay different families — the
#: hook spec, the delivery and the evidence are all per-family. But the
#: INSTRUCTION is identical in both: re-read the file before retrying the edit.
#: They shipped as two separately-authored wordings and scored 94% against each
#: other, which is the three-analyzers-one-rule case in miniature: both hooks
#: can be applied by the same user, so the same sentence lands twice.
#:
#: Which failure prompted it is the FAMILY's job to say (each has its own title
#: and evidence), not this text's. Naming the trigger here is what forked one
#: instruction into two in the first place.
RELEARN_REREAD_BEFORE_RETRYING_EDIT = register(FixRecord(
    key="relearn.reread_before_retrying_edit",
    text=(
        "PostToolUseFailure hook (Edit/Write/MultiEdit): react only after the "
        "edit has already failed, by injecting a re-Read reminder as "
        "additionalContext so the retry is written against the file's current "
        "contents — never touches a successful edit."
    ),
    delivery="injecting_hook",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="edits retried against remembered file contents after the file moved underneath them",
    lever=LEVER_AWARENESS,
))

#: The OPPOSITE failure to ``edit_string_not_found`` — too many matches, not
#: zero — and it takes the opposite fix, which is why the two families must
#: never share a bucket and why their records are separate here.
RELEARN_EDIT_AMBIGUOUS_MATCH = register(FixRecord(
    key="relearn.edit_ambiguous_match",
    text=(
        "CLAUDE.md/skill note: when an Edit's `old_string` appears more "
        "than once, include enough surrounding lines to make it unique "
        "rather than retrying the same short string — or pass "
        "`replace_all: true` when every occurrence really should change."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Edit calls rejected because old_string matched more than once",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_TOO_LARGE = register(FixRecord(
    key="relearn.read_too_large",
    text=(
        "CLAUDE.md/skill note: this file is too large to read whole. Grep "
        "for the symbol first and Read only the region around the hit "
        "(`offset`/`limit`), or delegate the sweep to a subagent so the "
        "bulk never lands in this thread's context."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls rejected for exceeding the tool's max-tokens ceiling",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_DIRECTORY = register(FixRecord(
    key="relearn.read_directory",
    text=(
        "CLAUDE.md/skill note: Read takes a file path. To see what is in a "
        "directory use Glob (or `ls` via Bash), then Read the file you "
        "actually want."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls pointed at a directory rather than a file",
    lever=LEVER_AWARENESS,
))

RELEARN_READ_OFFSET_MALFORMED = register(FixRecord(
    key="relearn.read_offset_malformed",
    text=(
        "CLAUDE.md/skill note: Read's `offset`/`limit` are scalars, not "
        "arrays — pass a single number for each."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Read calls rejected for passing offset or limit as an array",
    lever=LEVER_AWARENESS,
))

RELEARN_DEFERRED_TOOL_COLD = register(FixRecord(
    key="relearn.deferred_tool_cold",
    text=(
        "Skill/scoped note: deferred tools need a ToolSearch lookup for their "
        "schema before the first call; optionally a PreToolUse intercept hook."
    ),
    delivery="skill",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="deferred tools called before their schema was fetched, so the call could not resolve",
    lever=LEVER_AWARENESS,
))

RELEARN_COMMAND_NOT_FOUND = register(FixRecord(
    key="relearn.command_not_found",
    text=(
        "CLAUDE.md/skill note: this shell doesn't have that binary/builtin on "
        "PATH. Common causes here: using bare `python` instead of `python3`, "
        "or a bash-only builtin (`mapfile`, `shopt`, `[[ ... ]]` extensions) "
        "that doesn't exist under this shell (e.g. zsh, sh) or POSIX mode. "
        "Prefer the portable/explicit form."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Bash calls failing on a binary or builtin this shell does not have",
    lever=LEVER_AWARENESS,
))

RELEARN_BASH_TIMEOUT = register(FixRecord(
    key="relearn.bash_timeout",
    text=(
        "CLAUDE.md/skill note: this command outlived the tool's timeout "
        "and was killed, so its work was lost and the tokens spent "
        "waiting bought nothing. Run long jobs in the background "
        "(`run_in_background`) and poll for completion, or raise the "
        "call's own timeout when the wait is genuinely expected."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="Bash commands held in the foreground until the harness killed them",
    lever=LEVER_AWARENESS,
))

RELEARN_BASH_CHAINED_APPROVAL = register(FixRecord(
    key="relearn.bash_chained_approval",
    text=(
        "CLAUDE.md/skill note: a chained Bash command (`cd X && cmd`, "
        "`a; b`) is approved as a whole, so one un-allowlisted part blocks "
        "the entire chain. Issue the parts as separate Bash calls, and "
        "prefer an absolute path over a leading `cd`."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="chained Bash commands tripping the approval prompt the parts alone would not",
    lever=LEVER_AWARENESS,
))

RELEARN_GIT_BRANCH_EXISTS = register(FixRecord(
    key="relearn.git_branch_exists",
    text=(
        "CLAUDE.md/skill note: check out the existing branch "
        "(`git checkout <name>`) instead of re-creating it, or pick a "
        "fresh name — `git checkout -b` on an existing branch always "
        "fails."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="`git checkout -b` retried against a branch name that already exists",
    lever=LEVER_AWARENESS,
))

RELEARN_WEBFETCH_DOMAIN_BLOCKED = register(FixRecord(
    key="relearn.webfetch_domain_blocked",
    text=(
        "CLAUDE.md/skill note: this domain is blocked — use a search tool "
        "or a different source instead of retrying the fetch."
    ),
    delivery="claude_md_rule",
    personas=frozenset({PERSONA_CLAUDE_CODE}),
    analyzers=frozenset({"relearn"}),
    answers="fetches retried against a domain the harness will not reach",
    lever=LEVER_AWARENESS,
))


__all__ = [
    "EDIT_BEFORE_READ",
    "OFFLOAD_RULE",
    "RIGHTSIZE_TEMPLATE",
    "SDK_CACHE_CONTROL",
    "SUBAGENT_DEFINE_BUILTIN",
    "SUBAGENT_EFFORT",
    "SUBAGENT_RUBRIC",
]
