"""`tj init --hooks / --notes / --enforce`: the ledger steps that run after
(or instead of) the onboarding wizard, and the hook-state line every
`tj init` run prints (ledger W3; contracts §3, §5, §9).

Why they hang off `tj init` rather than a command of their own: the brief's
goal is that enforcement onboarding is one flag on the same command as
observability onboarding, and the commit hook is the per-repo half of the
stamping `tj init` already does (contracts §3). `tj init --hooks --enforce`
composes; each step is idempotent, so a re-run in the same repo is a no-op
that says so.

The hook install writes to the hooks directory git will actually consult
(`git rev-parse --git-path hooks`, which honours `core.hooksPath`), with one
refusal: a `core.hooksPath` inside the worktree is a tracked directory
(husky and friends) and tj never edits tracked files. A user who keeps a
global hooks directory outside any worktree (`git config --global
core.hooksPath ~/.git-hooks`) has opted in to a global install, and the
block lands there, once, for every repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import click

from tokenjam.core.commit_hooks import (
    HOOK_NAMES,
    STATE_CURRENT,
    STATE_DAMAGED,
    STATE_FOREIGN,
    STATE_STALE,
    HooksLocation,
    active_session_for,
    hook_states,
    install_hook_block,
    register_hooks_dir,
    registered_hooks_dirs,
    remove_hook_block,
    resolve_hooks_location,
)
from tokenjam.utils.formatting import console

#: Contracts §9, printed verbatim on every `--enforce` run.
SUBSCRIPTION_TRAFFIC_SENTENCE = (
    "Subscription-plan traffic is never intercepted and subscription OAuth "
    "credentials are never proxied or forwarded: the proxy classifies "
    "API-key (usage-billed) traffic only, and in suggest mode forwards "
    "everything unmodified."
)


@dataclass
class HookInstallReport:
    location: HooksLocation | None
    #: hook name -> install outcome (`written` / `updated` / `kept` / `skipped`)
    outcomes: dict[str, str] = field(default_factory=dict)
    refusal: str | None = None

    @property
    def ok(self) -> bool:
        return self.refusal is None and bool(self.outcomes)


def install_repo_hooks(cwd: str, *, notes: bool) -> HookInstallReport:
    """Install the trailer hook (and the notes hook when `notes`) for the
    repo at `cwd`. Pure outcome; the caller renders."""
    loc = resolve_hooks_location(cwd)
    if loc is None:
        return HookInstallReport(location=None, refusal="not a git repository")
    if loc.tracked:
        return HookInstallReport(
            location=loc,
            refusal=(
                f"core.hooksPath points at {loc.hooks_dir}, a directory inside the "
                "worktree; tj does not edit tracked files. Add the block from "
                "tokenjam/hooks/prepare-commit-msg to your own hook instead."
            ),
        )
    report = HookInstallReport(location=loc)
    names = ["prepare-commit-msg", "post-commit"] if notes else ["prepare-commit-msg"]
    for name in names:
        report.outcomes[name] = install_hook_block(name, loc.hooks_dir / name)
    if any(o in ("written", "updated", "kept") for o in report.outcomes.values()):
        register_hooks_dir(loc.hooks_dir)
    return report


def remove_all_repo_hooks(cwd: str) -> dict[str, dict[str, str]]:
    """Strip every tj-managed block from every hooks directory tj recorded
    writing to, plus the repo `cwd` is in. `{hooks_dir: {hook: outcome}}`,
    only for directories where something was actually removed. The notes
    ref itself is user data and stays."""
    dirs = list(registered_hooks_dirs())
    loc = resolve_hooks_location(cwd)
    if loc is not None and str(loc.hooks_dir) not in dirs:
        dirs.append(str(loc.hooks_dir))
    out: dict[str, dict[str, str]] = {}
    for d in dirs:
        hooks_dir = Path(d)
        outcomes = {name: remove_hook_block(name, hooks_dir / name) for name in HOOK_NAMES}
        if any(o != "absent" for o in outcomes.values()):
            out[d] = outcomes
    return out


def print_hook_install(report: HookInstallReport, *, notes: bool) -> None:
    if report.refusal is not None:
        console.print(f"[warn]Commit hook not installed:[/warn] {report.refusal}")
        if report.location is None:
            console.print("  Run [accent]tj init --hooks[/accent] from inside the repository.")
        return
    assert report.location is not None
    words = {"written": "installed", "updated": "updated", "kept": "already current",
             "skipped": "skipped (hook is not a shell script; left untouched)",
             "damaged": "skipped (a damaged tj block is in the way; remove it by hand)"}
    for name, outcome in report.outcomes.items():
        target = report.location.hooks_dir / name
        console.print(f"[ok]✓[/ok] {name} hook {words[outcome]}: [accent]{target}[/accent]",
                      soft_wrap=True)
    if report.location.hooks_path_setting:
        console.print(
            f"  [muted]core.hooksPath is set, so this block serves every repo "
            f"that resolves hooks to {report.location.hooks_dir}.[/muted]", soft_wrap=True,
        )
    console.print(
        "  Commits made from a plain shell while a Claude Code session is open "
        "in this repo now carry a [bold]TokenJam-Session:[/bold] trailer, so "
        "[accent]tj optimize shipped[/accent] joins them at deterministic confidence."
    )
    if notes:
        console.print(
            "  Each such commit also gets its session's measured cost written to "
            "[bold]refs/notes/tokenjam[/bold] when [accent]tj serve[/accent] is running."
        )


def hook_summary_line(cwd: str) -> str:
    """One line for the `tj init` end-of-run summary: the state of the
    trailer hook in the repo the command ran from."""
    loc = resolve_hooks_location(cwd)
    if loc is None:
        return "Commit hook: not a git repository here (run tj init --hooks inside a repo)"
    states = hook_states(loc.hooks_dir)
    trailer = states["prepare-commit-msg"]
    note = states["post-commit"]
    if trailer == STATE_CURRENT:
        line = "Commit hook: installed"
        if note == STATE_CURRENT:
            line += ", git notes on"
        elif note == STATE_STALE:
            line += ", git notes hook stale (re-run tj init --notes)"
        return line
    if trailer == STATE_STALE:
        return "Commit hook: installed by another build (re-run tj init --hooks to refresh)"
    if trailer == STATE_DAMAGED:
        return "Commit hook: block damaged (re-run tj init --hooks to rewrite it)"
    if trailer == STATE_FOREIGN:
        return "Commit hook: this repo's prepare-commit-msg is not a shell script; not installed"
    return "Commit hook: not installed (tj init --hooks adds the session trailer)"


def run_enforce(ctx: click.Context) -> None:
    """`--enforce`: turn the proxy on through the existing `tj proxy enable`
    wiring, against the config as it stands AFTER onboarding wrote it, then
    print the enforcement summary and the contracts §9 sentence."""
    from tokenjam.cli.cmd_proxy import proxy_enable
    from tokenjam.core.config import load_config, resolve_config_path
    from tokenjam.proxy.wiring import BASE_URL_ENV_VARS, proxy_base_url

    ctx.ensure_object(dict)
    path = resolve_config_path(ctx.obj.get("config_path_override"))
    if path:
        ctx.obj["config"] = load_config(str(path))
    if ctx.obj.get("config") is None:
        console.print(
            "[warn]Enforcement not enabled:[/warn] no tj config found. Run "
            "[accent]tj init[/accent] first, then [accent]tj init --enforce[/accent]."
        )
        return
    console.print()
    ctx.invoke(proxy_enable)
    config = ctx.obj["config"]
    console.print()
    console.print("[bold]Enforcement[/bold]")
    console.print(
        f"  Proxy: suggest mode on [accent]{proxy_base_url(config)}[/accent]; "
        f"{', '.join(BASE_URL_ENV_VARS)} in ~/.claude/settings.json point at it."
    )
    console.print("  Suggest mode changes no request; it records what a policy would have done.")
    console.print(f"  {SUBSCRIPTION_TRAFFIC_SENTENCE}", soft_wrap=True)
    console.print("  Status any time: [accent]tj proxy status[/accent]")


def print_active_session_state(cwd: str) -> None:
    """Whether a hand commit right now would join a session (the doctor
    check says the same thing; this is the init-time echo)."""
    entry = active_session_for(cwd)
    if entry:
        console.print(f"  Active session for this repo: [bold]{entry['session_id']}[/bold]")
    elif os.environ.get("CLAUDE_CODE_SESSION_ID"):
        console.print("  Running inside a Claude Code session; its id is in the environment.")
