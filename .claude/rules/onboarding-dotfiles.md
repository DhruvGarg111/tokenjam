---
description: Managed-block discipline for dotfiles onboard/uninstall write (zshrc, Codex config, Claude settings).
paths:
  - "tokenjam/cli/cmd_onboard.py"
  - "tokenjam/cli/cmd_uninstall.py"
  - "tokenjam/cli/cmd_stop.py"
  - "tokenjam/cli/cmd_statusline.py"
  - "tokenjam/cli/ledger_hooks.py"
  - "tokenjam/core/commit_hooks.py"
  - "tokenjam/hooks/**"
---

# Onboarding / managed dotfile rules

### Critical Rule 21 — Dotfile-managed blocks (onboard/uninstall) must never key off the current marker string

Onboard writes managed blocks into `~/.zshrc` (OTEL exports) and the `claude()` wrapper; match them by
a STABLE sentinel pair and strip **every** legacy marker before writing exactly one fresh block, never
"append if current-marker absent." Precedent: the zshrc OTEL marker drifted once already
(`# ocw harness observability` → `# tj harness observability` at the openclawwatch→Token Juice rename,
commit `281275f`, shipped with no migration), which orphaned already-onboarded users' `.zshrc` —
re-onboard appended a stale-secret duplicate instead of replacing it, and `tj uninstall` left the old
block behind (fixed via a shared `_strip_zshrc_otel_blocks()` in `cmd_onboard.py`). Codex's
`[otel]` config had the analogous issue, handled by `_codex_purge_legacy_ocw`. Any new
managed-dotfile-block feature needs an onboard→uninstall round-trip test asserting zero residue,
including from seeded legacy markers.

The git hook blocks `tj init --hooks` / `--notes` write (`core/commit_hooks.py`, templates in
`tokenjam/hooks/`) follow the same discipline: one sentinel pair per hook, install strips every
existing block (a damaged one included, cut through its own `unset -f` line) and writes exactly one
fresh block after the shebang, a hook the user owns is preserved around it, a non-shell hook is never
edited, and `tj uninstall` strips the blocks from every hooks directory recorded in
`~/.tj/commit_hooks.json` (hooks are per repo and uninstall runs from anywhere). Linked worktrees
share the canonical clone's hooks directory (`git rev-parse --git-path hooks` resolves to the common
dir), so an install from a worktree serves every checkout of that repo.
