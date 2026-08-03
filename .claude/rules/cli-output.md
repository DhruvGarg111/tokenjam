---
description: Terminal output discipline — the shared Rich console, the named theme roles, no raw colours.
paths:
  - "tokenjam/cli/**"
  - "tokenjam/utils/formatting.py"
  - "tokenjam/utils/theme.py"
  - "tokenjam/demo/**"
---

# CLI output rules

### Critical Rule 35 — All terminal output goes through the shared `console` in `tokenjam/utils/formatting.py`

It sets `highlight=False` — never construct a bare `rich.Console()`, and never write a raw colour name
in markup. Rich's automatic highlighter is **on by default** and repaints *prose* by pattern
regardless of the markup you wrote: numbers, paths, a bare `/` in a sentence, brackets, quoted
strings, ellipses. It does not merely add noise, it corrupts copy — `Max 20x plan` renders as `Max 2`
plus a cyan `0x`, because the highlighter matches `0x` as a hex literal. Stacked on the deliberate
palette it puts colours on screen that were chosen by nobody, and a fresh `Console()` anywhere
re-imports the whole problem for that surface. **Deliberate colour goes through the named roles in
`tokenjam/utils/theme.py`** (`accent`, `brand`, `url`, `label`, `heading`, `muted`, `ok`, `warn`,
`error`), so the palette is auditable and changeable in one file. The discipline those roles encode,
modelled on Claude Code's own CLI: prose is plain; **`accent` means exactly one thing — a string the
user can type or click** (command, path, config key, URL), never decoration or emphasis; structure is
weight (`label`/`heading`), not colour; **success is a `✓` plus bold, never green**, because a colour
spent on the least surprising outcome in the flow stops meaning anything; and past the accent, colour
is reserved for genuine state — `warn` for a blocker the user must act on *now*, `error` for a real
failure. Two traps worth naming: **`[bold]` nested inside `[dim]` renders bold-dim**, which is neither
emphasis nor recession, just a smudge; and a **single coloured row inside an otherwise plain field
list reads as a failure**, so an informational row in a summary block must stay plain even when it
reports a degraded outcome. `tests/unit/test_cli_palette.py` pins all of this — including that
`cmd_onboard.py` contains no raw `[green]`/`[yellow]`/`[red]` — so a regression fails rather than
merely looking off. When judging a change here, **render it and inventory the SGR codes**
(`FORCE_COLOR=1 … | cat -v`, then group the distinct `\e[...m` sequences); each emitted SGR code
should have a stateable job.
