# `tokenjam/core/summarize/`

Structure-aware prompt summarization (advisory). Pure domain logic — no `tokenjam.cli` /
`tokenjam.api` imports (delivery's API path lazily imports `httpx`, its lone outbound dependency).

- `detect.py` classifies prose vs. structure (fenced/inline code, tags, templates, tables).
- `candidates.py` (+ `catalog.py` / `estimate.py`) powers the `tj summarize list` scan.
- `wrap.py` is the pure protect→restore algorithm: wrap each structured span behind an id'd
  `<tj-keep>` marker, restore verbatim by id — structure is a hard guarantee.
- `session.py` is the no-scratch `prepare`/`check` lifecycle + staging (re-derives the wrap from the
  live file + a content hash; persists nothing but the staged result).
- `apply.py` writes a staged rewrite back to the file (default dry-run; `--go` writes) behind an
  owner + content-hash + symlink guard, with a gzip backup and `undo`.
- `backup.py` stores the gzipped original + metadata under `~/.tj/summary/backups/`.
- `delivery.py` is the CLI's automated rewrite step — `claude -p` (subprocess, timeout-guarded) or
  the Anthropic API (lazy `httpx` + the user's own `TJ_ANTHROPIC_API_KEY`) — plus the "pays for
  itself" amortization.
- `load_semantics.py` is the single source of truth for how an agent-config file loads (always
  resident vs. frontmatter-now / body-on-invocation); `invocations.py` observes the invocation
  multiplier from Claude Code transcripts. Both are consumed by `core/optimize/write_budget` — see
  `core/optimize/CLAUDE.md`.
- `repo_roots.py` resolves recorded session cwds to repo roots for `core/optimize/rule_placement`.

**Environment sensitivity (Critical Rule 31):** `candidates.list_candidates` takes no session CWD.
With no explicit `path` it scans the catalog **globals** plus `Path.cwd()` — the tj PROCESS's own
working directory, never the corpus's recorded session paths. Its floor is therefore the globals, so
moving the CWD barely moves the figure; it collapses only when the GLOBALS vanish too, which is what
repointing `HOME` does. Any measurement of its dollar figure must run against the real `HOME`.
