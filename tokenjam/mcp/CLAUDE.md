# `tokenjam/mcp/`

**The MCP is an SDK / API surface, not a Claude Code / Codex one.** It puts tj *in the request path* — the right place for SDK / API integrations doing real-time enforcement/policy/budgets. It is deliberately **not** wired for Claude Code / Codex subscription users: an in-loop MCP is a per-turn token burden on them (an A/B against a no-tj control measured a **materially higher** model-weighted token count; the figure itself is deliberately not restated here — it lives in the shipped product string that quotes it, and in the test pinning that string). Those users get tj **out-of-band**: the zero-token statusline (`tj statusline`, wired by `tj onboard --claude-code`) plus OTel telemetry ingest. `tj mcp` still works for anyone who invokes it; onboarding just no longer defaults CC/Codex users into it.

`server.py` is a FastMCP stdio server exposing observability data (plus the summarize tools —
`list_summarize_candidates`, `summarize_prep`, `summarize_check`, `summarize_apply`,
`summarize_undo`; see `core/summarize/`). It uses either a read-only DuckDB connection or an HTTP
proxy to `tj serve`, and is initialized via `init()` from `cli/cmd_mcp.py`.

`tj mcp` starts the server. The connection mode is chosen at startup by `cmd_mcp.py`:
1. If `tj serve` is reachable on `config.api.{host,port}`, MCP proxies to it via HTTP (live ingest visible).
2. Otherwise it tries to spawn `tj serve` in the background and waits for the port up to `_start_and_wait`'s `timeout` default (`cmd_mcp.py`).
3. If neither works, it falls back to a **read-only DuckDB connection** — read tools still work, but newly ingested spans won't appear until restart.
4. If no config file is found, `init()` is skipped and tools return a no-config sentinel.

SDK / API users who want the in-loop tools can wire it manually: `claude mcp add tj --scope user -- tj mcp`. The `--claude-code` and `--codex` onboard flows **no longer** register the MCP (they wire the out-of-band statusline / OTel instead), and a re-onboard retires any tj-managed `[mcp_servers.tj]` block a previous version wrote to `~/.codex/config.toml`.
