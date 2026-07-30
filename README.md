<div align="center">

<img src="docs/brand/tokenjam-repo-header.png" alt="TokenJam: token efficiency for AI agents. Reads your agent's telemetry, finds the waste, runs 100% local." width="830">

TokenJam reads your agent's telemetry, finds where the tokens actually go, and hands you the fix, not just the finding. Works with Claude Code, Codex, and your own SDK or API agents. Shows it all in a local browser dashboard. Runs entirely on your machine.

[![CI](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml/badge.svg)](https://github.com/Metabuilder-Labs/tokenjam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![Downloads](https://img.shields.io/pypi/dm/tokenjam?color=3d8eff&labelColor=0d1117&label=downloads)](https://pypi.org/project/tokenjam/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3d8eff?labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![npm](https://img.shields.io/npm/v/@tokenjam/sdk?color=3d8eff&labelColor=0d1117)](https://www.npmjs.com/package/@tokenjam/sdk)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d8eff?labelColor=0d1117)](LICENSE)
[![OTel](https://img.shields.io/badge/OTel-GenAI%20SemConv-3d8eff?labelColor=0d1117)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

**No cloud · No signup · No vendor lock-in**

</div>

---

## Get started

TokenJam ingests telemetry data about your agents from a multitude of sources and provides you a quick and easy way to visualize and optimize cost so that you get the most out of the tokens you pay for.

It no longer stops at telling you where the money went. Every analyzer ends in a **fix you can apply**: a rule written into the right instruction file, an unused MCP server scoped down, a subagent pinned to a cheaper model, a cache breakpoint placed in the request your code builds. Each one is staged as a diff, applied on your say-so, and undoable.

One command sets up live capture, the analyzers that fit how you work, Lens (the local dashboard), and the zero-token statusline:

```bash
npx tokenjam onboard   # or: pipx install tokenjam && tj onboard
```

`tj onboard` asks how you use AI agents (Claude Code, Codex, or your own SDK/API agents) and wires the right path; under npx it first offers to make itself a permanent install. For Claude Code and Codex that means backfilling your recent history plus the statusline and hooks; restart and you're live. Then run:

```bash
tj optimize          # cost-saving candidates from your actual usage
tj rules list        # the fixes on offer, and the files they'd be written into
tj serve             # open the dashboard at http://127.0.0.1:7391/
```

The statusline is **zero-token**; `tj statusline` runs out-of-band each turn (no model quota) and shows this session's re-read share with a `/compact` nudge: `◆ Opus 4.8  2.4M tok  🕳️ re-read 95%  → /compact to reclaim quota`. It does **not** add an in-loop MCP server (that's an SDK / API surface; an MCP would tax every turn).

Run bare `tj` any time and it points you to the next best action (`tj status`, `tj tokenmaxx`, `tj optimize`, or `tj serve`).

**Just looking?** `npx tokenjam` prints a 15-second read-only report over the logs you already have: no install, nothing kept.

Building your own agent with the SDK: install *in your project* (`pip install tokenjam` + `tj onboard`); see the table below.

<sub>`npx tokenjam` and `uvx tokenjam` launch the Python CLI via `uvx`/`pipx` under the hood; see [docs/installation.md](docs/installation.md) for the runner requirements and the full install matrix.</sub>

<div align="center"><img src="docs/assets/tokenjam-token-flow.png" alt="Token flow: telemetry from Claude Code, Codex, Google, AWS, the Python and TypeScript SDKs, LangChain/CrewAI, and OTLP/Langfuse flows into tokenjam, which decomposes where every token goes: 94% re-reads of history and context, 5.1% tool output, 0.9% net-new work, measured over a 61-session history" width="830"></div>

---

## Which path are you?

`pipx install tokenjam && tj onboard` is the entry point for everyone: it's an interactive wizard
that asks how you use AI agents and wires the right path for you. `--claude-code` / `--codex` just
pre-answer the wizard's first question (skip it in scripts/CI); they're shortcuts, not separate setups.

Your answer also decides **which analyzers run**. TokenJam sorts users into two personas: you're
driving a **coding agent** (Claude Code, Codex), or you're driving your **own SDK/API code**. Each can
change a different set of things, so each gets a different set of analyzers. Most run for both.
[The analyzers](#the-analyzers) below has the full split and the reasoning.

| You are | Run this | What you get |
|---|---|---|
| **Claude Code user** | `tj onboard` (or `tj onboard --claude-code` to skip the first question) | Auto-backfills your last 30 days, wires a zero-token statusline, unlocks the coding-agent analyzers + Lens |
| **Codex CLI user** | `tj onboard` (or `tj onboard --codex`) | Same onboarding flow, wired for Codex's session logs |
| **Python SDK / API agent dev** | `tj onboard` + `@watch()` in your code ([Python SDK](docs/python-sdk.md)) | Live capture from your own agent process, no CLI-specific backfill |
| **Framework user** (LangChain / CrewAI / AutoGen) | `pip install tokenjam[langchain]` (or `[crewai]` / `[autogen]`) + one `patch_*()` call | Framework-level spans with no manual instrumentation |
| **Already on Langfuse / Helicone** | `tj backfill langfuse --source-url <url> --api-key <key>`<br>(swap `langfuse` → `helicone`, same flags) | One-time import of your existing traces into the local DB |
| **Any OTel-emitting agent** | Point your OTLP exporter at `tj serve` (`http://127.0.0.1:7391/v1/traces`) | Zero-code ingestion: no SDK, no patch |

**Working across multiple projects?** Run `tj onboard` once inside each one — sessions and cost
proposals group per project in Lens, keyed on the project name each onboard run captures. Already
onboarded elsewhere? `tj onboard --add-project` registers just the current repo's namespace against
your existing setup, skipping the plan/budget prompts and backfill.

LlamaIndex and the OpenAI Agents SDK ship their own native OTel support; point their exporter at `tj serve` rather than installing an extra. Full matrix: [docs/framework-support.md](docs/framework-support.md).

A single page walks every path, each ending with a verify step: see
[docs/getting-started.md](docs/getting-started.md).

---

## The analyzers

<div align="center"><img src="docs/assets/tokenjam-flow-band.svg" alt="Your agents (Claude Code, Codex, Cursor) feed TokenJam, which breaks into three stages: Observe (Lens), Optimize (Downsize, Cache, Trim, Script, Reuse), and Prove (Bench)." width="830"></div>

TokenJam reads telemetry from the major agent runtimes, frameworks, providers, and observability tools,
then runs a suite of analyzers over it. Most run for everyone. A few are persona-scoped, because the
fix they produce is only reachable from one side of the line: **CC** = coding agent (Claude Code,
Codex), **SDK** = your own SDK/API code. `tj optimize` runs everything your persona can act on; name a
subset with `tj optimize downsize resend relearn`.

| Analyzer | CC | SDK | Description |
|---|:--:|:--:|---|
| `relearn` | ✅ | ✅ | Blockers your agent keeps re-hitting across sessions, and what the repeated recovery costs |
| `resend` | ✅ | ✅ | How much of each turn's prompt is context you already sent, whether or not caching is on |
| `summarize` | ✅ | ✅ | Instruction files large enough to tax every session, scanned from disk |
| `downsize` | ✅ | ✅ | Sessions where a cheaper same-family model is a candidate. Never claims quality equivalence |
| `subagent` | ✅ | ✅ | Per-subagent cost hidden inside the parent session's total, and which dispatches ran over-powered |
| `deadweight` | ✅ | — | MCP servers whose schemas load into every session and are never called |
| `budget-projection` | ✅ | ✅ | Your run-rate against a configured `[budget.<provider>]` ceiling |
| `cache` | — | ✅ | Your caching ratio per (provider, model), and where it is worst |
| `cache-recommend` | — | ✅ | Where to place prompt-cache breakpoints, from the prefixes you actually repeat |
| `trim` | — | ✅ | Prompt regions the model gives little weight to |
| `verbosity` | — | ✅ | Sessions whose output runs long against a per-(tool, task-shape) baseline |
| `script` | — | ✅ | Deterministic tool sequences a plain script could replace |
| `reuse` | — | ✅ | Sessions where your agent re-plans work it has already planned |
| `stream-usage` | — | ✅ | Streamed calls that closed before the provider reported usage, so their spend went unrecorded |

**Why a row is dashed matters.** For a coding-agent user, `cache` / `cache-recommend` / `trim` /
`script` / `reuse` / `verbosity` / `stream-usage` are skipped before they ever query: the harness builds
the request and owns the prompt template, so the lever lives on the other side of the line, and a
finding with no fix is a diagnostic rather than a product. For an SDK user, `deadweight` is skipped
because the data it reads (project `.mcp.json`, `.claude/settings*.json`, on-disk Claude Code
transcripts) doesn't exist in a generic SDK process. The gate is one map in
`core/optimize/runner.py`; the reasoning for each entry is in the comment beside it.

`verbosity` is the one dash that is a product decision rather than a missing lever: its only remedy
for a coding agent is a global "be concise" instruction, which buys tokens by making the agent
terser everywhere. That is a quality tax, and it is not a trade this product makes.

A tick means the analyzer runs and its fix is reachable for that persona, not that it will fire on
your data. `script` and `reuse` in particular hold deliberately strict thresholds and stay quiet
unless your workload really does repeat itself.

Lens's FAQ screen carries the same list for *your* setup: the checks that are live, then the ones
turned off, each with its reason. Deep dives: [docs/optimize/](docs/optimize/).

---

## From advice to applied fixes

<div align="center"><img src="docs/assets/tokenjam-waste-grid.svg" alt="Where your tokens go: Expensive model (using Opus for a Haiku-level task) → downsize; Uncached repeats (sending the same base prompt 100s of times) → cache; Bloated prompts (re-sending the same long context every call) → trim; Verbose output (getting 500-word answers to yes/no questions) → verbosity; Repeated planning (re-planning the same task every day) → reuse; Don't need an LLM (paying a model to do what code could) → script." width="830"></div>

TokenJam used to stop at the left column: here is what's costing you. Now every analyzer terminates in
a concrete diff to a file on your machine, and one lifecycle drives all of them: **list → stage →
check → apply → undo**, dry-run by default, nothing written until you say so.

```bash
tj rules list           # every fix on offer, and the files it would be written into
tj rules show <id>      # the rule text, its destinations, and why each was chosen
tj rules stage <id>     # render one diff per destination, staged for review
tj rules check          # re-verify staged diffs against the files as they stand now
tj rules apply          # write them
tj rules undo <id>      # revert, per destination
```

Three properties worth knowing:

- **Placement is derived, not assumed.** A rule lands in the project that actually exhibited the
  problem, chosen from the working directories your sessions recorded, rather than defaulting to the
  global `~/.claude/CLAUDE.md` that every session of every project pays for.
- **Every write is netted against its own standing cost.** A `CLAUDE.md` rule is re-sent for the rest
  of the file's life, so a rule that merely breaks even is refused rather than offered. TokenJam caps
  how many permanent writes it will propose at once and ranks them by net value.
- **Reversible by construction.** Applied writes snapshot the pre-image and commit when the target is
  in a repo; `tj rules undo` and `tj relearn revert` restore it.

Fixes that aren't rules have their own verbs: `tj relearn apply` handles the skill and hook rungs,
`tj summarize` rewrites an oversized instruction file in place with its protected structure
hash-guarded, and `tj optimize --export-config` emits a routing config instead of editing anything.
Everything also appears in Lens's **Review** inbox if you'd rather click than type.

---

## Lens: the local dashboard

`tj serve` runs Lens at `http://127.0.0.1:7391/`: a **Dashboard** that lands you on recoverable waste and current health, with an embedded explorer to slice your usage any way (metric × dimension × chart), and a **Review inbox** where every proposed fix waits for an approve or a dismiss. Around them: Optimize (with its Summarize and Rules sub-screens), Sessions, Traces, Cost, Alerts, Drift, and Budget. The nav is persona-aware: it hides screens your persona has no data for. Fully offline, no signup.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/tj-dashboard.png" alt="Dashboard: recoverable waste, current health, and the embedded pivot explorer" /></td>
<td width="50%"><img src="docs/screenshots/tj-cost.png" alt="Cost: spend over time + cache savings" /></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/tj-traces.png" alt="Trace waterfall: session-level spans with cost annotations" /></td>
<td width="50%"><img src="docs/screenshots/tj-status.png" alt="Sessions: per-agent cards" /></td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/tj-dashboard-tools.png" alt="Analytics explorer: tool-usage leaderboard" /></td>
<td width="50%"><img src="docs/screenshots/tj-dashboard-leaderboard.png" alt="Analytics explorer: cost-by-model leaderboard" /></td>
</tr>
</table>

→ [tokenjam.dev/products/lens](https://tokenjam.dev/products/lens) for the visual walkthrough.

---

## Beyond optimization

TokenJam is also a full observability stack. The analyzers and Lens ride on top.

- **Real-time cost tracking**: every LLM call priced as it happens
- **Session replay**: `tj session-story` reconstructs a session's *method* turn by turn: its ordered moves, and for each delegation the subagent's mandate and what it did; `tj resume-brief` recaps where a session left off
- **Shareable efficiency card**: `tj tokenmaxx`
- **Safety alerts**: 13 alert types, 6 channels (ntfy, Discord, Telegram, webhook, file, stdout)
- **Behavioral drift detection**: Z-score baselines, no LLM required
- **Schema validation**: declare or infer JSON Schema for tool outputs
- **Context & quota audits**: `tj context` (re-read vs. net-new split) and `tj quota-audit` (retroactive Opus usage check) over your Claude Code sessions
- **Close the loop**: `tj loop` annotates a run with a verdict, promotes a bad run into a stored expectation, and tracks whether later runs pass or regress against it
- **Instruction-file summarization**: `tj summarize` finds files worth condensing, estimates the per-session saving, and rewrites them in place with protected structure hash-guarded and reversible
- **Enforcement-plane proxy (suggest mode)**: `tj proxy` surfaces routing suggestions locally, without rewriting requests
- **OTel-native**: point any OTLP exporter at `tj serve` and you're done
- **Statusline**: a zero-token Claude Code status line (`tj statusline`, wired by `tj onboard --claude-code`) showing this session's re-read share + a `/compact` nudge
- **MCP server**: in-request-path tools for **SDK / API** users (not Claude Code / Codex subscription users, since an in-loop MCP would be a per-turn quota burden there; they get the out-of-band statusline instead)

---

## Prove a swap holds: TokenJam Bench

`tj optimize downsize` flags *candidates*. It never claims the cheaper model would have produced the same answer. **[TokenJam Bench](https://github.com/Metabuilder-Labs/tokenjam-bench)** is the companion that checks. It runs your original and candidate models against real task suites and reports the pass-rate difference with statistics (Wilson CI + McNemar), so you get a hedged verdict ("holds" or "regressed") instead of a guess.

```bash
pip install tokenjam-bench
tjb run --original anthropic:claude-opus-4-7 --candidate anthropic:claude-haiku-4-5
```

Bench reports measured pass-rate on a suite, never "certified" or "quality preserved." Open source and local, like TokenJam. [Learn more →](https://github.com/Metabuilder-Labs/tokenjam-bench)

---

## Documentation

| Topic | Where |
|---|---|
| Getting started: every entry path, by persona | [docs/getting-started.md](docs/getting-started.md) |
| The first hour: what to do once data flows | [docs/first-hour.md](docs/first-hour.md) |
| Full CLI reference, every command and flag | [docs/cli-reference.md](docs/cli-reference.md) |
| Downsize / Cache / Script / Trim deep-dives | [docs/optimize/](docs/optimize/) |
| Reuse analyzer deep-dive | [docs/optimize/reuse.md](docs/optimize/reuse.md) |
| Prove a downsize candidate holds (TokenJam Bench) | [tokenjam-bench](https://github.com/Metabuilder-Labs/tokenjam-bench) |
| Claude Code & Codex integration | [docs/claude-code-integration.md](docs/claude-code-integration.md) |
| Claude Code vs. Codex vs. SDK vs. OTLP: capability matrix | [docs/agent-capability-matrix.md](docs/agent-capability-matrix.md) |
| Harness run grouping (governors / fan-out launchers) | [docs/harness-integration.md](docs/harness-integration.md) |
| Python SDK reference | [docs/python-sdk.md](docs/python-sdk.md) |
| TypeScript SDK reference | [docs/typescript-sdk.md](docs/typescript-sdk.md) |
| Framework support (LangChain / CrewAI / etc.), including the full OTel provider/framework matrix | [docs/framework-support.md](docs/framework-support.md) |
| Alert channels & rule reference | [docs/alerts.md](docs/alerts.md) |
| Backfill from Langfuse / Helicone / OTLP | [docs/backfill/](docs/backfill/) |
| Enforcement-plane proxy (suggest mode) | [docs/proxy/overview.md](docs/proxy/overview.md) |
| Policy rules | [docs/policy/overview.md](docs/policy/overview.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Architecture deep-dive | [docs/architecture.md](docs/architecture.md) |
| Installation extras (Trim, framework patches) | [docs/installation.md](docs/installation.md) |
| Export to Grafana / Datadog / NDJSON | [docs/export.md](docs/export.md) |
| NemoClaw sandbox observer | [docs/nemoclaw-integration.md](docs/nemoclaw-integration.md) |
| Release notes | [GitHub Releases](https://github.com/Metabuilder-Labs/tokenjam/releases) |

---

## Roadmap

**Shipped:** The analyzer suite (Downsize · Cache · Cache-recommend · Script · Trim · Reuse · Verbosity · Subagent right-sizing · Deadweight · Resend · Relearn · Summarize · Budget projection · Stream-usage) · Persona-scoped analyzer gating · The `tj rules` write lifecycle (stage / check / apply / undo, with derived placement and net-of-cost budgeting) · Review inbox · Claude Code + Codex onboarding · MCP server · Lens web UI · Backfill adapters (Langfuse, Helicone, OTLP) · Period comparison · Routing-config export · Read-only policy preview · Context & quota audits · Session replay & resume briefs · Close-the-loop annotations/expectations · Enforcement-plane proxy (suggest mode)

**Up next** (roughly):
- [ ] Continued Lens polish + per-product visual branding
- [ ] `tj policy add | edit | apply`: unified rule surface (today: `tj policy list` / `tj policy decisions`)
- [ ] `tj replay`: replay captured sessions against new model versions
- [ ] TypeScript framework patches (LangChain JS, OpenAI Agents SDK)
- [ ] Vercel AI SDK & Mastra integrations
- [ ] Published Docker image
- [ ] GitHub Actions for CI drift/cost checks

Full version-by-version history: [GitHub Releases](https://github.com/Metabuilder-Labs/tokenjam/releases).

---

## Contributing

TokenJam is MIT, and contributions are welcome: from a one-line pricing fix to a whole new framework integration. A few easy on-ramps:

- **[Good first issues →](https://github.com/Metabuilder-Labs/tokenjam/labels/good%20first%20issue)**: scoped, newcomer-friendly tasks, ready to pick up.
- **Bugs**: notice something off? File a bug.
- **Documentation**: struggled with something while getting started? Help the next person by writing or updating documentation.
- **Model pricing**: `tokenjam/pricing/models.toml` is community-maintained. Fix a rate or add a model in a single PR; no issue needed.
- **Framework integrations**: provider/framework patches follow one clear pattern (`tokenjam/sdk/integrations/anthropic.py` is the reference). Open an issue first to align on approach.
- **Coding Agents are first-class citizens**: TokenJam is built by Humans AND AI coding agents, and contributing with one is first-class. **Claude Code:** read [CLAUDE.md](CLAUDE.md) and run `/init` to bring your agent up to speed. **Codex / other agents:** [AGENTS.md](AGENTS.md) has the critical rules.

Setup and the full dev workflow are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

If TokenJam saves you tokens, **star it** and **watch for releases**; we ship often.

---

<div align="center">

**[tokenjam.dev](https://tokenjam.dev)** · [PyPI](https://pypi.org/project/tokenjam/) · [npm](https://www.npmjs.com/package/@tokenjam/sdk) · [TokenJam Bench](https://github.com/Metabuilder-Labs/tokenjam-bench) · [Issues](https://github.com/Metabuilder-Labs/tokenjam/issues)

MIT License · Built by [Metabuilder Labs](https://github.com/Metabuilder-Labs)

</div>
