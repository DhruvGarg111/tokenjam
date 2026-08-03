---
description: Python SDK instrumentation and integration rules.
paths:
  - "tokenjam/sdk/**"
  - "examples/single_provider/**"
  - "examples/single_framework/**"
---

# SDK rules

### Critical Rule 3 — `@watch()` alone does NOT create LLM spans

`@watch()` creates only session start/end spans. Provider patches (`patch_anthropic()`,
`patch_openai()`, etc.) are needed for individual LLM call spans.

### Critical Rule 12 — New SDK integrations must call `ensure_initialised()`

Every `patch_*()` convenience function must call
`from tokenjam.sdk.bootstrap import ensure_initialised; ensure_initialised()` before installing
hooks. This lazily bootstraps the TracerProvider + IngestPipeline on first use.
