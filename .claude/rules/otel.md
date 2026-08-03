---
description: OTel semantic-convention constant discipline.
paths:
  - "tokenjam/otel/**"
  - "tokenjam/core/ingest.py"
  - "tokenjam/core/ingest_adapters/**"
  - "tokenjam/api/routes/spans.py"
  - "tokenjam/api/routes/logs.py"
  - "tokenjam/sdk/integrations/**"
---

# OTel rules

### Critical Rule 10 — Use semconv constants

Reference `GenAIAttributes` and `TjAttributes` from `tokenjam/otel/semconv.py` instead of hardcoding
OTel attribute name strings. `semconv.py` is pure constants with no internal imports, so importing it
from anywhere is free of layering risk.

OTLP parsing has exactly one home, `tokenjam/otel/otlp_parsing.py` — see `tokenjam/otel/CLAUDE.md`.
