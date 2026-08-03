---
description: REST API auth rules for the local tj serve surface.
paths:
  - "tokenjam/api/**"
  - "tokenjam/cli/cmd_serve.py"
---

# API rules

### Critical Rule 4 — Ingest auth

`POST /api/v1/spans` requires `Authorization: Bearer <ingest_secret>` from
`security.ingest_secret` in `tj.toml`. It is enforced by `IngestAuthMiddleware`, which returns a
`JSONResponse` directly — `HTTPException` does not propagate from `BaseHTTPMiddleware.dispatch`.
