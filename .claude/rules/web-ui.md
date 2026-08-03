---
description: Lens SPA offline-first requirement — every dependency vendored, zero render-time external HTTP.
paths:
  - "tokenjam/ui/**"
  - "tests/unit/test_ui_offline.py"
  - "tokenjam/api/app.py"
---

# Web UI rules

### Critical Rule 18 — Web UI must work fully offline

`tokenjam/ui/index.html` is the served dashboard ("TokenJam Lens"; see `tokenjam/ui/CLAUDE.md`). It is
intentionally a single-file SPA with **zero external HTTP loads at render time**. Preact + hooks + htm
+ **uPlot** are vendored under `tokenjam/ui/vendor/` (ESM via `<script type="importmap">`; uPlot as a
plain `<script>` IIFE global); fonts use system-font fallbacks (no Google Fonts); the favicon is
inlined as a `data:` URL. The FastAPI app mounts `/ui/vendor` as `StaticFiles`. The
`tests/unit/test_ui_offline.py` regression test asserts no render-time external URLs exist anywhere
outside `<a href>` (clickable links to github.com are fine — they only fetch on click) and that
vendored CSS has no external `url()`. If you add a CDN font, script, or stylesheet, that test will
fail. Vendor the asset locally instead. See issue #87 + PR #88.
