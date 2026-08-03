# `tokenjam/core/rulewrite/`

The ONE rule-write lifecycle every rule-writing analyzer shares: `plan.py` (list), `apply.py`
(stage/check/apply/undo), `store.py` (staging + gzip backups), `types.py` (shapes),
`delivery.py` (the `DeliveryKind` seam). Pure domain — no `tokenjam.cli` / `tokenjam.api` imports.

Reuses `core/summarize/apply`'s guard model and `relearn_apply`'s block renderer rather than
reimplementing either; the store persists rendered OUTPUT, never a recipe, so what a reviewer
approved in the diff is byte-for-byte what apply writes.

Surfaced by `tj rules` (`cli/cmd_rules.py`) and `/api/v1/rules/*`.

`core/optimize/rule_placement.py` is the WHERE half — named that, not `placement`, because
`placement` is already a registered analyzer name for an unrelated question (the Batch API lane,
`analyzers/batch_placement.py`; Critical Rule 19).

`delivery.py` is the delivery-mechanism seam: a `DeliveryKind` owns its own renderer AND its own
pricer, and adding a mechanism is a registration plus two functions — never an edit to the staging,
diff, apply, undo or budget machinery, none of which may name a CLAUDE.md. See
`core/optimize/CLAUDE.md` for the full delivery-kind, hook-rails and path-scoped-rule contracts.
