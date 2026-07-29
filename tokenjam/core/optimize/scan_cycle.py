"""One scan cycle over every analyzer store the surfaces read.

THE DEFECT THIS EXISTS FOR. Three stores hold analyzer output — the full report
(``report_store``), the relearn detector's cache (``relearn_store``) and the
cost proposals (``cost_proposals``) — and every user-facing figure comes from
one of them. They were refreshed by three independently registered jobs, three
separate startup kicks, and two different on-demand endpoints that each covered
a different subset:

  * ``POST /optimize/rescan`` refreshed the REPORT only, and the Dashboard's
    and Optimize's auto-poll drove it — so those two surfaces kept their store
    warm by SCANNING on a five-minute timer.
  * the Review inbox's own Refresh button refreshed relearn AND cost proposals,
    but not the report.
  * the Review inbox polls too, but only to RE-READ. It never triggered a scan,
    so absent a human pressing Refresh its headline's only refresh was a 6h job.

The result was that "Rescan" meant something different depending on which
screen you pressed it from, and the Dashboard's waste tiles could be hours
fresher than the Review inbox headline they are naturally compared against —
with nothing on either surface disclosing it. Two figures for one metric that
age apart are the same class of defect as two figures computed on different
bases; the reader cannot see either one.

So there is ONE cycle. Every trigger — the schedule, the startup kick, and an
explicit rescan from any surface — refreshes all three stores, and
``scan_enabled`` gates all three rather than only the report (it is documented
as "keeps the daemon from ever scanning on its own", and relearn and the cost
proposals are scans).

ONE PASS, TWO VIEWS. The deeper defect the anchor work uncovered: a cycle ran
``build_report`` TWICE. The report store built one; ``recompute_cost_proposals``
built another. So an analyzer like ``subagent`` was computed twice per cycle, by
two separate scans of a database that ingestion keeps writing to, and the two
results were stored separately and then published side by side on two surfaces.
No amount of window or anchor agreement can make those match, because they read
the corpus at different moments — the figures were only ever as close as the gap
between two scans. The cycle now builds ONE report and hands it to the cost
adapters, which are a pure transformation of a report rather than a second
measurement of the corpus, so the two surfaces are identical BY CONSTRUCTION.

ONE ANCHOR, RESOLVED HERE. The window LENGTH is one seam already
(``core/optimize/report_window.py``); the instant it is subtracted from was
not. Each pass called ``utcnow()`` for itself, so two surfaces publishing one
metric covered windows whose edges sat wherever their threads happened to
start. That is unobservable in the stored artifacts and indistinguishable from
a real basis divergence, so the cycle resolves ONE ``until`` and hands it to
every pass. A store refreshed on its own (a lone trigger, a test) still owns
its own anchor: ``None`` means "you decide".

WHAT THIS DOES NOT DO. It does not run the passes sequentially. Each still
fires on its own daemon thread with its own connection, because they have very
different durations (relearn is a full-corpus scan with a distill pass; the
report is an analyzer sweep) and serialising them would make the slowest one
decide when the fastest one is allowed to be fresh. Each recompute keeps its
own overlap guard, so a cycle landing on top of a still-running pass costs
nothing. Nor does it give relearn the shared anchor: relearn's figure is
unbounded and its precomputed buckets are trailing windows from its OWN run
instant, which the inbox selects against by label rather than by bound.

NOT A POLLING HOOK. ``[optimize] scan_ui_poll_seconds`` is documented as "how
often a UI surface re-reads the stored result (NOT how often the scan runs)".
It is a READ cadence and no surface may drive a scan cycle from it. Doing so
put a full-corpus pass on a five-minute timer against passes that take minutes,
which is most of a duty cycle of continuous scanning; the surfaces now re-read
on that timer and the daemon owns when scanning happens.
"""
from __future__ import annotations

from typing import Any, Callable


def scan_enabled(config: Any) -> bool:
    """Whether the daemon may scan on its own at all.

    Gates the whole cycle. An explicit rescan from a surface is a human asking,
    so it is deliberately NOT gated here — the flag stops automatic scanning,
    it does not make the product unrefreshable.
    """
    optimize = getattr(config, "optimize", None)
    return bool(getattr(optimize, "scan_enabled", True))


def trigger_scan_cycle(
    backend_factory: Callable[[], Any], config: Any,
) -> dict[str, bool]:
    """Refresh every analyzer store. Returns which passes actually STARTED.

    Fire-and-forget: each recompute runs on its own daemon thread against its
    own fresh backend, so nothing here blocks a request thread or the daemon's
    bind path. A ``False`` in the result means that store's own overlap guard
    declined — a pass was already in flight — which is a no-op, not a failure.

    Never raises. One store failing to start must not stop the other two from
    refreshing; a store that raises on trigger is reported as not started and
    its own error channel records why (each recompute stores its exception so
    the surface can show "last refresh failed" rather than a stale figure with
    no explanation).
    """
    from tokenjam.core.optimize import relearn_store
    from tokenjam.utils.time_parse import utcnow

    # Resolved ONCE, before anything is dispatched, so every pass in this cycle
    # subtracts its window from the same instant.
    anchor = utcnow()
    started: dict[str, bool] = {}

    def _try(name: str, fn: Callable[[], bool]) -> None:
        try:
            started[name] = bool(fn())
        except Exception:  # noqa: BLE001 - one store must not sink the cycle
            started[name] = False

    _try("report_and_cost", lambda: _trigger_report_and_cost(
        backend_factory, config, anchor,
    ))
    _try("relearn", lambda: relearn_store.trigger_background_recompute(
        backend_factory, config=config,
    ))
    return started


def _trigger_report_and_cost(
    backend_factory: Callable[[], Any], config: Any, anchor: Any,
) -> bool:
    """The single analyzer pass, feeding BOTH the report and the cost stores.

    One thread, one backend, one ``build_report``. The report store is written
    first (it is the one every analyzer surface reads), then the SAME findings
    are adapted into cost proposals — see this module's docstring on why that
    has to be one measurement rather than two.

    Returns ``False`` when the report store's own overlap guard declined, which
    means a pass is already in flight and this cycle is a no-op.
    """
    import threading

    from tokenjam.core.optimize import cost_proposals, report_store

    if report_store.is_computing():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            stored = report_store.recompute_now(backend, config, until=anchor)
            if stored is None:
                # The overlap guard declined between the check above and here.
                return
            # The typed report the pass just wrote. `None` when the stored
            # payload cannot be rehydrated (corrupt, or written by a newer
            # producer); the cost recompute then builds its own, which is the
            # old behaviour and strictly better than publishing nothing.
            report = report_store.stored_report(config)
            cost_proposals.recompute_cost_proposals(
                backend, config, until=anchor, report=report,
            )
        except Exception:  # noqa: BLE001 - background job, never crash a thread
            pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="analyzer-scan-cycle", daemon=True).start()
    return True
