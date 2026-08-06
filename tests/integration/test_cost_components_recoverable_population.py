"""The spend bar's two figures must cover ONE population.

`GET /cost/components` publishes a measured total and an estimated-recoverable
ceiling side by side, and the Optimize screen draws the second as a shaded
region of the first. That is a ratio, so the two figures have to be over the
same window and the same agents, or the shaded fraction is meaningless.

TWO DEFECTS, MEASURED ON THE FOUNDER'S REAL CORPUS, BOTH FIXED HERE.

1. **The window.** `total_recoverable_usd` comes out of the STORED analyzer
   report, computed on the daemon's own schedule over its own window (no route
   may run an analyzer, so it cannot follow an arbitrary `since`). It was
   paired with `total_cost_usd`, which DOES follow `since`. At `since=30d` the
   pair read 2611.92 / 13518.97 = 19.3%; at `since=7d` the numerator did not
   move by a cent while the denominator fell to 3575.29, so the same estimate
   shaded 73.1% of a week's spend as overspend. Narrower windows trend past
   100%.

2. **The scope.** The two figures also had to agree on WHICH AGENTS they cover.
   The stored report is corpus-wide, so a denominator narrowed by `agent_id`
   (or by the persona picker) would be a fraction of the spend the numerator
   was derived from.

`recoverable_basis_cost_usd` is the fix: measured spend over the report's own
window across every agent, i.e. the ceiling's own denominator. `total_cost_usd`
still answers the caller's window and is no longer a valid divisor for it.

WHY THE PERSONA DOES NOT SCOPE THE DENOMINATOR. The picker selects which
ANALYZERS contribute (`_collect_recoverable` filters findings by lever), not
which traffic is measured: every finding is computed over the whole corpus. On
the real corpus that distinction is stark. Scoping the denominator to the SDK
persona's own traffic gave $1.37 of spend against a $1,023.58 ceiling at 30d,
and $0.00 at 7d. `recoverable_basis_note` states the scope on the bar instead.

THE CORPUS BELOW IS DELIBERATELY ASYMMETRIC. Both defects were invisible on the
small purpose-built corpus a previous worker verified against, because there
every window and every persona covered roughly the same spend. Here the two
personas' traffic differs by an order of magnitude AND sits in different parts
of the window, so a denominator that silently followed either axis fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

# The stored report's window. Everything below is positioned relative to it.
REPORT_WINDOW_DAYS = 30

# Recoverable estimates, keyed by an analyzer each persona gates differently
# (`disabled_analyzers_for_persona`): `subagent` is claude-code-only, `reuse` is
# SDK-only. So the NUMERATOR genuinely differs per persona while the population
# it was measured over does not, which is the whole point of the fix.
CC_ONLY_USD = 55.0      # subagent
SDK_ONLY_USD = 120.0    # reuse
SHARED_USD = 12.0       # downsize, live for both


@dataclass
class _FakeFinding:
    past_overspend_usd: float | None
    past_overspend_tokens: int | None
    estimate_basis: str = "seeded"
    caveat: str = "seeded"


@dataclass
class _FakeReport:
    downgrade: object | None = None
    findings: dict = field(default_factory=dict)
    persona: str = "claude-code"


def _seed(db) -> dict[str, float]:
    """An asymmetric corpus, and the spend figures it implies.

    * A Claude Code agent spending heavily INSIDE the last 7 days.
    * A non-interactive (SDK) agent spending a tenth as much, and only OUTSIDE
      the last 7 days, so it is in the report's 30-day window but not in a
      7-day request window.

    A denominator that followed the request window would drop the SDK spend; one
    that followed the persona would drop nine tenths of the corpus or all of it.
    Both are separately detectable against the numbers returned here.
    """
    now = utcnow()
    # $0.03/span at 3000 input tokens on the seeded model is irrelevant: the
    # component split reprices from the pricing table, so the assertions below
    # compare API-reported figures against each other, never against a literal.
    for i in range(10):
        sid = f"cc{i}"
        db.upsert_session(make_session(session_id=sid, agent_id="claude-code-app"))
        db.insert_span(make_llm_span(
            session_id=sid, agent_id="claude-code-app", model="claude-opus-4-7",
            provider="anthropic", input_tokens=40_000, output_tokens=6_000,
            start_time=now - timedelta(days=1, hours=i),
        ))
    for i in range(3):
        sid = f"sdk{i}"
        db.upsert_session(make_session(session_id=sid, agent_id="billing-service"))
        db.insert_span(make_llm_span(
            session_id=sid, agent_id="billing-service", model="claude-opus-4-7",
            provider="anthropic", input_tokens=4_000, output_tokens=600,
            start_time=now - timedelta(days=20, hours=i),
        ))
    return {}


def _install_report(monkeypatch):
    """A stored report whose window is the LAST 30 DAYS, ending now.

    `stored_report` / `stored_report_block` are patched rather than a real
    artifact written, because what is under test is how the ROUTE pairs the
    stored figures with a measured denominator. Which findings survive the
    persona gate is `_collect_recoverable`'s job and is pinned in
    `tests/unit/test_cost_api_recoverable_presentation.py`.
    """
    from tokenjam.core.optimize import report_store

    now = utcnow()
    since = now - timedelta(days=REPORT_WINDOW_DAYS)
    # `downsize` rides the TYPED `downgrade` slot, not `findings` --
    # `_collect_recoverable` reads it from there and skips the findings entry of
    # that name. It is live for both personas, so it is what makes the ceiling a
    # sum of two or more estimates (and therefore what makes the overlap note
    # fire) under either side of the picker.
    report = _FakeReport(
        downgrade=_FakeFinding(SHARED_USD, 90_000),
        findings={
            "subagent": _FakeFinding(CC_ONLY_USD, 500_000),
            "reuse": _FakeFinding(SDK_ONLY_USD, 900_000),
        },
    )
    block = {
        "status": "ready",
        "computed_at": now.isoformat(),
        "window_days": REPORT_WINDOW_DAYS,
        "computing": False,
        "cycle_id": "test-cycle",
        "cycle_computing": False,
        "computed_build": "test", "build": "test", "build_provenance": {},
        "scan_since": since.isoformat(),
        "scan_until": now.isoformat(),
        "provenance": None,
        "degraded": False, "last_error": None, "last_error_at": None,
    }
    monkeypatch.setattr(report_store, "stored_report", lambda *a, **k: report)
    monkeypatch.setattr(report_store, "stored_report_block", lambda *a, **k: block)
    return since, now


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


async def _get(client, *, persona=None, since=None, agent_id=None) -> dict:
    params = {}
    if persona:
        params["persona"] = persona
    if since:
        params["since"] = since
    if agent_id:
        params["agent_id"] = agent_id
    r = await client.get("/api/v1/cost/components", params=params)
    assert r.status_code == 200, r.text
    return r.json()


PERSONAS = ("claude-code", "sdk")
WINDOWS = ("30d", "7d")


@pytest.mark.asyncio
async def test_the_corpus_is_asymmetric_enough_for_the_other_tests_to_bite():
    """A GUARD ON THE FIXTURE, not on the product.

    Both defects were shipped past a corpus where every window and persona
    covered the same spend, which made a wrong denominator indistinguishable
    from a right one. If a later edit flattens this fixture, the coupling tests
    below would still pass while testing nothing, so the asymmetry is asserted
    outright.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        wide = await _get(c, since="30d")
        narrow = await _get(c, since="7d")
        one_agent = await _get(c, since="30d", agent_id="billing-service")

    # The request window really does change measured spend.
    assert narrow["total_cost_usd"] < wide["total_cost_usd"] * 0.98
    # And a single agent really is a small slice of it, so an agent-scoped
    # denominator would be visibly wrong rather than coincidentally close.
    assert 0 < one_agent["total_cost_usd"] < wide["total_cost_usd"] * 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS)
async def test_the_ceiling_and_its_denominator_share_one_window(persona, monkeypatch):
    """DEFECT 2. The shaded fraction must not move when only the picker does.

    The numerator cannot follow the request window (it is a stored figure), so
    the denominator is pinned to the stored report's window instead. Asserted
    as a ratio because that is what the bar draws.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    since_dt, until_dt = _install_report(monkeypatch)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        by_window = {w: await _get(c, persona=persona, since=w) for w in WINDOWS}

    wide, narrow = by_window["30d"], by_window["7d"]

    # The requested window still moves the figure that ANSWERS the request.
    assert narrow["total_cost_usd"] < wide["total_cost_usd"]
    # The pair does not move: same numerator, same denominator, same ratio.
    assert narrow["total_recoverable_usd"] == wide["total_recoverable_usd"]
    assert narrow["recoverable_basis_cost_usd"] == wide["recoverable_basis_cost_usd"]
    assert narrow["recoverable_basis_tokens"] == wide["recoverable_basis_tokens"]
    ratios = {
        w: d["total_recoverable_usd"] / d["recoverable_basis_cost_usd"]
        for w, d in by_window.items()
    }
    assert ratios["7d"] == pytest.approx(ratios["30d"])
    # THE BAD STATE, pinned absent: the ceiling over the REQUEST window's total
    # is the ratio that used to be drawn, and on this corpus it differs.
    assert ratios["7d"] != pytest.approx(
        narrow["total_recoverable_usd"] / narrow["total_cost_usd"]
    )

    # And the window is published, so a reader can check it rather than trust it.
    for d in by_window.values():
        assert d["recoverable_basis_since"] == int(since_dt.timestamp())
        assert d["recoverable_basis_until"] == int(until_dt.timestamp())
        assert d["recoverable_window_days"] == REPORT_WINDOW_DAYS


@pytest.mark.asyncio
@pytest.mark.parametrize("window", WINDOWS)
async def test_the_ceiling_and_its_denominator_share_one_agent_population(
    window, monkeypatch,
):
    """DEFECT 1. The persona changes the LEVER SET, never the traffic measured.

    So the denominator is identical across personas (both cover the whole
    corpus, which is what the stored report covers) while the numerator is not.
    A denominator that tracked the persona's own traffic would differ here, and
    on the real corpus would divide a whole-corpus ceiling by 0.01% of the
    spend it came from.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    _install_report(monkeypatch)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        by_persona = {p: await _get(c, persona=p, since=window) for p in PERSONAS}
        # What the persona-partitioned spend actually is, for the contrast below.
        cc_only = await _get(c, since="30d", agent_id="claude-code-app")
        sdk_only = await _get(c, since="30d", agent_id="billing-service")

    cc, sdk = by_persona["claude-code"], by_persona["sdk"]

    # The numerator IS persona-specific: each persona's own levers, and only
    # those. (subagent is claude-code-only, reuse is SDK-only, downsize is both.)
    assert cc["total_recoverable_usd"] == pytest.approx(CC_ONLY_USD + SHARED_USD)
    assert sdk["total_recoverable_usd"] == pytest.approx(SDK_ONLY_USD + SHARED_USD)
    assert cc["total_recoverable_usd"] != sdk["total_recoverable_usd"]

    # The denominator is NOT: one population, one number, both personas.
    assert cc["recoverable_basis_cost_usd"] == sdk["recoverable_basis_cost_usd"]
    assert cc["recoverable_basis_tokens"] == sdk["recoverable_basis_tokens"]

    # THE BAD STATE, pinned absent: neither persona's own traffic is the
    # denominator. Both are real, both are wrong, and both are far enough from
    # the right answer that a coincidence cannot hide it.
    basis = cc["recoverable_basis_cost_usd"]
    assert basis > cc_only["total_cost_usd"]
    assert basis > sdk_only["total_cost_usd"] * 5


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS)
async def test_an_agent_filter_does_not_narrow_the_ceiling_s_denominator(
    persona, monkeypatch,
):
    """The same coupling on the other axis. The stored report is corpus-wide, so
    an `agent_id` that narrows the denominator would recreate the defect: a
    whole-corpus ceiling over one agent's spend."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    _install_report(monkeypatch)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        whole = await _get(c, persona=persona, since="30d")
        filtered = await _get(
            c, persona=persona, since="30d", agent_id="billing-service",
        )

    # The requested figure narrows, as it should.
    assert filtered["total_cost_usd"] < whole["total_cost_usd"]
    # The ceiling's own denominator does not.
    assert filtered["recoverable_basis_cost_usd"] == whole["recoverable_basis_cost_usd"]
    assert filtered["total_recoverable_usd"] == whole["total_recoverable_usd"]


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("window", WINDOWS)
async def test_the_disclosure_travels_with_the_pair(persona, window, monkeypatch):
    """Every qualification the figures need, on the payload beside them.

    The overlap note (the ceiling is not additive) was already required to
    travel with the bar; the basis note joins it, because the window mismatch
    was invisible precisely because nothing on screen said which window the
    shaded region came from. `largest_recoverable_analyzer` stays linked as the
    one entry that is honest standalone.
    """
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    _install_report(monkeypatch)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = await _get(c, persona=persona, since=window)

    assert d["recoverable_additive"] is False
    assert d["recoverable_overlap_note"], "the ceiling is a sum of overlapping estimates"
    assert d["largest_recoverable_analyzer"]
    assert d["largest_recoverable_usd"] is not None

    note = d["recoverable_basis_note"]
    assert note, "the bar must say which window and which agents it covers"
    # It names the window it is actually drawn over, and says the picker does
    # not move it. Both are the claims the old bar failed to make.
    assert str(REPORT_WINDOW_DAYS) in note
    assert "every agent" in note
    assert "does not follow the range selected above" in note
    # The persona sentence: what the picker DOES change, stated plainly, because
    # this bar's denominator is not persona-scoped and a reader must not assume
    # it is.
    assert "persona picker changes which analyzers" in note
    # House rule: no em dashes in user-facing copy.
    assert "—" not in note


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS)
async def test_a_cold_store_publishes_no_denominator_and_no_zero(persona):
    """A `0` denominator would render an unshaded full-width bar, which reads as
    "no waste found" -- the most reassuring thing this surface could say and the
    one an un-run scan has no evidence for. `None` is the only honest answer."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = await _get(c, persona=persona, since="30d")

    assert d["recoverable_available"] is False
    assert d["total_recoverable_usd"] is None
    assert d["recoverable_basis_cost_usd"] is None
    assert d["recoverable_basis_tokens"] is None
    assert d["recoverable_basis_since"] is None
    assert d["recoverable_basis_note"] == ""
    # The live measured total is unaffected: it never depended on the scan.
    assert d["total_cost_usd"] > 0
