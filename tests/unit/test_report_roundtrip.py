"""`report_to_dict` -> `report_from_dict` must not drop a single field.

Why this file exists
--------------------
`report_to_dict` is a generic recursive walk that serialises everything.
`report_from_dict` used to be ~330 lines of hand-written
`Cls(field=d.get("field"), ...)` builders — one argument list per finding, each
of which somebody had to keep in sync with its dataclass by hand. They were not
in sync, and nothing anywhere failed when they drifted: a field nobody named
simply came back as its dataclass default.

Measured casualties at the time this test was written: `resend` lost 19 fields
including `cost_of_waste_usd`, `cost_of_waste_tokens`, `cost_of_waste_basis`,
`rightsize_recoverable_usd`, `coverage_note` and `sessions_in_scope`; `relearn`
lost `past_reread_usd`, `past_reread_tokens`, `corpus_basis`, `window_days`,
`archived_sessions_scanned` and its four `below_threshold_*` fields; `cache`,
`script`, `reuse`, `trim`, `subagent` and `verbosity` each lost some too.

That was survivable only while the live path handed the renderer a real
`OptimizeReport` object and only the HTTP shim round-tripped. It stopped being
survivable the moment analyzer results started being served from a store,
because then EVERY consumer round-trips. A dropped dollar field comes back as
its default, and a surface renders a smaller number — or a zero — with no error
raised anywhere. A wrong figure that looks like a successful read is worse than
a read that visibly failed.

`hydrate_dataclass` now rebuilds by introspecting each dataclass's own fields,
so there is no argument list left to drift. This test is what keeps it that
way: it populates every field of every finding with a non-default sentinel and
fails if the trip back changes any of them. A field added to an analyzer
tomorrow is covered automatically — that is the whole point, since the previous
failure mode was precisely "nobody remembered to add it".
"""
from __future__ import annotations

import dataclasses
import types
import typing
from datetime import datetime

import pytest

from tokenjam.core.optimize.runner import (
    _finding_class_for,
    finding_class_names,
    hydrate_dataclass,
    report_from_dict,
    report_to_dict,
)
from tokenjam.core.optimize.types import (
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    WindowSummary,
)
from tokenjam.utils.time_parse import utcnow

# Depth cap for recursive/self-referential shapes. Two levels is enough to
# exercise nested dataclasses and lists of them without unbounded recursion.
_MAX_DEPTH = 2


def _sentinel(hint: object, name: str, depth: int) -> object:
    """A NON-DEFAULT value for `hint`, so a dropped field is always detectable.

    Sentinels matter more than they look: if the value happened to equal the
    dataclass default, a field the round-trip drops would still compare equal
    and the test would pass while the bug shipped.
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)

    if isinstance(hint, types.UnionType) or origin is typing.Union:
        inner = [a for a in args if a is not type(None)]   # noqa: E721
        return _sentinel(inner[0], name, depth) if len(inner) == 1 else None
    if origin in (list, tuple) and args:
        return [] if depth >= _MAX_DEPTH else [_sentinel(args[0], name, depth + 1)]
    if origin is dict and len(args) == 2:
        return {} if depth >= _MAX_DEPTH else {"k": _sentinel(args[1], name, depth + 1)}
    if hint is bool:
        return True
    if hint is int:
        return 4242
    if hint is float:
        return 42.42
    if hint is str:
        return f"sentinel-{name}"
    if isinstance(hint, type) and issubclass(hint, datetime):
        return utcnow()
    if dataclasses.is_dataclass(hint):
        return _populate(hint, depth + 1)
    return None


def _populate(cls: object, depth: int = 0) -> object:
    """An instance of `cls` with EVERY field explicitly set."""
    hints = typing.get_type_hints(cls)
    kwargs = {
        f.name: _sentinel(hints.get(f.name, f.type), f.name, depth)
        for f in dataclasses.fields(cls)
    }
    return cls(**kwargs)


def _lossy_fields(cls: object) -> list[str]:
    """Field names whose value does not survive to_dict -> from_dict."""
    obj = _populate(cls)
    back = hydrate_dataclass(cls, report_to_dict(obj))
    return [
        f.name for f in dataclasses.fields(cls)
        if report_to_dict(getattr(obj, f.name)) != report_to_dict(getattr(back, f.name))
    ]


@pytest.mark.parametrize("name", finding_class_names())
def test_no_finding_field_is_dropped_by_the_round_trip(name):
    """Every field of every finding survives. Parametrized over the live class
    table rather than a hand-listed set, so a NEW analyzer is covered the day
    it registers — the previous bug was exactly a field nobody added by hand."""
    cls = _finding_class_for(name)
    assert cls is not None, f"{name} has no class in the round-trip table"
    lost = _lossy_fields(cls)
    assert lost == [], f"{name} loses {lost} on the way back from the store"


@pytest.mark.parametrize("cls", [WindowSummary, DowngradeFinding, BudgetProjection])
def test_no_top_level_dataclass_field_is_dropped(cls):
    """The report's own dataclasses drifted too — `WindowSummary.active_days`
    was dropped by the hand-written version, so a stored report came back
    claiming a different number of active days than the one that was measured."""
    lost = _lossy_fields(cls)
    assert lost == [], f"{cls.__name__} loses {lost} on the way back from the store"


def test_a_whole_report_round_trips_byte_identically():
    report = OptimizeReport(
        window=_populate(WindowSummary),
        downgrade=_populate(DowngradeFinding),
        budgets=[_populate(BudgetProjection)],
        notes=["a note"],
        findings={n: _populate(_finding_class_for(n)) for n in finding_class_names()},
        persona="sdk",
    )
    once = report_to_dict(report)
    twice = report_to_dict(report_from_dict(once))
    assert twice == once


def test_every_registered_analyzer_can_be_rebuilt():
    """A finding name absent from the class table is dropped SILENTLY on the
    way back (forward-compatibility for a newer daemon). That is correct for an
    unknown name and a data-loss bug for a name this install actually produces,
    so every registered analyzer must be rebuildable.

    `downsize` and `budget-projection` are excluded deliberately: they occupy
    typed top-level slots (`downgrade` / `budgets`) rather than the `findings`
    dict, and are covered by the top-level test above.
    """
    from tokenjam.core.optimize import ANALYZER_REGISTRY

    typed_slots = {"downsize", "budget-projection"}
    rebuildable = set(finding_class_names())
    missing = sorted(set(ANALYZER_REGISTRY) - typed_slots - rebuildable)
    assert missing == [], (
        f"analyzers {missing} produce findings the round-trip cannot rebuild; "
        f"add them to _build_finding_classes() in core/optimize/runner.py"
    )


def test_an_unknown_finding_name_is_dropped_rather_than_raising():
    """The forward-compatibility half of the same rule: a daemon newer than
    this install may advertise a finding it cannot render, and that must not
    break the whole report."""
    report = report_from_dict({
        "window": report_to_dict(_populate(WindowSummary)),
        "findings": {"a-finding-from-the-future": {"whatever": 1}},
        "persona": "sdk",
    })
    assert "a-finding-from-the-future" not in report.findings
    assert report.persona == "sdk"
