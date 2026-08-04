"""The plugin lane: what an ENABLED, IN-SCOPE plugin costs every session.

Two gates decide whether a plugin costs anything and NEITHER is visible on the
filesystem. Measured on a real machine while building this: 1,299 SKILL.md files
were installed under ``~/.claude/plugins`` and 15 of them were resident — the
other 1,284 belonged to plugins that were switched off or scoped to one project.
Pricing installed-ness would have overstated the population by 87x.

The second gate is what gets counted rather than which plugin. A skill's BODY
arrives when the skill is invoked; its ``name: description`` line is listed
before anything is invoked. On the same machine those differ by more than three
orders of magnitude, so counting bodies is not a conservative overestimate — it
is a different number about a different thing.

Both are pinned below, plus the rule that plugin files never enter the summarize
catalog: they are third-party, under a versioned cache path, and the next plugin
update reverts any edit, so a "shorten this file" fix would appear to succeed
and then silently regress.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenjam.core import agent_config as ac

_SKILL = """---
name: {name}
description: {desc}
---

# {name}

{body}
"""


def _plugin(root: Path, marketplace: str, plugin: str, version: str, skills: int) -> Path:
    """An installed plugin with ``skills`` skills, each with a huge body.

    The bodies are deliberately enormous relative to the frontmatter: a test
    where they were similar in size could not tell "counted the listing" from
    "counted the file".
    """
    install = root / "plugins" / "cache" / marketplace / plugin / version
    for i in range(skills):
        skill_dir = install / "skills" / f"s{i}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _SKILL.format(
                name=f"{plugin}-skill-{i}",
                desc="Does one specific thing.",
                body="body prose that is never resident " * 400,
            ),
            encoding="utf-8",
        )
    return install


@pytest.fixture
def claude_dir(tmp_path):
    """Enabled+user, disabled+user, and enabled+project — the three cases."""
    root = tmp_path / ".claude"
    root.mkdir()
    installs = {
        "on@mkt": _plugin(root, "mkt", "on", "1.0.0", skills=3),
        "off@mkt": _plugin(root, "mkt", "off", "1.0.0", skills=40),
        "scoped@mkt": _plugin(root, "mkt", "scoped", "1.0.0", skills=25),
    }
    (root / "settings.json").write_text(json.dumps({
        "enabledPlugins": {"on@mkt": True, "off@mkt": False, "scoped@mkt": True},
    }), encoding="utf-8")
    (root / "plugins" / "installed_plugins.json").write_text(json.dumps({
        "version": 2,
        "plugins": {
            "on@mkt": [{"scope": "user", "installPath": str(installs["on@mkt"]),
                        "version": "1.0.0"}],
            "off@mkt": [{"scope": "user", "installPath": str(installs["off@mkt"]),
                         "version": "1.0.0"}],
            "scoped@mkt": [{"scope": "project", "projectPath": "/somewhere/else",
                            "installPath": str(installs["scoped@mkt"]),
                            "version": "1.0.0"}],
        },
    }), encoding="utf-8")
    return root


# --- The two gates ----------------------------------------------------------

def test_only_enabled_and_in_scope_plugins_are_priced(claude_dir):
    """THE gate test. Installed on disk is not evidence of being loaded."""
    by_name = {r.name: r for r in ac.scan_plugins(claude_dir=claude_dir)}
    assert set(by_name) == {"on@mkt", "off@mkt", "scoped@mkt"}

    assert by_name["on@mkt"].detail["resident"] is True
    assert by_name["on@mkt"].tokens > 0

    # Gate 1: switched off. 40 skills on disk, zero of them resident.
    assert by_name["off@mkt"].detail["resident"] is False
    assert by_name["off@mkt"].detail["skills"] == 40
    assert by_name["off@mkt"].tokens == 0
    assert "disabled" in by_name["off@mkt"].detail["not_resident_because"]

    # Gate 2: enabled, but installed for one project — it does not load here.
    assert by_name["scoped@mkt"].detail["resident"] is False
    assert by_name["scoped@mkt"].detail["skills"] == 25
    assert by_name["scoped@mkt"].tokens == 0
    assert "project-scoped" in by_name["scoped@mkt"].detail["not_resident_because"]


def test_a_gated_off_plugin_still_gets_a_row(claude_dir):
    """"Disabled, so free" and "we never looked" must not be the same absence.

    And the disabled rows are exactly what a user deciding whether to ENABLE
    something large needs to see.
    """
    rows = ac.scan_plugins(claude_dir=claude_dir)
    assert len(rows) == 3
    assert all(r.detail["not_resident_because"] or r.detail["resident"] for r in rows)


def test_skill_bodies_are_never_counted_as_resident(claude_dir):
    """The other half, and the one that would inflate the figure most.

    The fixture's bodies are ~13,000 characters each against a ~35-character
    listing line. If a body ever leaks into the resident figure this fails by
    two orders of magnitude, not by a rounding error.
    """
    row = {r.name: r for r in ac.scan_plugins(claude_dir=claude_dir)}["on@mkt"]
    skills = sorted((claude_dir / "plugins" / "cache" / "mkt" / "on").rglob("SKILL.md"))
    assert len(skills) == 3
    whole_files = sum(len(p.read_text(encoding="utf-8")) for p in skills)

    assert row.detail["resident_chars"] < whole_files / 50
    assert row.tokens < ac.tokens_for_chars(whole_files) / 50
    # And positively: it IS the name+description line, for every skill.
    expected = sum(len(ac.skill_listing_line(p)) for p in skills)
    assert row.detail["resident_chars"] == expected


def test_the_listing_line_is_frontmatter_only():
    """A helper that could return a body is one refactor from pricing one."""
    tmp = Path(__file__).parent
    path = tmp / "_plugin_probe_SKILL.md"
    path.write_text(_SKILL.format(
        name="probe", desc="A short description.", body="SECRET BODY TEXT " * 100,
    ), encoding="utf-8")
    try:
        line = ac.skill_listing_line(path)
        assert line == "probe: A short description."
        assert "SECRET BODY TEXT" not in line
    finally:
        path.unlink()


def test_plugin_usage_reads_recorded_counts_and_omits_the_unrecorded(tmp_path):
    home = tmp_path
    (home / ".claude.json").write_text(json.dumps({"pluginUsage": {
        "used@mkt": {"usageCount": 42, "lastUsedAt": 1},
        "never@mkt": {"usageCount": 0, "lastUsedAt": 1},
        "malformed@mkt": {"lastUsedAt": 1},
    }}), encoding="utf-8")
    usage = ac.plugin_usage(home)
    assert usage == {"used@mkt": 42, "never@mkt": 0}
    # A key with no recorded count is ABSENT, not zero: never-recorded is
    # absence of evidence and only a recorded zero is evidence of absence.
    assert "malformed@mkt" not in usage


# --- Through the analyzer ---------------------------------------------------

def _finding(claude_dir, tmp_path, usage: dict):
    from datetime import datetime, timedelta, timezone

    from tokenjam.core.optimize.analyzers import deadweight as dw

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".claude.json").write_text(json.dumps({
        "pluginUsage": {k: {"usageCount": v} for k, v in usage.items()},
    }), encoding="utf-8")

    root = tmp_path / "projects"
    project = root / "-repo"
    project.mkdir(parents=True)
    (project / "s0.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "cwd": str(project)},
        {"type": "assistant", "message": {
            "id": "m1", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }},
    ]), encoding="utf-8")
    now = datetime.now(timezone.utc)
    return dw.compute_deadweight_finding(
        now - timedelta(days=7), now + timedelta(days=1),
        projects_root=root, claude_home=home, claude_dir=claude_dir,
        store=ac.InMemoryAgentConfigStore(),
    )


def test_an_enabled_never_used_plugin_is_flagged_and_priced(claude_dir, tmp_path):
    finding = _finding(claude_dir, tmp_path, {"on@mkt": 0, "off@mkt": 0, "scoped@mkt": 0})

    assert [p.name for p in finding.dead_plugins] == ["on@mkt"]
    dead = finding.dead_plugins[0]
    assert dead.resident is True
    assert dead.resident_tokens > 0
    assert dead.estimated_tax_tokens_window > 0
    assert "name: description" in dead.tax_construction
    assert "BODIES are NOT counted" in dead.tax_construction
    assert "enabledPlugins" in dead.fix
    # The gated-off ones are visible but cost nothing, even at usage zero.
    assert finding.plugins_resident == 1
    assert len(finding.plugins) == 3
    assert finding.past_overspend_tokens == dead.estimated_tax_tokens_window


def test_a_used_plugin_is_not_flagged(claude_dir, tmp_path):
    finding = _finding(claude_dir, tmp_path, {"on@mkt": 12})
    assert finding.dead_plugins == []
    assert finding.past_overspend_tokens is None
    assert any("are actually resident" in n for n in finding.notes)


def test_a_plugin_with_no_recorded_usage_is_never_flagged(claude_dir, tmp_path):
    """Claude Code never recording a plugin is not evidence it was never used.

    Only a recorded zero is. Treating the two the same would flag a plugin
    installed five minutes ago and offer to disable it.
    """
    finding = _finding(claude_dir, tmp_path, {})
    assert finding.dead_plugins == []
    assert {p.usage_count for p in finding.plugins} == {None}


# --- The catalog exclusion --------------------------------------------------

def test_plugin_paths_never_enter_the_summarize_catalog():
    """A PIN, so a future change cannot quietly pull them in.

    Plugin files live under a versioned third-party cache path the user did not
    author. Offering to shorten one produces a fix that appears to succeed and
    is reverted by the next plugin update — a saving the product would keep
    claiming and never actually collect.
    """
    from tokenjam.core.summarize.catalog import load_catalog

    catalog = load_catalog()
    everything = (
        list(catalog.project_files) + list(catalog.project_globs)
        + list(catalog.global_paths)
    )
    offenders = [entry for entry in everything if "plugin" in entry.lower()]
    assert not offenders, (
        "plugin paths must never be summarize candidates — they are third-party "
        "files under a versioned cache path, and the next plugin update reverts "
        f"any edit: {offenders}"
    )


def test_the_summarize_scan_does_not_pick_up_a_plugin_skill(tmp_path, monkeypatch):
    """The behavioural half of the pin: not just absent from the catalog, but
    genuinely not scanned even when a plugin tree sits under the scan root."""
    from tokenjam.core.summarize import candidates

    project = tmp_path / "repo"
    (project / ".claude" / "plugins" / "cache" / "mkt" / "p" / "1.0" / "skills" / "s").mkdir(
        parents=True,
    )
    (project / ".claude" / "plugins" / "cache" / "mkt" / "p" / "1.0" / "skills" / "s"
     / "SKILL.md").write_text(_SKILL.format(
         name="p", desc="d", body="plugin body prose " * 300,
     ), encoding="utf-8")
    (project / "CLAUDE.md").write_text("real instructions " * 200, encoding="utf-8")

    scan = candidates.list_candidates(
        project_roots=[str(project)], config=None, include_global=False,
    )
    paths = [c.path for c in scan.candidates]
    assert any(p.endswith("CLAUDE.md") for p in paths)
    assert not [p for p in paths if "plugins" in p], paths


def test_the_plugin_lane_renders_even_with_no_mcp_servers(capsys):
    """A capability nobody has a path to does not exist.

    The deadweight renderer returns early when no MCP server is configured, and
    a user in that state can still be paying for an enabled plugin in every
    session. Pinned because the early return is exactly the kind of thing a
    later edit reinstates.
    """
    from tokenjam.cli.cmd_optimize import _render_deadweight
    from tokenjam.core.optimize.analyzers.deadweight import (
        DeadweightFinding,
        PluginDeadweight,
    )

    plugin = PluginDeadweight(
        name="on@mkt", enabled=True, install_scope="user", resident=True,
        not_resident_because="", skills=3, resident_tokens=90, usage_count=0,
        sessions_present=4, dead=True, estimated_tax_tokens_window=400,
        estimated_tax_usd_window=0.0012, priced_model="claude-sonnet-4-5",
        tax_construction="90 tok resident per call.",
        fix="Disable it in enabledPlugins.",
    )
    finding = DeadweightFinding(
        sessions_scanned=4, configured_servers=0,
        plugins=[plugin], dead_plugins=[plugin], plugins_resident=1,
    )
    _render_deadweight(finding, pricing_mode="api", marker="1")
    out = capsys.readouterr().out
    assert "no MCP server is" in out
    assert "on@mkt" in out
    assert "1 of 1 installed are resident" in out
    assert "enabledPlugins" in out
