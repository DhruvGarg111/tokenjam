"""GET/POST /api/v1/budget — the redesigned Budget page's two-zone response.

Pins the response SHAPE (`coding` zone grouped by tool, `sdk` zone grouped by
literal agent_id) and the write paths for each: a coding-tool group's daily
cap (`scope: "group:<id>"`), the coding-zone default (`scope:
"defaults_coding"`), the SDK-zone default (`scope: "defaults"`), and an SDK
workflow's own agent_id. Does not touch `POST /budget/provider` (a separate,
unrelated forecast concept).
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    AgentConfig,
    BudgetConfig,
    CodingGroupConfig,
    DefaultsConfig,
    GroupBudgetConfig,
    TjConfig,
    active_config_path,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import build_default_pipeline
from tests.factories import make_session


def _config(tmp_path, **kwargs) -> TjConfig:
    cfg = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)), **kwargs)
    path = tmp_path / "tokenjam.toml"
    path.write_text('version = "1"\n')
    cfg.config_path = path
    return cfg


def _client(config, db):
    app = create_app(config=config, db=db, ingest_pipeline=build_default_pipeline(db, config))
    return TestClient(app)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def test_get_budget_shape_has_coding_and_sdk_zones(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    assert resp.status_code == 200
    body = resp.json()
    assert "coding" in body
    assert "sdk" in body
    assert "groups" in body["coding"]
    assert "defaults" in body["coding"]
    assert "agents" in body["sdk"]
    assert "defaults" in body["sdk"]
    # Unchanged, unrelated concept — still present.
    assert "provider_budgets" in body
    assert "framing" in body


def test_coding_zone_groups_claude_code_projects_into_one_row(tmp_path, db):
    for aid in ["claude-code-proj-a", "claude-code-proj-b", "claude-code"]:
        db.upsert_session(make_session(agent_id=aid, session_id=f"s-{aid}"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code"}
    assert set(groups["claude-code"]["members"]) == {
        "claude-code-proj-a", "claude-code-proj-b", "claude-code",
    }


def test_codex_exact_id_forms_its_own_group_separate_from_claude_code(tmp_path, db):
    db.upsert_session(make_session(agent_id="claude-code-proj-a", session_id="s1"))
    db.upsert_session(make_session(agent_id="codex_exec", session_id="s2"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code", "codex"}
    assert groups["codex"]["members"] == ["codex_exec"]


def test_sdk_workflow_gets_its_own_row_not_grouped(tmp_path, db):
    db.upsert_session(make_session(agent_id="sdk-workflow-a", session_id="s1"))
    db.upsert_session(make_session(agent_id="sdk-workflow-b", session_id="s2"))
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    sdk_agents = resp.json()["sdk"]["agents"]
    assert set(sdk_agents.keys()) == {"sdk-workflow-a", "sdk-workflow-b"}


def test_only_present_or_explicitly_configured_groups_are_returned(tmp_path, db):
    """No data at all for codex, but it IS explicitly configured -- still
    shown (so a pre-set cap survives before the first session lands).
    claude-code has data but no explicit config -- also shown. Nothing else
    invented."""
    db.upsert_session(make_session(agent_id="claude-code-proj-a", session_id="s1"))
    config = _config(
        tmp_path,
        coding_agents={"codex": CodingGroupConfig(budget=GroupBudgetConfig(daily_usd=20.0))},
    )
    with _client(config, db) as client:
        resp = client.get("/api/v1/budget")
    groups = resp.json()["coding"]["groups"]
    assert set(groups.keys()) == {"claude-code", "codex"}
    assert groups["codex"]["members"] == []
    assert groups["codex"]["configured"]["daily_usd"] == 20.0


def test_post_group_scope_writes_the_group_cap(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "group:claude-code", "daily_usd": 50.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["coding"]["groups"]["claude-code"]["configured"]["daily_usd"] == 50.0
    assert body["coding"]["groups"]["claude-code"]["effective"]["daily_usd"] == 50.0


def test_post_group_scope_rejects_session_usd(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post(
            "/api/v1/budget",
            json={"scope": "group:claude-code", "daily_usd": 50.0, "session_usd": 5.0},
        )
    assert resp.status_code == 400
    assert "session_usd" in resp.json()["error"]


def test_post_defaults_coding_scope_writes_the_zone_default(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "defaults_coding", "daily_usd": 25.0})
    assert resp.status_code == 200
    assert resp.json()["coding"]["defaults"]["daily_usd"] == 25.0


def test_post_sdk_agent_scope_unchanged_daily_and_session(tmp_path, db):
    config = _config(tmp_path)
    with _client(config, db) as client:
        resp = client.post(
            "/api/v1/budget",
            json={"scope": "sdk-workflow-x", "daily_usd": 12.0, "session_usd": 3.0},
        )
    assert resp.status_code == 200
    sdk = resp.json()["sdk"]["agents"]["sdk-workflow-x"]
    assert sdk["configured"]["daily_usd"] == 12.0
    assert sdk["configured"]["session_usd"] == 3.0


def test_post_daily_only_preserves_an_existing_session_usd(tmp_path, db):
    """A save that only touches daily_usd must not erase a session_usd the
    user already had configured on that same SDK agent scope."""
    config = _config(
        tmp_path,
        agents={"legacy-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0, session_usd=1.5))},
    )
    with _client(config, db) as client:
        resp = client.post("/api/v1/budget", json={"scope": "legacy-agent", "daily_usd": 8.0})
    assert resp.status_code == 200
    sdk = resp.json()["sdk"]["agents"]["legacy-agent"]
    assert sdk["configured"]["daily_usd"] == 8.0
    assert sdk["configured"]["session_usd"] == 1.5
