"""Tests for tokenjam.core.ingest — sanitizer, pipeline, session resolution, capture stripping."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tokenjam.core.config import TjConfig, SecurityConfig, CaptureConfig, AgentConfig
from tokenjam.core.ingest import (
    IngestPipeline,
    SpanRejectedError,
    SpanSanitizer,
)
from tokenjam.core.models import NormalizedSpan, SessionRecord
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tests.factories import (
    make_invoke_agent_span,
    make_llm_span,
    make_session,
    make_tool_span,
)


# ---------------------------------------------------------------------------
# Minimal in-memory storage stub — named _StubBackend to avoid collision
# with tokenjam.core.db.InMemoryBackend, which is the real DuckDB-backed
# in-memory backend used in integration tests.
# ---------------------------------------------------------------------------

class _StubBackend:
    """Stub StorageBackend that stores everything in dicts/lists."""

    def __init__(self) -> None:
        self.spans: list[NormalizedSpan] = []
        self.sessions: dict[str, SessionRecord] = {}

    def insert_span(self, span: NormalizedSpan) -> None:
        self.spans.append(span)

    def upsert_session(self, session: SessionRecord) -> None:
        self.sessions[session.session_id] = session

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None:
        for s in self.sessions.values():
            if s.conversation_id == conversation_id:
                return s
        return None

    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]:
        return [s for s in self.spans if s.trace_id == trace_id]

    def get_span(self, trace_id: str, span_id: str) -> NormalizedSpan | None:
        for s in self.spans:
            if s.trace_id == trace_id and s.span_id == span_id:
                return s
        return None

    def get_session_id_for_trace(self, trace_id: str) -> str | None:
        for s in self.spans:
            if s.trace_id == trace_id and s.session_id:
                return s.session_id
        return None

    def get_session_ids_for_trace(self, trace_id: str) -> list[str]:
        return list(dict.fromkeys(
            s.session_id for s in self.spans
            if s.trace_id == trace_id and s.session_id
        ))

    def get_marker_session_ids_for_trace(self, trace_id: str) -> list[str]:
        return sorted(list(dict.fromkeys(
            s.session_id for s in self.spans
            if s.trace_id == trace_id
            and s.session_id
            and s.attribution_step in {None, "explicit", "conversation"}
        )))

    def reconcile_trace_session_attribution(self, trace_id: str) -> None:
        marker_ids = self.get_marker_session_ids_for_trace(trace_id)
        if not marker_ids:
            return

        trace_spans = [s for s in self.spans if s.trace_id == trace_id]
        parent_map = {s.span_id: s.parent_span_id for s in trace_spans if s.span_id}
        explicit_session_map = {
            s.span_id: s.session_id
            for s in trace_spans
            if s.span_id and s.session_id
            and (s.attribution_step in {"explicit", "conversation"} or (s.attribution_step is None and s.session_id in marker_ids))
        }

        resolved_cache: dict[str, str | None] = {}

        def resolve_chain(span_id: str, visited: set[str]) -> str | None:
            if span_id in resolved_cache:
                return resolved_cache[span_id]
            if span_id in explicit_session_map:
                return explicit_session_map[span_id]
            p_id = parent_map.get(span_id)
            if not p_id or p_id in visited or len(visited) >= 128:
                return None
            visited.add(p_id)
            res = resolve_chain(p_id, visited)
            resolved_cache[span_id] = res
            return res

        old_session_ids: set[str] = set()
        for span in trace_spans:
            if span.attribution_step in {"explicit", "conversation"}:
                continue
            chain_sess = resolve_chain(span.span_id, set())
            if chain_sess is not None:
                new_sess = chain_sess
                new_step = "step1_parent"
            elif len(marker_ids) == 1:
                new_sess = marker_ids[0]
                new_step = "step2_marker"
            else:
                new_sess = None
                new_step = "step3_unattributed"

            if span.session_id != new_sess or span.attribution_step != new_step:
                if span.session_id is not None:
                    old_session_ids.add(span.session_id)
                span.session_id = new_sess
                span.attribution_step = new_step

        for session_id in old_session_ids - set(marker_ids):
            if not any(s.session_id == session_id for s in self.spans):
                sess = self.sessions.get(session_id)
                if sess:
                    sess.status = "superseded"
                    sess.input_tokens = 0
                    sess.output_tokens = 0
                    sess.cache_tokens = 0
                    sess.cache_write_tokens = 0
                    sess.total_cost_usd = 0.0
                    sess.tool_call_count = 0

    def get_unattributed_spend(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        unatt_spans = [
            s for s in self.spans
            if s.session_id is None
            and (agent_id is None or s.agent_id == agent_id)
            and (since is None or (s.start_time and s.start_time >= since))
            and (until is None or (s.start_time and s.start_time <= until))
        ]
        spend_usd = sum(s.cost_usd or 0.0 for s in unatt_spans)
        trace_ids = {s.trace_id for s in unatt_spans if s.trace_id}
        return {
            "cost_usd": spend_usd,
            "spend_usd": spend_usd,
            "span_count": len(unatt_spans),
            "trace_count": len(trace_ids),
        }


# ---------------------------------------------------------------------------
# No-op hook stubs
# ---------------------------------------------------------------------------

class NoopCostEngine:
    def process_span(self, span: NormalizedSpan) -> None:
        pass


class NoopAlertEngine:
    def evaluate(self, span: NormalizedSpan) -> None:
        pass


class NoopSchemaValidator:
    def validate(self, span: NormalizedSpan) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    *,
    security: SecurityConfig | None = None,
    capture: CaptureConfig | None = None,
    db: _StubBackend | None = None,
    agents: dict | None = None,
) -> tuple[IngestPipeline, _StubBackend]:
    """Create an IngestPipeline with sensible defaults for testing."""
    db = db or _StubBackend()
    config = TjConfig(
        version="1",
        security=security or SecurityConfig(),
        capture=capture or CaptureConfig(),
        agents=agents or {},
    )
    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=NoopCostEngine(),
        alert_engine=NoopAlertEngine(),
        schema_validator=NoopSchemaValidator(),
    )
    return pipeline, db


# ===========================================================================
# SpanSanitizer tests
# ===========================================================================

class TestSpanSanitizer:

    def test_passes_valid_span(self):
        sanitizer = SpanSanitizer(SecurityConfig())
        # Should not raise
        sanitizer.validate({"key": "value", "count": 42})

    def test_rejects_too_many_attributes(self):
        config = SecurityConfig(max_attributes_per_span=5)
        sanitizer = SpanSanitizer(config)
        attrs = {f"key_{i}": i for i in range(10)}
        with pytest.raises(SpanRejectedError, match="10 attributes"):
            sanitizer.validate(attrs)

    def test_rejects_oversized_attribute(self):
        config = SecurityConfig(max_attribute_bytes=100)
        sanitizer = SpanSanitizer(config)
        attrs = {"big": "x" * 200}
        with pytest.raises(SpanRejectedError, match="bytes"):
            sanitizer.validate(attrs)

    def test_rejects_deeply_nested_attributes(self):
        config = SecurityConfig(max_attribute_depth=3)
        sanitizer = SpanSanitizer(config)
        # Build nesting: {"a": {"b": {"c": {"d": 1}}}} = depth 4
        nested: dict = {"d": 1}
        for key in ["c", "b", "a"]:
            nested = {key: nested}
        with pytest.raises(SpanRejectedError, match="nesting depth"):
            sanitizer.validate(nested)

    def test_passes_at_exact_depth_limit(self):
        config = SecurityConfig(max_attribute_depth=3)
        sanitizer = SpanSanitizer(config)
        # depth 3: {"a": {"b": {"c": 1}}}
        attrs = {"a": {"b": {"c": 1}}}
        sanitizer.validate(attrs)  # Should not raise

    def test_empty_attributes_pass(self):
        sanitizer = SpanSanitizer(SecurityConfig())
        sanitizer.validate({})  # Should not raise


# ===========================================================================
# Session resolution tests
# ===========================================================================

class TestSessionResolution:

    def test_conversation_id_resolves_to_existing_session(self):
        pipeline, db = _make_pipeline()

        # Pre-create a session with a known conversation_id
        existing_session = make_session(
            session_id="sess-original",
            conversation_id="conv-1",
        )
        db.upsert_session(existing_session)

        # Ingest a span with the same conversation_id
        span = make_llm_span(conversation_id="conv-1")
        pipeline.process(span)

        # The span should have been assigned to the existing session
        assert db.spans[-1].session_id == "sess-original"

    def test_new_conversation_id_creates_new_session(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(conversation_id="conv-new")
        pipeline.process(span)

        # A new session should have been created
        assert len(db.sessions) == 1
        session = list(db.sessions.values())[0]
        assert session.conversation_id == "conv-new"

    def test_span_with_existing_session_id_keeps_it(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(session_id="my-session")
        pipeline.process(span)

        assert db.spans[-1].session_id == "my-session"

    def test_cache_write_tokens_aggregate_separately_from_reads(self):
        # Cache reads accumulate into cache_tokens; cache writes/creation into
        # cache_write_tokens. The two must never be conflated.
        pipeline, db = _make_pipeline()

        pipeline.process(make_llm_span(
            conversation_id="conv-cache", cache_tokens=100, cache_write_tokens=40,
        ))
        pipeline.process(make_llm_span(
            conversation_id="conv-cache", cache_tokens=200, cache_write_tokens=10,
        ))

        session = db.get_session_by_conversation("conv-cache")
        assert session is not None
        assert session.cache_tokens == 300            # reads only
        assert session.cache_write_tokens == 50       # writes only

    def test_span_without_session_or_conversation_gets_new_session(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(conversation_id=None)
        span.session_id = None
        pipeline.process(span)

        assert db.spans[-1].session_id is not None
        assert len(db.sessions) == 1

    def test_trace_id_resolves_to_session_of_sibling_span(self):
        """Raw-OTLP cost spans without session_id attach to the session of a
        sibling span on the same trace (#326).

        Scenario: a fan-out harness emits an invoke_agent marker with a
        session_id on trace T, then emits cost-bearing gen_ai.llm.call spans
        on the same trace T but *without* a session_id.  _resolve_session must
        attach those cost spans to the session from the marker span rather than
        minting a fresh throwaway session for each.
        """
        trace_id = "trace-fanout-1"
        pipeline, db = _make_pipeline()

        # 1. Marker span — carries session_id "s1" and the shared trace_id.
        marker = make_invoke_agent_span(session_id="s1", trace_id=trace_id)
        pipeline.process(marker)
        assert len(db.sessions) == 1

        # 2. Cost span — same trace_id, no session_id, no conversation_id.
        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        # Both spans must land on the same session.
        assert db.spans[-1].session_id == "s1"
        # No new session should have been created.
        assert len(db.sessions) == 1

    def test_shared_trace_does_not_choose_an_arbitrary_session(self):
        """Ambiguous trace membership must not assign one session's cost to
        another session merely because its marker arrived first.
        """
        trace_id = "trace-shared-by-two-sessions"
        pipeline, db = _make_pipeline()

        pipeline.process(make_invoke_agent_span(session_id="w1", trace_id=trace_id))
        pipeline.process(make_invoke_agent_span(session_id="w2", trace_id=trace_id))

        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        assert db.spans[-1].session_id is None
        assert len(db.sessions) == 2

    def test_parent_session_wins_when_trace_has_multiple_sessions(self):
        """A parent marker identifies the child even when another session's
        marker was inserted first on the same trace.
        """
        trace_id = "trace-parent-disambiguates"
        pipeline, db = _make_pipeline()

        w2 = make_invoke_agent_span(session_id="w2", trace_id=trace_id)
        pipeline.process(w2)
        w1 = make_invoke_agent_span(session_id="w1", trace_id=trace_id)
        pipeline.process(w1)

        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        cost_span.parent_span_id = w1.span_id
        pipeline.process(cost_span)

        assert db.spans[-1].session_id == "w1"

    def test_spans_on_different_traces_get_independent_sessions(self):
        """Spans on different traces are not accidentally merged (#326 guard).

        Even when two spans lack a session_id, if their trace_ids differ they
        must each get their own independent session.
        """
        pipeline, db = _make_pipeline()

        span_a = make_llm_span(trace_id="trace-A")
        span_a.session_id = None
        span_a.conversation_id = None
        pipeline.process(span_a)

        span_b = make_llm_span(trace_id="trace-B")
        span_b.session_id = None
        span_b.conversation_id = None
        pipeline.process(span_b)

        session_id_a = db.spans[0].session_id
        session_id_b = db.spans[1].session_id
        assert session_id_a is not None
        assert session_id_b is not None
        assert session_id_a != session_id_b
        assert len(db.sessions) == 2

    def test_cost_span_before_marker_is_reparented_when_marker_arrives(self):
        """A late marker must not leave a trace-only cost span in a permanent
        throwaway session.
        """
        trace_id = "trace-reverse-order"
        pipeline, db = _make_pipeline()

        # 1. Cost span arrives first — no sibling with a session_id yet.
        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        provisional_session_id = db.spans[0].session_id
        assert provisional_session_id is not None
        assert len(db.sessions) == 1

        # 2. Marker span arrives second with its own session_id.
        marker = make_invoke_agent_span(session_id="s-marker", trace_id=trace_id)
        pipeline.process(marker)

        # The late marker absorbs the provisional trace-only span.
        assert db.spans[0].session_id == "s-marker"
        assert db.spans[1].session_id == "s-marker"
        assert db.sessions[provisional_session_id].status == "superseded"
        assert db.sessions[provisional_session_id].total_cost_usd == 0.0

    def test_parent_derived_span_is_reparented_with_provisional_parent(self):
        """Children of a provisional span must follow its late marker
        reconciliation instead of retaining the throwaway session.
        """
        trace_id = "trace-parent-reverse-order"
        pipeline, db = _make_pipeline()

        parent = make_llm_span(trace_id=trace_id)
        parent.session_id = None
        parent.conversation_id = None
        pipeline.process(parent)
        child = make_llm_span(trace_id=trace_id)
        child.session_id = None
        child.conversation_id = None
        child.parent_span_id = parent.span_id
        pipeline.process(child)
        provisional_session_id = db.spans[0].session_id
        assert provisional_session_id is not None

        pipeline.process(make_invoke_agent_span(session_id="s-marker", trace_id=trace_id))

        assert db.spans[0].session_id == "s-marker"
        assert db.spans[1].session_id == "s-marker"
        assert db.sessions[provisional_session_id].status == "superseded"

    def test_multi_marker_reverse_arrival_resolves_parent_chain(self):
        """When multiple markers exist, a child span arriving before its parent
        marker must resolve to its parent's session (Step 1), not be wiped to
        unattributed (Step 3).
        """
        trace_id = "trace-multi-marker-reverse"
        pipeline, db = _make_pipeline()

        # Child 1 arrives first; its parent will be marker 1
        c1 = make_llm_span(trace_id=trace_id)
        c1.span_id = "c1"
        c1.parent_span_id = "m1"
        c1.session_id = None
        c1.conversation_id = None
        pipeline.process(c1)

        # Unparented arrives second
        u = make_llm_span(trace_id=trace_id)
        u.span_id = "u"
        u.parent_span_id = None
        u.session_id = None
        u.conversation_id = None
        pipeline.process(u)

        # Marker 1 arrives third
        m1 = make_invoke_agent_span(session_id="sess-1", trace_id=trace_id)
        m1.span_id = "m1"
        pipeline.process(m1)

        # Marker 2 arrives fourth
        m2 = make_invoke_agent_span(session_id="sess-2", trace_id=trace_id)
        m2.span_id = "m2"
        pipeline.process(m2)

        stored_c1 = next(s for s in db.spans if s.span_id == "c1")
        stored_u = next(s for s in db.spans if s.span_id == "u")

        # Child 1 resolved via parent chain to sess-1 (Step 1 wins over Step 3)
        assert stored_c1.session_id == "sess-1"
        assert stored_c1.attribution_step == "step1_parent"

        # Unparented span on >1 marker trace resolved to unattributed (Step 3)
        assert stored_u.session_id is None
        assert stored_u.attribution_step == "step3_unattributed"

    def test_generic_otlp_sole_marker_with_custom_name_resolves_child_spans(self):
        """Generic OTLP markers with custom names (e.g. 'workflow') must act
        as sole markers when exactly one exists on the trace, resolving child spans
        via step2_marker (#749).
        """
        trace_id = "trace-otlp-custom-sole"
        pipeline, db = _make_pipeline()

        marker = make_invoke_agent_span(session_id="wf-session", trace_id=trace_id)
        marker.name = "workflow"
        pipeline.process(marker)
        assert len(db.sessions) == 1

        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        assert db.spans[-1].session_id == "wf-session"
        assert db.spans[-1].attribution_step == "step2_marker"
        assert len(db.sessions) == 1

    def test_generic_otlp_multi_marker_with_custom_names_resolves_to_unattributed(self):
        """Generic OTLP traces with multiple custom-named session roots
        (e.g. 'workflow' and 'agent.run') must recognize all markers and resolve
        unparented spans to step3_unattributed (#749).
        """
        trace_id = "trace-otlp-custom-multi"
        pipeline, db = _make_pipeline()

        m1 = make_invoke_agent_span(session_id="wf-1", trace_id=trace_id)
        m1.name = "workflow"
        pipeline.process(m1)

        m2 = make_invoke_agent_span(session_id="wf-2", trace_id=trace_id)
        m2.name = "agent.run"
        pipeline.process(m2)

        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        assert db.spans[-1].session_id is None
        assert db.spans[-1].attribution_step == "step3_unattributed"
        assert len(db.sessions) == 2

    def test_generic_otlp_reverse_arrival_with_custom_name_triggers_reconciliation(self):
        """Late arrival of a custom-named marker span (e.g. 'workflow') must trigger
        trace reconciliation, reparenting provisional child spans to step2_marker (#749).
        """
        trace_id = "trace-otlp-custom-reverse"
        pipeline, db = _make_pipeline()

        # 1. Cost span arrives first — provisional session minted
        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        provisional_id = db.spans[0].session_id
        assert provisional_id is not None
        assert db.spans[0].attribution_step == "provisional"
        assert len(db.sessions) == 1

        # 2. Custom-named marker arrives late
        marker = make_invoke_agent_span(session_id="wf-late", trace_id=trace_id)
        marker.name = "workflow"
        pipeline.process(marker)

        # Provisional span reparented to custom marker session via step2_marker
        assert db.spans[0].session_id == "wf-late"
        assert db.spans[0].attribution_step == "step2_marker"
        assert db.spans[1].session_id == "wf-late"
        assert db.sessions[provisional_id].status == "superseded"
        assert db.sessions[provisional_id].total_cost_usd == 0.0

    def test_generic_otlp_nested_custom_markers_forward_arrival_resolves_to_parent_marker(self):
        """Nested generic OTLP markers (workflow -> agent.run) must attribute child
        spans to the immediate parent's session via step1_parent, not unattributed (#749).
        """
        trace_id = "trace-otlp-nested-forward"
        pipeline, db = _make_pipeline()

        # 1. Root workflow marker arrives
        root = make_invoke_agent_span(session_id="wf-root", trace_id=trace_id)
        root.span_id = "s-root"
        root.name = "workflow"
        pipeline.process(root)

        # 2. Subagent marker arrives (nested child of root with explicit session)
        sub = make_invoke_agent_span(session_id="sub-agent", trace_id=trace_id)
        sub.span_id = "s-sub"
        sub.parent_span_id = "s-root"
        sub.name = "agent.run"
        pipeline.process(sub)

        # 3. Child span under subagent arrives
        child = make_llm_span(trace_id=trace_id)
        child.span_id = "s-child"
        child.parent_span_id = "s-sub"
        child.session_id = None
        child.conversation_id = None
        pipeline.process(child)

        assert db.spans[-1].session_id == "sub-agent"
        assert db.spans[-1].attribution_step == "step1_parent"
        assert len(db.sessions) == 2

    def test_generic_otlp_nested_custom_markers_reverse_arrival_resolves_to_parent_marker(self):
        """When child span arrives before nested markers (child -> workflow -> agent.run),
        trace reconciliation must resolve child to immediate parent's session via step1_parent (#749).
        """
        trace_id = "trace-otlp-nested-reverse"
        pipeline, db = _make_pipeline()

        # 1. Child span arrives first with parent pointing to subagent
        child = make_llm_span(trace_id=trace_id)
        child.span_id = "s-child"
        child.parent_span_id = "s-sub"
        child.session_id = None
        child.conversation_id = None
        pipeline.process(child)

        # 2. Root workflow arrives
        root = make_invoke_agent_span(session_id="wf-root", trace_id=trace_id)
        root.span_id = "s-root"
        root.name = "workflow"
        pipeline.process(root)

        # 3. Subagent arrives
        sub = make_invoke_agent_span(session_id="sub-agent", trace_id=trace_id)
        sub.span_id = "s-sub"
        sub.parent_span_id = "s-root"
        sub.name = "agent.run"
        pipeline.process(sub)

        stored_child = next(s for s in db.spans if s.span_id == "s-child")
        assert stored_child.session_id == "sub-agent"
        assert stored_child.attribution_step == "step1_parent"

    def test_generic_otlp_multiple_spans_same_custom_name_same_session(self):
        """Multiple spans with the same custom name and same session_id are recognized
        as a sole marker, resolving unparented spans to step2_marker (#749).
        """
        trace_id = "trace-otlp-same-name-same-sess"
        pipeline, db = _make_pipeline()

        m1 = make_invoke_agent_span(session_id="wf-shared", trace_id=trace_id)
        m1.name = "workflow"
        pipeline.process(m1)

        m2 = make_invoke_agent_span(session_id="wf-shared", trace_id=trace_id)
        m2.name = "workflow"
        pipeline.process(m2)

        cost_span = make_llm_span(trace_id=trace_id)
        cost_span.session_id = None
        cost_span.conversation_id = None
        pipeline.process(cost_span)

        assert db.spans[-1].session_id == "wf-shared"
        assert db.spans[-1].attribution_step == "step2_marker"
        assert len(db.sessions) == 1

    def test_child_spans_inheriting_conversation_id_resolve_via_step1_parent(self):
        """Child spans inheriting conversation_id from parent session must resolve via
        step1_parent (Step 1 of ladder), not conversation, preventing redundant full-trace
        reconciliation and preserving parent cache (#749).
        """
        trace_id = "trace-conv-inherit"
        pipeline, db = _make_pipeline()
        conv_id = "conv-inherited-42"

        # 1. Root span with conversation_id
        root = make_invoke_agent_span(conversation_id=conv_id, trace_id=trace_id)
        root.session_id = None
        pipeline.process(root)

        root_span = db.spans[0]
        assert root_span.attribution_step == "conversation"
        assert pipeline._is_session_marker(root_span) is True

        # Cache should contain root span
        cache_key = (trace_id, root.span_id)
        assert cache_key in pipeline._parent_cache

        # 2. Child span inherits conversation_id and has parent_span_id
        child = make_llm_span(conversation_id=conv_id, trace_id=trace_id)
        child.session_id = None
        child.parent_span_id = root.span_id
        pipeline.process(child)

        child_span = db.spans[1]
        assert child_span.session_id == root_span.session_id
        assert child_span.attribution_step == "step1_parent"
        assert pipeline._is_session_marker(child_span) is False

        # Parent cache must NOT be wiped by child ingest
        assert cache_key in pipeline._parent_cache
        child_cache_key = (trace_id, child.span_id)
        assert child_cache_key in pipeline._parent_cache

    def test_five_trace_only_cost_spans_mint_exactly_one_session(self):
        """Preserve #326 behavior: 5 trace-only cost spans without marker
        must share exactly one minted provisional session (must fail at five if broken).
        """
        trace_id = "trace-five-spans-no-marker"
        pipeline, db = _make_pipeline()

        for _ in range(5):
            span = make_llm_span(trace_id=trace_id)
            span.session_id = None
            span.conversation_id = None
            pipeline.process(span)

        session_ids = {s.session_id for s in db.spans}
        assert len(session_ids) == 1
        assert None not in session_ids
        assert len(db.sessions) == 1

    def test_attribution_step_stored_on_field_not_in_attributes(self):
        """Attribution step is recorded on span.attribution_step, NOT in span.attributes."""
        trace_id = "trace-attr-provenance"
        pipeline, db = _make_pipeline()

        span = make_llm_span(trace_id=trace_id)
        span.session_id = None
        span.conversation_id = None
        pipeline.process(span)

        stored = db.spans[0]
        assert stored.attribution_step == "provisional"
        assert "tokenjam.session_attribution" not in stored.attributes

    def test_parent_chain_resolution_uses_get_span_not_get_trace_spans(self):
        """Issue #749: Parent resolution must use get_span / cache, never get_trace_spans."""
        trace_id = "trace-bounded-resolution"
        pipeline, db = _make_pipeline()

        parent = make_llm_span(trace_id=trace_id)
        parent.session_id = None
        parent.conversation_id = None
        pipeline.process(parent)

        # Monkeypatch get_trace_spans to fail if called
        def boom(_trace_id):
            raise AssertionError("get_trace_spans must not be called during parent chain resolution")

        db.get_trace_spans = boom

        child = make_llm_span(trace_id=trace_id)
        child.session_id = None
        child.conversation_id = None
        child.parent_span_id = parent.span_id
        # Should succeed using _parent_cache / get_span without calling get_trace_spans
        pipeline.process(child)

        assert db.spans[1].session_id == db.spans[0].session_id

    def test_scale_deep_parent_chain_ingest_is_linear(self):
        """Scale benchmark: ingesting 500 chained spans must complete quickly without O(n^2) blowup."""
        trace_id = "trace-deep-chain-scale"
        pipeline, db = _make_pipeline()

        root = make_invoke_agent_span(session_id="root-sess", trace_id=trace_id)
        pipeline.process(root)

        last_span_id = root.span_id
        for i in range(500):
            span = make_llm_span(trace_id=trace_id)
            span.parent_span_id = last_span_id
            span.session_id = None
            span.conversation_id = None
            pipeline.process(span)
            last_span_id = span.span_id

        assert len(db.spans) == 501
        assert all(s.session_id == "root-sess" for s in db.spans)
        assert all(s.attribution_step == "step1_parent" for s in db.spans[1:])


# ===========================================================================
# Capture content stripping tests
# ===========================================================================

class TestCaptureStripping:

    def test_prompt_content_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.PROMPT_CONTENT: "secret prompt",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert GenAIAttributes.PROMPT_CONTENT not in stored_span.attributes

    def test_prompt_content_kept_when_capture_on(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=True))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.PROMPT_CONTENT: "kept prompt",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert stored_span.attributes[GenAIAttributes.PROMPT_CONTENT] == "kept prompt"

    def test_system_prefix_content_stripped_when_prompts_capture_off(self):
        # The recovered system prefix is a project's CLAUDE.md text, stamped
        # onto backfilled spans. It is prompt content and must ride the same
        # `prompts` toggle -- the gate's contract covers ALL content keyed to
        # a [capture] toggle, not just the keys that existed when it was
        # written.
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=False))

        span = make_llm_span(extra_attributes={
            TjAttributes.SYSTEM_PREFIX_CONTENT: "# CLAUDE.md\nsecret project rules",
        })
        pipeline.process(span)

        assert TjAttributes.SYSTEM_PREFIX_CONTENT not in db.spans[-1].attributes

    def test_system_prefix_content_kept_when_prompts_capture_on(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=True))

        span = make_llm_span(extra_attributes={
            TjAttributes.SYSTEM_PREFIX_CONTENT: "# CLAUDE.md\nkept rules",
        })
        pipeline.process(span)

        assert db.spans[-1].attributes[TjAttributes.SYSTEM_PREFIX_CONTENT] == (
            "# CLAUDE.md\nkept rules"
        )

    def test_tool_output_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(tool_outputs=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.TOOL_OUTPUT: "secret output",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert GenAIAttributes.TOOL_OUTPUT not in stored_span.attributes

    def test_completion_content_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(completions=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.COMPLETION_CONTENT: "secret completion",
        })
        pipeline.process(span)

        assert GenAIAttributes.COMPLETION_CONTENT not in db.spans[-1].attributes

    def test_tool_input_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(tool_inputs=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.TOOL_INPUT: "secret input",
        })
        pipeline.process(span)

        assert GenAIAttributes.TOOL_INPUT not in db.spans[-1].attributes


# ===========================================================================
# Session totals tests
# ===========================================================================

class TestSessionTotals:

    def test_session_totals_updated_after_multiple_spans(self):
        pipeline, db = _make_pipeline()

        conv_id = "conv-totals"
        for _ in range(3):
            span = make_llm_span(
                input_tokens=100,
                output_tokens=50,
                conversation_id=conv_id,
            )
            span.session_id = None  # Force session resolution
            pipeline.process(span)

        # All spans should share the same session
        session_ids = {s.session_id for s in db.spans}
        assert len(session_ids) == 1

        session = list(db.sessions.values())[0]
        assert session.input_tokens == 300
        assert session.output_tokens == 150

    def test_error_span_increments_error_count(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(status="error", conversation_id="conv-err")
        pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.error_count == 1

    def test_tool_span_increments_tool_call_count(self):
        pipeline, db = _make_pipeline()

        span = make_tool_span(tool_name="my_tool", conversation_id="conv-tool")
        span.session_id = None
        pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.tool_call_count == 1

    def test_cost_accumulated_in_session(self):
        pipeline, db = _make_pipeline()

        conv_id = "conv-cost"
        for _ in range(2):
            span = make_llm_span(cost_usd=0.05, conversation_id=conv_id)
            span.session_id = None
            pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.total_cost_usd == pytest.approx(0.10)


# ===========================================================================
# Error handling tests
# ===========================================================================

class TestErrorHandling:

    def test_span_rejected_error_not_written_to_db(self):
        security = SecurityConfig(max_attributes_per_span=2)
        pipeline, db = _make_pipeline(security=security)

        span = make_llm_span(extra_attributes={
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
        })
        with pytest.raises(SpanRejectedError):
            pipeline.process(span)

        assert len(db.spans) == 0

    def test_hook_failure_does_not_crash_pipeline(self):
        """Post-ingest hook errors are logged, not propagated."""
        db = _StubBackend()
        config = TjConfig(version="1")

        class FailingCostEngine:
            def process_span(self, span: NormalizedSpan) -> None:
                raise RuntimeError("cost engine broke")

        pipeline = IngestPipeline(
            db=db,
            config=config,
            cost_engine=FailingCostEngine(),
            alert_engine=NoopAlertEngine(),
            schema_validator=NoopSchemaValidator(),
        )

        span = make_llm_span()
        # Should NOT raise even though cost engine fails
        pipeline.process(span)
        assert len(db.spans) == 1


# ===========================================================================
# Session lifecycle tests
#
# Regression coverage for the Claude Code / Codex logs path, where each
# user_prompt event is mapped to a zero-duration invoke_agent span. Treating
# those turn-start markers as session completions force-completed every live
# session on its first prompt — the dashboard showed active work as
# "completed" with 0 duration, and the drift/alert session-end hooks fired on
# every turn.
# ===========================================================================

class TestSessionLifecycle:

    def test_zero_duration_invoke_agent_marker_keeps_session_active(self):
        # Claude Code maps each user_prompt to a zero-duration invoke_agent
        # span (end_time == start_time). It marks the START of a turn.
        pipeline, db = _make_pipeline()
        marker = make_invoke_agent_span(session_id="s1", duration_ms=0.0)

        pipeline.process(marker)

        session = db.get_session("s1")
        assert session is not None
        assert session.status == "active"

    def test_streaming_activity_keeps_session_active(self):
        # A marker followed by real LLM activity is still an ongoing session.
        pipeline, db = _make_pipeline()
        pipeline.process(make_invoke_agent_span(session_id="s1", duration_ms=0.0))
        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").status == "active"

    def test_real_invoke_agent_span_completes_session(self):
        # The SDK @watch() path emits one invoke_agent span that brackets the
        # whole run (end_time strictly after start_time). That DOES complete it.
        pipeline, db = _make_pipeline()
        end_span = make_invoke_agent_span(session_id="s1", duration_ms=5000.0)

        pipeline.process(end_span)

        assert db.get_session("s1").status == "completed"

    def test_activity_reactivates_mistakenly_completed_session(self):
        # An in-flight session left "completed" (e.g. by the old bug, or a
        # prior restart) must self-heal when new activity arrives.
        pipeline, db = _make_pipeline()
        db.upsert_session(make_session(session_id="s1", status="completed"))

        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").status == "active"


class TestServiceNamespace:
    """service.namespace (project grouping) capture on the session."""

    def test_session_captures_service_namespace(self):
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1", service_namespace="aquanode"))

        assert db.get_session("s1").service_namespace == "aquanode"

    def test_namespace_late_resolves_from_later_span(self):
        # A tool span with no namespace creates the session; a later LLM span
        # that carries the namespace backfills it.
        pipeline, db = _make_pipeline()
        pipeline.process(make_invoke_agent_span(session_id="s1", service_namespace=None))
        assert db.get_session("s1").service_namespace is None

        pipeline.process(make_llm_span(session_id="s1", service_namespace="aquanode"))
        assert db.get_session("s1").service_namespace == "aquanode"

    def test_namespace_absent_stays_none(self):
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").service_namespace is None

    def test_namespace_falls_back_to_configured_project(self):
        # An already-running agent never sends service.namespace; the agent's
        # configured project supplies it server-side (no restart needed).
        pipeline, db = _make_pipeline(
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline.process(make_llm_span(agent_id="claude-code-harness", session_id="s1"))

        assert db.get_session("s1").service_namespace == "aquanode"

    def test_wire_namespace_wins_over_configured_project(self):
        pipeline, db = _make_pipeline(
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="s1",
            service_namespace="explicit-ns"))

        assert db.get_session("s1").service_namespace == "explicit-ns"

    def test_session_captures_service_instance_id(self):
        # The per-terminal instance id (e.g. "dev-box") is persisted on the
        # session for use as its display label.
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1", service_instance_id="dev-box"))

        assert db.get_session("s1").service_instance_id == "dev-box"
