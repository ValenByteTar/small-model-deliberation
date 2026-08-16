"""
Tests para LLMSupport (RES-004, ADR-0031) y BitNetModelProvider.

Verifica:
- Hypothesis contract (suggestion, confidence, reasoning, stage, run_id)
- LLMSupport passive mode: observa, genera hipotesis, no bloquea
- FanOutTraceSink: fan-out con aislamiento de errores
- BitNetModelProvider: contrato ModelProvider (mockeado, sin servidor real)
"""

import json
import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.kernel.llm_support import LLMSupport
from hybrid_rag.kernel.observability import (
    FanOutTraceSink,
    InMemoryTraceSink,
    NullTraceSink,
)
from hybrid_rag.kernel.state import ExecutionState, Hypothesis, TraceEvent


# --- Fixtures ---


class MockModelProvider:
    """ModelProvider mock que retorna una hipotesis JSON predeterminada."""

    name = "mock"
    model = "mock-model"

    def __init__(self, response: str = None, fail: bool = False) -> None:
        self._response = response or "GOOD (score above 0.40)"
        self._fail = fail
        self.calls = []

    def generate(self, prompt, *, options=None, timeout=None) -> str:
        self.calls.append(prompt)
        if self._fail:
            raise RuntimeError("mock failure")
        return self._response


@pytest.fixture
def mock_provider():
    return MockModelProvider()


@pytest.fixture
def llm_support(mock_provider):
    support = LLMSupport(
        model_provider=mock_provider,
        mode="passive",
        max_hypotheses=10,
    )
    support.start()
    yield support
    support.stop()


@pytest.fixture
def sample_state():
    return ExecutionState(question="What is ISO 27001?")


# --- Hypothesis contract ---


class TestHypothesisContract:
    def test_hypothesis_fields(self):
        h = Hypothesis(
            suggestion="RETRY_RETRIEVAL",
            confidence=0.71,
            reasoning="BM25 and embeddings disagree",
            stage="post_reranker",
            run_id="abc123",
            model="bitnet-b1.58-2b-4t",
        )
        d = h.to_dict()
        assert d["suggestion"] == "RETRY_RETRIEVAL"
        assert d["confidence"] == 0.71
        assert d["reasoning"] == "BM25 and embeddings disagree"
        assert d["stage"] == "post_reranker"
        assert d["run_id"] == "abc123"
        assert d["model"] == "bitnet-b1.58-2b-4t"
        assert "ts" in d

    def test_hypothesis_defaults(self):
        h = Hypothesis()
        assert h.suggestion == ""
        assert h.confidence == 0.0
        assert h.reasoning == ""
        assert h.stage == ""
        assert h.run_id == ""


# --- LLMSupport passive mode ---


class TestLLMSupportPassive:
    def test_off_mode_does_nothing(self, mock_provider, sample_state):
        support = LLMSupport(model_provider=mock_provider, mode="off")
        support.start()
        event = TraceEvent(kind="controller.start", message="test")
        support.emit(event, sample_state)
        support.stop()
        assert len(support.get_hypotheses()) == 0
        assert support.stats["events_observed"] == 0

    def test_passive_observes_and_generates(self, llm_support, sample_state):
        event = TraceEvent(kind="capability.end", message="retrieve", data={"duration_ms": 50.0})
        llm_support.emit(event, sample_state)
        # Wait for worker to process
        for _ in range(50):
            if llm_support.stats["hypotheses_generated"] > 0:
                break
            time.sleep(0.05)
        hyps = llm_support.get_hypotheses()
        assert len(hyps) == 1
        assert hyps[0].suggestion == "GOOD_EVIDENCE"
        assert hyps[0].confidence == 0.7
        assert hyps[0].stage == "capability.end"
        assert hyps[0].run_id == sample_state.run_id

    def test_non_trigger_event_does_not_generate(self, llm_support, sample_state):
        event = TraceEvent(kind="policy.decision", message="invoke")
        llm_support.emit(event, sample_state)
        time.sleep(0.2)
        assert llm_support.stats["events_observed"] == 1
        assert llm_support.stats["events_triggered"] == 0
        assert len(llm_support.get_hypotheses()) == 0

    def test_emit_does_not_block(self, llm_support, sample_state):
        """emit() debe retornar inmediatamente sin esperar al modelo."""
        event = TraceEvent(kind="capability.end", message="generate")
        t0 = time.time()
        llm_support.emit(event, sample_state)
        dt = time.time() - t0
        assert dt < 0.01  # debe ser casi instantaneo

    def test_failed_generation_counted(self):
        provider = MockModelProvider(fail=True)
        support = LLMSupport(model_provider=provider, mode="passive")
        support.start()
        event = TraceEvent(kind="capability.end", message="retrieve")
        support.emit(event, ExecutionState(question="test"))
        for _ in range(50):
            if support.stats["hypotheses_failed"] > 0:
                break
            time.sleep(0.05)
        support.stop()
        assert support.stats["hypotheses_failed"] == 1
        assert support.stats["hypotheses_generated"] == 0

    def test_hypothesis_limit(self, mock_provider):
        support = LLMSupport(model_provider=mock_provider, mode="passive", max_hypotheses=3)
        support.start()
        state = ExecutionState(question="test")
        for i in range(10):
            event = TraceEvent(kind="capability.end", message=f"cap_{i}")
            support.emit(event, state)
        # Wait for processing
        time.sleep(2.0)
        support.stop()
        assert len(support.get_hypotheses()) <= 3

    def test_clear(self, llm_support, sample_state):
        event = TraceEvent(kind="capability.end", message="retrieve")
        llm_support.emit(event, sample_state)
        time.sleep(0.5)
        llm_support.clear()
        assert len(llm_support.get_hypotheses()) == 0
        assert llm_support.stats["events_observed"] == 0


# --- FanOutTraceSink ---


class TestFanOutTraceSink:
    def test_fanout_to_multiple_sinks(self):
        primary = InMemoryTraceSink()
        secondary = InMemoryTraceSink()
        fan = FanOutTraceSink(primary, secondary)
        event = TraceEvent(kind="test", message="hello")
        fan.emit(event)
        assert len(primary.events) == 1
        assert len(secondary.events) == 1

    def test_secondary_error_does_not_affect_primary(self):
        primary = InMemoryTraceSink()

        class BrokenSink:
            def emit(self, event, state=None):
                raise RuntimeError("broken")

        fan = FanOutTraceSink(primary, BrokenSink())
        event = TraceEvent(kind="test", message="hello")
        fan.emit(event)  # should not raise
        assert len(primary.events) == 1

    def test_fanout_with_llm_support(self, mock_provider):
        primary = InMemoryTraceSink()
        support = LLMSupport(model_provider=mock_provider, mode="passive")
        support.start()
        fan = FanOutTraceSink(primary, support)
        event = TraceEvent(kind="capability.end", message="retrieve")
        state = ExecutionState(question="test")
        fan.emit(event, state)
        # Primary gets event immediately
        assert len(primary.events) == 1
        # LLMSupport processes in background
        for _ in range(50):
            if support.stats["hypotheses_generated"] > 0:
                break
            time.sleep(0.05)
        support.stop()
        assert support.stats["hypotheses_generated"] == 1


# --- BitNetModelProvider (mocked, no real server) ---


class TestBitNetModelProvider:
    def test_contract_compliance(self):
        """BitNetModelProvider implementa el contrato ModelProvider (ADR-0007)."""
        from hybrid_rag.providers.bitnet_provider import BitNetModelProvider

        provider = BitNetModelProvider(
            model_path="models/test.gguf",
            auto_start=False,
        )
        assert provider.name == "bitnet"
        assert hasattr(provider, "model")
        assert hasattr(provider, "generate")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "is_available")

    @patch("hybrid_rag.providers.bitnet_provider.requests")
    def test_generate_calls_server(self, mock_requests):
        from hybrid_rag.providers.bitnet_provider import BitNetModelProvider

        provider = BitNetModelProvider(
            model_path="models/test.gguf",
            auto_start=False,
        )
        # Mock server as available
        mock_get = MagicMock()
        mock_get.status_code = 200
        mock_requests.get.return_value = mock_get

        mock_post = MagicMock()
        mock_post.status_code = 200
        mock_post.json.return_value = {"content": "test response"}
        mock_requests.post.return_value = mock_post

        result = provider.generate("test prompt")
        assert result == "test response"
        mock_requests.post.assert_called_once()

    def test_generate_returns_empty_when_server_down(self):
        from hybrid_rag.providers.bitnet_provider import BitNetModelProvider

        provider = BitNetModelProvider(
            model_path="models/nonexistent.gguf",
            auto_start=False,
        )
        result = provider.generate("test prompt")
        assert result == ""
