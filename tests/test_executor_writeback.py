"""
Tests for execution writeback — Task 5.

Verifies that after execute_with_fallback:
  1. LearningStore has a record for the model that was called.
  2. BudgetTracker has non-zero spend recorded.

Spine is booted with real (tmp-path) stores and torn down between tests.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from spine import Core

from ControlCore.adapters.executor import AdapterRegistry, ExecutionEngine
from ControlCore.adapters.interface import (
    AdapterProvenance,
    AdapterResult,
    AdapterStatus,
    AdapterTiming,
)
from ControlCore.circuit_breaker import CircuitBreakerRegistry
from ControlCore.registry.budget import BudgetConfig, BudgetTracker
from ControlCore.registry.learning import LearningStore
from ControlCore.registry.preferences import Preferences
from ControlCore.registry.schema import ModelRegistry, Provider
from ControlCore.schemas import CallStatus

from tests.conftest import MockAdapter, make_call, make_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_registry(*aliases: str) -> ModelRegistry:
    models = [make_model(alias=alias, provider=Provider.local) for alias in aliases]
    return ModelRegistry(models=models)


def make_adapter_registry(*adapters: MockAdapter) -> AdapterRegistry:
    reg = AdapterRegistry()
    for adapter in adapters:
        reg.register(adapter)
    return reg


def make_engine(
    model_aliases: list[str],
    adapter_registry: AdapterRegistry,
) -> ExecutionEngine:
    model_registry = make_registry(*model_aliases)
    return ExecutionEngine(
        model_registry=model_registry,
        adapter_registry=adapter_registry,
        circuit_registry=CircuitBreakerRegistry(),
    )


def _success_result_with_tokens(
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> AdapterResult:
    """AdapterResult with provenance that includes token counts for cost estimation."""
    from datetime import datetime, timedelta

    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=250)
    timing = AdapterTiming.create(start, end)

    prov = AdapterProvenance(
        adapter_name="mock",
        adapter_version="1.0.0",
        model_alias="qwen:32b",
        timing=timing,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return AdapterResult(
        status=AdapterStatus.success,
        content="mock response",
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_spine():
    """Ensure spine singleton is cleared before and after every test."""
    Core._reset_instance()
    yield
    Core._reset_instance()


@pytest.fixture
def learning_store(tmp_path):
    db = tmp_path / "test_learning.db"
    store = LearningStore(db_path=str(db))
    yield store
    store.close()


@pytest.fixture
def budget_tracker():
    return BudgetTracker(BudgetConfig(daily_limit=10.0, hourly_limit=2.0))


@pytest.fixture
def booted_spine(learning_store, budget_tracker):
    """Boot spine with real learning + budget + preferences registered."""

    def setup(c: Core) -> None:
        c.register("learning", learning_store)
        c.register("budget", budget_tracker)
        c.register("preferences", Preferences())
        c.boot(env="test")

    core = Core.boot_once(setup)
    return core


# ---------------------------------------------------------------------------
# Test 1: LearningStore receives a record after execute_with_fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writeback_records_to_learning_store(booted_spine, learning_store):
    """After execute_with_fallback succeeds, learning store has a record for the model."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=_success_result_with_tokens(input_tokens=200, output_tokens=100),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call)

    assert cc_result.status == CallStatus.complete

    stats = learning_store.stats("qwen:32b")
    assert stats.call_count == 1, f"Expected 1 call record, got {stats.call_count}"
    assert stats.success_rate == 1.0


# ---------------------------------------------------------------------------
# Test 2: BudgetTracker receives spend after execute_with_fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writeback_records_to_budget_tracker(booted_spine, budget_tracker):
    """After execute_with_fallback, budget tracker has recorded non-zero spend."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=_success_result_with_tokens(input_tokens=1000, output_tokens=500),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call)

    assert cc_result.status == CallStatus.complete

    spent = budget_tracker.spent_today()
    # cost = (1000 * 0.01 + 500 * 0.03) / 1000 = (10 + 15) / 1000 = 0.025
    assert spent > 0.0, f"Expected spend > 0, got {spent}"
    assert abs(spent - 0.025) < 1e-9, f"Expected cost ~0.025, got {spent}"


# ---------------------------------------------------------------------------
# Test 3: Writeback on failure is still recorded (failed outcome)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writeback_records_failed_outcome(booted_spine, learning_store):
    """execute_with_fallback writeback records even when the model errors."""
    from ControlCore.registry.fallback import (
        FallbackPolicy,
        ModelSwitchConfig,
        QueueEscalationConfig,
    )

    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=AdapterResult(
            status=AdapterStatus.error,
            error_message="model down",
        ),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    policy = FallbackPolicy(
        max_total_attempts=2,
        max_same_model_retries=1,
        queue_escalation=QueueEscalationConfig(enabled=False),
    )

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call, policy=policy)

    assert cc_result.status == CallStatus.failed

    stats = learning_store.stats("qwen:32b")
    assert stats.call_count >= 1, "Expected at least 1 record for the failed attempt"
    assert stats.success_rate == 0.0


# ---------------------------------------------------------------------------
# Test 4: Writeback is silent when spine is not booted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writeback_silent_without_spine():
    """_writeback does not raise when spine has not been booted."""
    # Spine is not booted (reset_spine fixture cleared it).
    from ControlCore.adapters.executor import _writeback

    result = AdapterResult(status=AdapterStatus.success, content="hi")
    # Should complete without raising
    _writeback("some-model", result, "lookup")
