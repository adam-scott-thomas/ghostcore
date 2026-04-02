"""
Tests for ExecutionEngine (executor.py).

Covers:
- execute_once: success path
- execute_once: no adapter available
- execute_with_fallback: single-model success
- execute_with_fallback: first model fails, second succeeds (fallback)
- execute_with_fallback: all models fail → ALL_EXHAUSTED
"""

from __future__ import annotations

import pytest

from ControlCore.adapters.executor import (
    AdapterRegistry,
    ExecutionEngine,
    ExecutionOutcome,
)
from ControlCore.adapters.interface import AdapterResult, AdapterStatus
from ControlCore.circuit_breaker import CircuitBreakerRegistry, CircuitConfig
from ControlCore.registry.fallback import (
    FallbackPolicy,
    ModelSwitchConfig,
    ModelSwitchCondition,
    QueueEscalationConfig,
)
from ControlCore.registry.schema import ModelRegistry, Provider
from ControlCore.schemas import CallStatus

from tests.conftest import MockAdapter, make_call, make_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_registry(*aliases: str) -> ModelRegistry:
    """Build a ModelRegistry containing one ModelEntry per alias."""
    models = [make_model(alias=alias, provider=Provider.local) for alias in aliases]
    return ModelRegistry(models=models)


def make_adapter_registry(*adapters: MockAdapter) -> AdapterRegistry:
    """Build an AdapterRegistry with the given adapters registered."""
    reg = AdapterRegistry()
    for adapter in adapters:
        reg.register(adapter)
    return reg


def make_engine(
    model_aliases: list[str],
    adapter_registry: AdapterRegistry,
    circuit_registry: CircuitBreakerRegistry | None = None,
) -> ExecutionEngine:
    """Convenience constructor for ExecutionEngine."""
    model_registry = make_registry(*model_aliases)
    return ExecutionEngine(
        model_registry=model_registry,
        adapter_registry=adapter_registry,
        circuit_registry=circuit_registry or CircuitBreakerRegistry(),
    )


# ---------------------------------------------------------------------------
# execute_once tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_once_success():
    """execute_once returns AdapterStatus.success when adapter handles the model."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=AdapterResult(status=AdapterStatus.success, content="hello"),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    result, used_adapter = await engine.execute_once(call, "qwen:32b")

    assert result.status == AdapterStatus.success
    assert result.content == "hello"
    assert used_adapter is mock
    assert len(mock.calls) == 1
    assert mock.calls[0]["model_alias"] == "qwen:32b"


@pytest.mark.asyncio
async def test_execute_once_no_adapter():
    """execute_once returns an error result when no adapter can handle the model."""
    # Register an adapter that handles a *different* model
    mock = MockAdapter(handled_models={"other:model"})
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    result, used_adapter = await engine.execute_once(call, "qwen:32b")

    assert result.status == AdapterStatus.error
    assert result.error_code == "NO_ADAPTER"
    assert used_adapter is None
    # MockAdapter should not have been called
    assert len(mock.calls) == 0


@pytest.mark.asyncio
async def test_execute_once_circuit_open():
    """execute_once returns CIRCUIT_OPEN error when circuit breaker is open."""
    mock = MockAdapter(handled_models={"qwen:32b"})
    adapter_reg = make_adapter_registry(mock)

    # Create a circuit registry — failure_threshold=1 means a single failure opens it
    circuit_reg = CircuitBreakerRegistry(
        default_config=CircuitConfig(
            failure_threshold=1,
        )
    )
    engine = make_engine(["qwen:32b"], adapter_reg, circuit_registry=circuit_reg)

    # Manually trip the circuit open
    circuit = circuit_reg.get_circuit("mock", "qwen:32b")
    circuit.record_failure()

    call = make_call(target_alias="qwen:32b")
    result, used_adapter = await engine.execute_once(call, "qwen:32b")

    assert result.status == AdapterStatus.error
    assert result.error_code == "CIRCUIT_OPEN"
    assert len(mock.calls) == 0  # adapter never invoked


# ---------------------------------------------------------------------------
# execute_with_fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_with_fallback_success():
    """execute_with_fallback returns SUCCESS when the first model succeeds."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=AdapterResult(status=AdapterStatus.success, content="answer"),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call)

    assert trace.outcome == ExecutionOutcome.SUCCESS
    assert cc_result.status == CallStatus.complete
    assert cc_result.answer == "answer"
    assert trace.total_attempts == 1
    assert trace.attempts[0].model_alias == "qwen:32b"
    assert trace.attempts[0].succeeded is True


@pytest.mark.asyncio
async def test_execute_with_fallback_first_fails_second_succeeds():
    """execute_with_fallback falls back to the second model when the first fails."""
    failing = MockAdapter(
        handled_models={"model-a"},
        result=AdapterResult(
            status=AdapterStatus.error,
            error_message="model-a down",
        ),
        adapter_name="failing",
    )
    succeeding = MockAdapter(
        handled_models={"model-b"},
        result=AdapterResult(status=AdapterStatus.success, content="fallback answer"),
        adapter_name="succeeding",
    )

    adapter_reg = AdapterRegistry()
    adapter_reg.register(failing)
    adapter_reg.register(succeeding)

    # Registry with both models; routing will try them in order
    model_registry = ModelRegistry(models=[
        make_model(alias="model-a", provider=Provider.local),
        make_model(alias="model-b", provider=Provider.local),
    ])
    engine = ExecutionEngine(
        model_registry=model_registry,
        adapter_registry=adapter_reg,
        circuit_registry=CircuitBreakerRegistry(),
    )

    # Policy: allow model switching on error, up to 2 models
    policy = FallbackPolicy(
        max_total_attempts=4,
        max_same_model_retries=0,
        model_switch=ModelSwitchConfig(
            enabled=True,
            max_models_to_try=2,
            conditions=[ModelSwitchCondition.error],
        ),
    )

    # Call targets model-a specifically; engine uses routing over registry
    call = make_call(target_alias="model-a")
    cc_result, trace = await engine.execute_with_fallback(call, policy=policy)

    assert trace.outcome == ExecutionOutcome.SUCCESS
    assert cc_result.status == CallStatus.complete
    assert cc_result.answer == "fallback answer"
    # At least 2 attempts recorded (model-a failed, model-b succeeded)
    assert trace.total_attempts >= 2

    statuses = [a.result.status for a in trace.attempts]
    assert AdapterStatus.error in statuses
    assert AdapterStatus.success in statuses

    # The last attempt must be the successful one
    assert trace.attempts[-1].succeeded is True
    assert trace.attempts[-1].model_alias == "model-b"


@pytest.mark.asyncio
async def test_execute_with_fallback_all_models_fail():
    """execute_with_fallback returns ALL_EXHAUSTED when every model fails."""
    failing_a = MockAdapter(
        handled_models={"model-a"},
        result=AdapterResult(status=AdapterStatus.error, error_message="a down"),
        adapter_name="adapter-a",
    )
    failing_b = MockAdapter(
        handled_models={"model-b"},
        result=AdapterResult(status=AdapterStatus.error, error_message="b down"),
        adapter_name="adapter-b",
    )

    adapter_reg = AdapterRegistry()
    adapter_reg.register(failing_a)
    adapter_reg.register(failing_b)

    model_registry = ModelRegistry(models=[
        make_model(alias="model-a", provider=Provider.local),
        make_model(alias="model-b", provider=Provider.local),
    ])
    engine = ExecutionEngine(
        model_registry=model_registry,
        adapter_registry=adapter_reg,
        circuit_registry=CircuitBreakerRegistry(),
    )

    policy = FallbackPolicy(
        max_total_attempts=4,
        max_same_model_retries=0,
        model_switch=ModelSwitchConfig(
            enabled=True,
            max_models_to_try=2,
            conditions=[ModelSwitchCondition.error],
        ),
        # Disable queue escalation so we get ALL_EXHAUSTED, not QUEUED
        queue_escalation=QueueEscalationConfig(enabled=False),
    )

    call = make_call(target_alias="model-a")
    cc_result, trace = await engine.execute_with_fallback(call, policy=policy)

    assert trace.outcome == ExecutionOutcome.ALL_EXHAUSTED
    assert cc_result.status == CallStatus.failed
    assert len(cc_result.errors) > 0
    # All attempts should have failed
    for attempt in trace.attempts:
        assert attempt.succeeded is False


@pytest.mark.asyncio
async def test_execute_with_fallback_no_eligible_models():
    """execute_with_fallback returns failed when no models are in the registry."""
    adapter_reg = AdapterRegistry()  # No adapters needed — won't reach that point
    engine = make_engine([], adapter_reg)  # Empty registry

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call)

    assert cc_result.status == CallStatus.failed
    assert len(cc_result.errors) > 0


@pytest.mark.asyncio
async def test_execute_with_fallback_result_contains_provenance():
    """Successful execute_with_fallback result includes non-empty provenance."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=AdapterResult(status=AdapterStatus.success, content="test"),
    )
    adapter_reg = make_adapter_registry(mock)
    engine = make_engine(["qwen:32b"], adapter_reg)

    call = make_call(target_alias="qwen:32b")
    cc_result, trace = await engine.execute_with_fallback(call)

    assert cc_result.provenance is not None
    assert cc_result.call_id == call.call_id


@pytest.mark.asyncio
async def test_execute_once_skip_circuit_breaker():
    """execute_once with skip_circuit_breaker=True bypasses an open circuit."""
    mock = MockAdapter(
        handled_models={"qwen:32b"},
        result=AdapterResult(status=AdapterStatus.success, content="bypassed"),
    )
    adapter_reg = make_adapter_registry(mock)

    # Trip the circuit open
    circuit_reg = CircuitBreakerRegistry(
        default_config=CircuitConfig(failure_threshold=1)
    )
    engine = make_engine(["qwen:32b"], adapter_reg, circuit_registry=circuit_reg)
    circuit_reg.get_circuit("mock", "qwen:32b").record_failure()

    call = make_call(target_alias="qwen:32b")
    result, used_adapter = await engine.execute_once(
        call, "qwen:32b", skip_circuit_breaker=True
    )

    assert result.status == AdapterStatus.success
    assert len(mock.calls) == 1
