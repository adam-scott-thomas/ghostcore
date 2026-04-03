"""
Tests for intelligent routing factors: observed latency, budget pressure,
task affinity, and load balance jitter.

These factors read from spine-backed stores (LearningStore, BudgetTracker,
Preferences) and fall back to neutral scores when spine is not booted.
"""

from __future__ import annotations

import pytest
from spine import Core

from ControlCore.registry.routing import (
    RoutingWeights,
    compute_routing_order,
)
from ControlCore.registry.schema import CostHints
from ControlCore.registry.learning import LearningStore, ModelStats
from ControlCore.registry.budget import BudgetConfig, BudgetTracker
from ControlCore.registry.preferences import AffinityRule, Preferences

from tests.conftest import make_call, make_model


# ---------------------------------------------------------------------------
# Spine reset fixture — runs for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_spine():
    Core._reset_instance()
    yield
    Core._reset_instance()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boot_spine(**capabilities):
    """Boot spine with the given capabilities registered."""
    def setup(c: Core):
        for name, value in capabilities.items():
            c.register(name, value)
        c.boot(env="test")
    return Core.boot_once(setup)


def _weights_only_factor(factor: str, weight: float = 100.0) -> RoutingWeights:
    """Create RoutingWeights with all factors zeroed except the specified one."""
    all_zero = dict(
        trust_tier=0.0,
        capability_match=0.0,
        latency_hint=0.0,
        cost_hint=0.0,
        context_headroom=0.0,
        refusal_rate=0.0,
        observed_latency=0.0,
        budget_pressure=0.0,
        task_affinity=0.0,
        load_balance_jitter=0.0,
    )
    all_zero[factor] = weight
    return RoutingWeights(**all_zero)


class FakeLearningStore:
    """In-memory stub that returns pre-configured stats per alias."""

    def __init__(self, data: dict[str, ModelStats]):
        self._data = data

    def stats(self, model_alias: str, **kwargs) -> ModelStats:
        return self._data.get(model_alias, ModelStats())


# ---------------------------------------------------------------------------
# 1. Observed latency prefers faster model
# ---------------------------------------------------------------------------

def test_observed_latency_prefers_faster_model():
    """Model with lower observed latency should rank higher."""
    fast_stats = ModelStats(call_count=20, avg_latency_ms=500.0)
    slow_stats = ModelStats(call_count=20, avg_latency_ms=8000.0)
    store = FakeLearningStore({
        "fast:model": fast_stats,
        "slow:model": slow_stats,
    })
    _boot_spine(learning=store)

    fast = make_model(alias="fast:model")
    slow = make_model(alias="slow:model")
    call = make_call()
    weights = _weights_only_factor("observed_latency", 100.0)

    result = compute_routing_order(call, [slow, fast], weights=weights)
    assert result.ordered_aliases[0] == "fast:model"


def test_observed_latency_insufficient_data_is_neutral():
    """Models with < 5 calls get neutral score, not penalised."""
    few_stats = ModelStats(call_count=3, avg_latency_ms=9000.0)
    many_stats = ModelStats(call_count=20, avg_latency_ms=5000.0)
    store = FakeLearningStore({
        "few:model": few_stats,
        "many:model": many_stats,
    })
    _boot_spine(learning=store)

    few = make_model(alias="few:model")
    many = make_model(alias="many:model")
    call = make_call()
    weights = _weights_only_factor("observed_latency", 100.0)

    result = compute_routing_order(call, [few, many], weights=weights)
    # few:model has neutral (weight*0.5 = 50), many:model has ~55 for 5000ms
    # Both are close but the point is few:model doesn't get 0 for its 9000ms
    few_ranked = [r for r in result.ordered if r.alias == "few:model"][0]
    assert few_ranked.score == pytest.approx(50.0), "Insufficient data should give neutral score"


# ---------------------------------------------------------------------------
# 2. Budget pressure penalises expensive when budget is low
# ---------------------------------------------------------------------------

def test_budget_pressure_penalizes_expensive_model():
    """When budget > 80% consumed, expensive models should score lower."""
    budget = BudgetTracker(BudgetConfig(daily_limit=10.0))
    # Spend 95% of budget
    budget.record_spend(9.5)
    _boot_spine(budget=budget)

    cheap = make_model(
        alias="cheap:model",
        cost_hints=CostHints(input_per_1k_tokens=0.0001),
    )
    expensive = make_model(
        alias="expensive:model",
        cost_hints=CostHints(input_per_1k_tokens=0.03),
    )

    call = make_call()
    weights = _weights_only_factor("budget_pressure", 100.0)

    result = compute_routing_order(call, [expensive, cheap], weights=weights)
    assert result.ordered_aliases[0] == "cheap:model"


def test_budget_pressure_neutral_when_budget_ok():
    """When budget < 80% consumed, both models get neutral scores."""
    budget = BudgetTracker(BudgetConfig(daily_limit=10.0))
    budget.record_spend(3.0)  # 30% consumed
    _boot_spine(budget=budget)

    cheap = make_model(alias="cheap:model", cost_hints=CostHints(input_per_1k_tokens=0.0001))
    expensive = make_model(alias="expensive:model", cost_hints=CostHints(input_per_1k_tokens=0.03))

    call = make_call()
    weights = _weights_only_factor("budget_pressure", 100.0)

    result = compute_routing_order(call, [expensive, cheap], weights=weights)
    # Both should get neutral (weight*0.5 = 50)
    for ranked in result.ordered:
        assert ranked.score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 3. Task affinity boosts matched model
# ---------------------------------------------------------------------------

def test_task_affinity_boosts_matched_model():
    """Model with affinity rule for the call's intent gets boosted."""
    prefs = Preferences(
        affinities=[
            AffinityRule(intent="lookup", model_alias="boosted:model", boost=50.0),
        ]
    )
    _boot_spine(preferences=prefs)

    boosted = make_model(alias="boosted:model")
    plain = make_model(alias="plain:model")

    call = make_call(intent_class=__import__("ControlCore.schemas", fromlist=["IntentClass"]).IntentClass.lookup)
    weights = _weights_only_factor("task_affinity", 25.0)

    result = compute_routing_order(call, [plain, boosted], weights=weights)
    assert result.ordered_aliases[0] == "boosted:model"


def test_task_affinity_no_match_gives_zero():
    """Model with no affinity rules gets zero boost."""
    prefs = Preferences(affinities=[])
    _boot_spine(preferences=prefs)

    model = make_model(alias="plain:model")
    call = make_call()
    weights = _weights_only_factor("task_affinity", 25.0)

    result = compute_routing_order(call, [model], weights=weights)
    ranked = result.ordered[0]
    # Find the task affinity reason
    affinity_reason = [r for r in ranked.reasons if r.factor.value == "task_affinity"][0]
    assert affinity_reason.score_contribution == 0.0


# ---------------------------------------------------------------------------
# 4. Load balance jitter
# ---------------------------------------------------------------------------

def test_load_balance_jitter_zero_is_deterministic():
    """With jitter=0, routing is fully deterministic."""
    model_a = make_model(alias="model-a")
    model_b = make_model(alias="model-b")
    call = make_call()
    weights = RoutingWeights(load_balance_jitter=0.0)

    result1 = compute_routing_order(call, [model_a, model_b], weights=weights)
    result2 = compute_routing_order(call, [model_a, model_b], weights=weights)

    assert result1.ordered_aliases == result2.ordered_aliases
    assert [m.score for m in result1.ordered] == [m.score for m in result2.ordered]


def test_load_balance_jitter_positive_does_not_crash():
    """With jitter > 0, routing still returns valid results."""
    model_a = make_model(alias="model-a")
    model_b = make_model(alias="model-b")
    call = make_call()
    weights = RoutingWeights(load_balance_jitter=5.0)

    result = compute_routing_order(call, [model_a, model_b], weights=weights)

    assert len(result.ordered) == 2
    assert set(result.ordered_aliases) == {"model-a", "model-b"}
    for ranked in result.ordered:
        assert isinstance(ranked.score, float)


# ---------------------------------------------------------------------------
# 5. No spine = graceful fallback
# ---------------------------------------------------------------------------

def test_no_spine_all_factors_still_work():
    """Without spine booted, router returns valid results with neutral scores for new factors."""
    model = make_model(alias="test:model")
    call = make_call()

    # Don't boot spine — just call the router
    result = compute_routing_order(call, [model])

    assert len(result.ordered) == 1
    assert result.ordered[0].alias == "test:model"
    assert isinstance(result.ordered[0].score, float)
    assert result.ordered[0].score > 0.0

    # Verify new factors produced reasons
    factor_values = {r.factor.value for r in result.ordered[0].reasons}
    assert "observed_latency" in factor_values
    assert "budget_pressure" in factor_values
    assert "task_affinity" in factor_values
    assert "load_balance" in factor_values


def test_no_spine_multiple_models_still_ranked():
    """Without spine, multi-model routing uses existing factors and falls back gracefully."""
    from ControlCore.registry.schema import TrustTier

    trusted = make_model(alias="trusted:model", trust_tier=TrustTier.trusted)
    standard = make_model(alias="standard:model", trust_tier=TrustTier.standard)
    call = make_call()

    result = compute_routing_order(call, [standard, trusted])

    assert result.ordered_aliases[0] == "trusted:model"
    assert result.ordered_aliases[1] == "standard:model"
