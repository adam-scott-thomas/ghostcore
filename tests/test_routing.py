"""
Tests for ControlCore.registry.routing — compute_routing_order()

Covers:
  - Higher trust tier ranked first
  - Cheaper model preferred when trust is tied
  - Empty eligible list returns empty result
  - Routing is deterministic (same input = same output)
  - Rank indices are 1-based and sequential
  - Top property returns the first model
  - Ordered aliases property works
  - Custom weights change ordering
  - Refusal history affects ranking
"""

from __future__ import annotations

import pytest

from ControlCore.registry.routing import (
    RefusalHistory,
    RoutingResult,
    RoutingWeights,
    compute_routing_order,
)
from ControlCore.registry.schema import CostHints, TrustTier, TimeoutDefaults
from ControlCore.schemas import IntentClass

from tests.conftest import make_call, make_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trusted_model(alias: str = "judge:trusted", **kwargs):
    return make_model(alias=alias, trust_tier=TrustTier.trusted, **kwargs)


def make_standard_model(alias: str = "qwen:32b", **kwargs):
    return make_model(alias=alias, trust_tier=TrustTier.standard, **kwargs)


def make_untrusted_model(alias: str = "exp:llama3", **kwargs):
    return make_model(alias=alias, trust_tier=TrustTier.untrusted, **kwargs)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_eligible_list_returns_empty():
    call = make_call()
    result = compute_routing_order(call, [])

    assert isinstance(result, RoutingResult)
    assert result.ordered == []
    assert result.ordered_aliases == []
    assert result.top is None


# ---------------------------------------------------------------------------
# Trust tier ordering
# ---------------------------------------------------------------------------

def test_higher_trust_tier_ranked_first():
    trusted = make_trusted_model(alias="judge:trusted")
    standard = make_standard_model(alias="qwen:32b")
    untrusted = make_untrusted_model(alias="exp:llama3")

    call = make_call()
    result = compute_routing_order(call, [untrusted, standard, trusted])

    aliases = result.ordered_aliases
    assert aliases[0] == "judge:trusted", f"Expected trusted first, got: {aliases}"
    assert aliases[-1] == "exp:llama3", f"Expected untrusted last, got: {aliases}"


def test_trusted_model_is_top():
    trusted = make_trusted_model(alias="judge:trusted")
    standard = make_standard_model(alias="qwen:32b")

    call = make_call()
    result = compute_routing_order(call, [standard, trusted])

    assert result.top is not None
    assert result.top.alias == "judge:trusted"


def test_two_models_same_tier_ranked_by_other_factors():
    a = make_standard_model(alias="model-a")
    b = make_standard_model(alias="model-b")

    call = make_call()
    result = compute_routing_order(call, [a, b])

    # Both should appear, order may differ but must be stable
    assert set(result.ordered_aliases) == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# Cost ordering (tiebreaker)
# ---------------------------------------------------------------------------

def test_cheaper_model_preferred_when_trust_tied():
    """When two standard-tier models have the same trust tier, cheaper one scores higher."""
    cheap = make_standard_model(
        alias="cheap:model",
        cost_hints=CostHints(input_per_1k_tokens=0.0005),
    )
    expensive = make_standard_model(
        alias="expensive:model",
        cost_hints=CostHints(input_per_1k_tokens=0.018),
    )

    # Use weights that only care about cost (zero out everything else)
    weights = RoutingWeights(
        trust_tier=0.0,
        capability_match=0.0,
        latency_hint=0.0,
        cost_hint=10.0,
        context_headroom=0.0,
        refusal_rate=0.0,
    )

    call = make_call()
    result = compute_routing_order(call, [expensive, cheap], weights=weights)

    assert result.ordered_aliases[0] == "cheap:model"


def test_no_cost_hints_gets_neutral_score():
    """Models without cost hints still get ranked (neutral score applied)."""
    no_cost = make_standard_model(alias="nocost:model", cost_hints=None)
    call = make_call()
    result = compute_routing_order(call, [no_cost])

    assert len(result.ordered) == 1
    assert result.ordered[0].alias == "nocost:model"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_routing_is_deterministic_same_input_same_output():
    trusted = make_trusted_model(alias="judge:trusted")
    standard = make_standard_model(alias="qwen:32b")
    untrusted = make_untrusted_model(alias="exp:llama3")
    models = [trusted, standard, untrusted]
    call = make_call()

    result1 = compute_routing_order(call, models)
    result2 = compute_routing_order(call, models)

    assert result1.ordered_aliases == result2.ordered_aliases
    assert [m.score for m in result1.ordered] == [m.score for m in result2.ordered]


def test_routing_deterministic_with_reversed_input():
    """Input order should not affect output order."""
    trusted = make_trusted_model(alias="judge:trusted")
    standard = make_standard_model(alias="qwen:32b")
    models_fwd = [trusted, standard]
    models_rev = [standard, trusted]
    call = make_call()

    result_fwd = compute_routing_order(call, models_fwd)
    result_rev = compute_routing_order(call, models_rev)

    assert result_fwd.ordered_aliases == result_rev.ordered_aliases


# ---------------------------------------------------------------------------
# Rank properties
# ---------------------------------------------------------------------------

def test_rank_indices_are_one_based_and_sequential():
    models = [
        make_standard_model(alias="model-a"),
        make_standard_model(alias="model-b"),
        make_standard_model(alias="model-c"),
    ]
    call = make_call()
    result = compute_routing_order(call, models)

    ranks = [m.rank for m in result.ordered]
    assert ranks == [1, 2, 3]


def test_single_model_gets_rank_one():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    result = compute_routing_order(call, [model])

    assert result.ordered[0].rank == 1
    assert result.ordered[0].alias == "qwen:32b"


def test_each_ranked_model_has_score():
    trusted = make_trusted_model(alias="judge:trusted")
    standard = make_standard_model(alias="qwen:32b")
    call = make_call()

    result = compute_routing_order(call, [trusted, standard])

    for ranked in result.ordered:
        assert isinstance(ranked.score, float)
        assert ranked.score >= 0.0


# ---------------------------------------------------------------------------
# Reasons / explainability
# ---------------------------------------------------------------------------

def test_each_ranked_model_has_reasons():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    result = compute_routing_order(call, [model])

    ranked = result.ordered[0]
    assert len(ranked.reasons) > 0


def test_explain_all_returns_string():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    result = compute_routing_order(call, [model])

    explanation = result.explain_all()
    assert isinstance(explanation, str)
    assert "qwen:32b" in explanation


def test_explain_all_empty_result():
    call = make_call()
    result = compute_routing_order(call, [])

    explanation = result.explain_all()
    assert "No models" in explanation


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

def test_custom_weights_applied():
    """With trust weight=0, slower/cheaper differences drive ordering."""
    fast = make_standard_model(
        alias="fast:model",
        timeouts=TimeoutDefaults(soft_ms=5000, hard_ms=30000),
    )
    slow = make_standard_model(
        alias="slow:model",
        timeouts=TimeoutDefaults(soft_ms=55000, hard_ms=120000),
    )
    weights = RoutingWeights(
        trust_tier=0.0,
        capability_match=0.0,
        latency_hint=100.0,
        cost_hint=0.0,
        context_headroom=0.0,
        refusal_rate=0.0,
    )
    call = make_call()
    result = compute_routing_order(call, [slow, fast], weights=weights)

    assert result.ordered_aliases[0] == "fast:model"


# ---------------------------------------------------------------------------
# Refusal history
# ---------------------------------------------------------------------------

def test_high_refusal_rate_deprioritized():
    """Model with high refusal rate should score lower than clean model."""
    clean = make_standard_model(alias="clean:model")
    refuser = make_standard_model(alias="refuser:model")

    history = RefusalHistory(rates={"refuser:model": 0.45, "clean:model": 0.0})

    weights = RoutingWeights(
        trust_tier=0.0,
        capability_match=0.0,
        latency_hint=0.0,
        cost_hint=0.0,
        context_headroom=0.0,
        refusal_rate=100.0,
    )
    call = make_call()
    result = compute_routing_order(call, [refuser, clean], weights=weights, refusal_history=history)

    assert result.ordered_aliases[0] == "clean:model"


def test_no_refusal_history_gives_neutral_score():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    # No refusal_history passed → uses default neutral 0.8 score
    result = compute_routing_order(call, [model], refusal_history=None)

    assert len(result.ordered) == 1


# ---------------------------------------------------------------------------
# RoutingResult metadata
# ---------------------------------------------------------------------------

def test_routing_result_includes_call_id():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    result = compute_routing_order(call, [model])

    assert result.call_id == call.call_id


def test_routing_result_policy_version_set():
    call = make_call()
    result = compute_routing_order(call, [])

    assert result.policy_version == "1.0.0"


def test_to_dict_serializable():
    model = make_standard_model(alias="qwen:32b")
    call = make_call()
    result = compute_routing_order(call, [model])

    d = result.to_dict()
    assert "ordered" in d
    assert "policy_version" in d
    assert d["ordered"][0]["alias"] == "qwen:32b"
