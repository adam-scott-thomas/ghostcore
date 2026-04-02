"""
Tests for ControlCore/registry/fallback.py

Covers:
- All five factory functions (default, fail_fast, aggressive_retry, queue_preferred, cost_sensitive)
- FallbackPolicy field contracts
- to_dict() serialization
- ModelSwitchCondition and QueueEscalationCondition enum values
- describe() output
- Validation constraints (max_same_model_retries < max_total_attempts)
"""

from __future__ import annotations

import pytest

from ControlCore.registry.fallback import (
    FallbackPolicy,
    ModelSwitchCondition,
    QueueEscalationCondition,
    RephraseConfig,
    RephraseStrategy,
    ModelSwitchConfig,
    QueueEscalationConfig,
    RetryTiming,
    aggressive_retry_policy,
    cost_sensitive_policy,
    default_policy,
    fail_fast_policy,
    queue_preferred_policy,
)


# ---------------------------------------------------------------------------
# default_policy()
# ---------------------------------------------------------------------------

class TestDefaultPolicy:
    def test_returns_fallback_policy(self):
        p = default_policy()
        assert isinstance(p, FallbackPolicy)

    def test_max_total_attempts_is_reasonable(self):
        p = default_policy()
        assert p.max_total_attempts >= 2

    def test_max_same_model_retries_less_than_total_attempts(self):
        p = default_policy()
        assert p.max_same_model_retries < p.max_total_attempts

    def test_model_switch_enabled(self):
        p = default_policy()
        assert p.model_switch.enabled is True

    def test_fail_fast_is_false(self):
        p = default_policy()
        assert p.fail_fast is False

    def test_audit_all_attempts_enabled(self):
        p = default_policy()
        assert p.audit_all_attempts is True

    def test_preserve_partial_results_enabled(self):
        p = default_policy()
        assert p.preserve_partial_results is True

    def test_policy_version_present(self):
        p = default_policy()
        assert p.policy_version and len(p.policy_version) > 0


# ---------------------------------------------------------------------------
# fail_fast_policy()
# ---------------------------------------------------------------------------

class TestFailFastPolicy:
    def test_returns_fallback_policy(self):
        p = fail_fast_policy()
        assert isinstance(p, FallbackPolicy)

    def test_fail_fast_flag_true(self):
        p = fail_fast_policy()
        assert p.fail_fast is True

    def test_max_total_attempts_is_one(self):
        p = fail_fast_policy()
        assert p.max_total_attempts == 1

    def test_max_same_model_retries_is_zero(self):
        p = fail_fast_policy()
        assert p.max_same_model_retries == 0

    def test_rephrase_disabled(self):
        p = fail_fast_policy()
        assert p.rephrase.enabled is False

    def test_model_switch_disabled(self):
        p = fail_fast_policy()
        assert p.model_switch.enabled is False

    def test_max_models_to_try_is_one(self):
        p = fail_fast_policy()
        assert p.model_switch.max_models_to_try == 1

    def test_policy_id_is_fail_fast(self):
        p = fail_fast_policy()
        assert p.policy_id == "fail_fast"


# ---------------------------------------------------------------------------
# aggressive_retry_policy()
# ---------------------------------------------------------------------------

class TestAggressiveRetryPolicy:
    def test_returns_fallback_policy(self):
        p = aggressive_retry_policy()
        assert isinstance(p, FallbackPolicy)

    def test_max_total_attempts_is_high(self):
        p = aggressive_retry_policy()
        assert p.max_total_attempts >= 4

    def test_max_models_to_try_is_high(self):
        p = aggressive_retry_policy()
        assert p.model_switch.max_models_to_try >= 4

    def test_model_switch_enabled(self):
        p = aggressive_retry_policy()
        assert p.model_switch.enabled is True

    def test_rephrase_enabled(self):
        p = aggressive_retry_policy()
        assert p.rephrase.enabled is True

    def test_rephrase_has_multiple_strategies(self):
        p = aggressive_retry_policy()
        assert len(p.rephrase.allowed_strategies) >= 2

    def test_max_rephrase_attempts_is_high(self):
        p = aggressive_retry_policy()
        assert p.rephrase.max_rephrase_attempts >= 3

    def test_policy_id_is_aggressive_retry(self):
        p = aggressive_retry_policy()
        assert p.policy_id == "aggressive_retry"

    def test_consistency_constraint_satisfied(self):
        p = aggressive_retry_policy()
        assert p.max_same_model_retries < p.max_total_attempts


# ---------------------------------------------------------------------------
# queue_preferred_policy()
# ---------------------------------------------------------------------------

class TestQueuePreferredPolicy:
    def test_returns_fallback_policy(self):
        p = queue_preferred_policy()
        assert isinstance(p, FallbackPolicy)

    def test_queue_escalation_enabled(self):
        p = queue_preferred_policy()
        assert p.queue_escalation.enabled is True

    def test_soft_timeout_threshold_is_low(self):
        # queue_preferred should escalate quickly
        p = queue_preferred_policy()
        assert p.queue_escalation.soft_timeout_threshold_ms <= 3000

    def test_high_latency_condition_present(self):
        p = queue_preferred_policy()
        conditions = p.queue_escalation.conditions
        assert QueueEscalationCondition.high_latency_expected in conditions

    def test_all_models_exhausted_condition_present(self):
        p = queue_preferred_policy()
        conditions = p.queue_escalation.conditions
        assert QueueEscalationCondition.all_models_exhausted in conditions

    def test_policy_id_is_queue_preferred(self):
        p = queue_preferred_policy()
        assert p.policy_id == "queue_preferred"

    def test_consistency_constraint_satisfied(self):
        p = queue_preferred_policy()
        assert p.max_same_model_retries < p.max_total_attempts


# ---------------------------------------------------------------------------
# cost_sensitive_policy()
# ---------------------------------------------------------------------------

class TestCostSensitivePolicy:
    def test_returns_fallback_policy(self):
        p = cost_sensitive_policy()
        assert isinstance(p, FallbackPolicy)

    def test_max_total_attempts_is_low(self):
        p = cost_sensitive_policy()
        assert p.max_total_attempts <= 3

    def test_max_models_to_try_is_low(self):
        p = cost_sensitive_policy()
        assert p.model_switch.max_models_to_try <= 3

    def test_only_narrower_rephrase_strategy(self):
        p = cost_sensitive_policy()
        assert p.rephrase.allowed_strategies == [RephraseStrategy.narrower]

    def test_max_rephrase_attempts_is_one(self):
        p = cost_sensitive_policy()
        assert p.rephrase.max_rephrase_attempts == 1

    def test_switch_conditions_limited(self):
        p = cost_sensitive_policy()
        conditions = p.model_switch.conditions
        assert ModelSwitchCondition.refusal in conditions
        assert ModelSwitchCondition.error in conditions

    def test_policy_id_is_cost_sensitive(self):
        p = cost_sensitive_policy()
        assert p.policy_id == "cost_sensitive"

    def test_same_model_retries_is_zero(self):
        p = cost_sensitive_policy()
        assert p.max_same_model_retries == 0

    def test_consistency_constraint_satisfied(self):
        p = cost_sensitive_policy()
        assert p.max_same_model_retries < p.max_total_attempts


# ---------------------------------------------------------------------------
# to_dict() serialization
# ---------------------------------------------------------------------------

class TestPolicySerialization:
    def test_to_dict_returns_dict(self):
        d = default_policy().to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_max_total_attempts(self):
        d = default_policy().to_dict()
        assert "max_total_attempts" in d

    def test_to_dict_has_max_same_model_retries(self):
        d = default_policy().to_dict()
        assert "max_same_model_retries" in d

    def test_to_dict_has_fail_fast(self):
        d = default_policy().to_dict()
        assert "fail_fast" in d

    def test_to_dict_has_timing_block(self):
        d = default_policy().to_dict()
        assert "timing" in d
        assert isinstance(d["timing"], dict)

    def test_to_dict_has_rephrase_block(self):
        d = default_policy().to_dict()
        assert "rephrase" in d
        assert isinstance(d["rephrase"], dict)

    def test_to_dict_has_model_switch_block(self):
        d = default_policy().to_dict()
        assert "model_switch" in d
        assert isinstance(d["model_switch"], dict)

    def test_to_dict_has_queue_escalation_block(self):
        d = default_policy().to_dict()
        assert "queue_escalation" in d
        assert isinstance(d["queue_escalation"], dict)

    def test_to_dict_has_policy_version(self):
        d = default_policy().to_dict()
        assert "policy_version" in d

    def test_to_dict_has_audit_all_attempts(self):
        d = default_policy().to_dict()
        assert "audit_all_attempts" in d

    def test_to_dict_has_preserve_partial_results(self):
        d = default_policy().to_dict()
        assert "preserve_partial_results" in d

    def test_fail_fast_policy_serializes_correctly(self):
        d = fail_fast_policy().to_dict()
        assert d["fail_fast"] is True
        assert d["max_total_attempts"] == 1

    def test_serialized_rephrase_strategies_are_strings(self):
        d = default_policy().to_dict()
        strategies = d["rephrase"]["allowed_strategies"]
        assert all(isinstance(s, str) for s in strategies)

    def test_serialized_model_switch_conditions_are_strings(self):
        d = default_policy().to_dict()
        conditions = d["model_switch"]["conditions"]
        assert all(isinstance(c, str) for c in conditions)


# ---------------------------------------------------------------------------
# ModelSwitchCondition enum values
# ---------------------------------------------------------------------------

class TestModelSwitchCondition:
    def test_refusal_value(self):
        assert ModelSwitchCondition.refusal.value == "refusal"

    def test_timeout_value(self):
        assert ModelSwitchCondition.timeout.value == "timeout"

    def test_low_confidence_exists(self):
        assert ModelSwitchCondition.low_confidence is not None

    def test_error_exists(self):
        assert ModelSwitchCondition.error is not None

    def test_rate_limit_exists(self):
        assert ModelSwitchCondition.rate_limit is not None

    def test_content_filter_exists(self):
        assert ModelSwitchCondition.content_filter is not None

    def test_default_policy_includes_refusal_and_timeout(self):
        p = default_policy()
        conditions = p.model_switch.conditions
        assert ModelSwitchCondition.refusal in conditions
        assert ModelSwitchCondition.timeout in conditions


# ---------------------------------------------------------------------------
# QueueEscalationCondition enum values
# ---------------------------------------------------------------------------

class TestQueueEscalationCondition:
    def test_all_models_exhausted_value(self):
        assert QueueEscalationCondition.all_models_exhausted.value == "all_models_exhausted"

    def test_timeout_exceeded_value(self):
        assert QueueEscalationCondition.timeout_exceeded.value == "timeout_exceeded"

    def test_explicit_queue_exists(self):
        assert QueueEscalationCondition.explicit_queue is not None

    def test_high_latency_expected_exists(self):
        assert QueueEscalationCondition.high_latency_expected is not None


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_returns_non_empty_string(self):
        result = default_policy().describe()
        assert isinstance(result, str) and len(result) > 0

    def test_contains_max_attempts(self):
        result = default_policy().describe()
        assert "Max attempts" in result or "max_total_attempts" in result.lower() or str(default_policy().max_total_attempts) in result

    def test_contains_fail_fast(self):
        result = default_policy().describe()
        assert "Fail fast" in result or "fail_fast" in result.lower()

    def test_contains_rephrase_section(self):
        result = default_policy().describe()
        assert "Rephrase" in result

    def test_contains_model_switch_section(self):
        result = default_policy().describe()
        assert "Model Switch" in result

    def test_contains_queue_escalation_section(self):
        result = default_policy().describe()
        assert "Queue Escalation" in result

    def test_fail_fast_policy_describes_correctly(self):
        result = fail_fast_policy().describe()
        assert "True" in result  # fail_fast: True

    def test_all_factories_produce_valid_descriptions(self):
        factories = [
            default_policy,
            fail_fast_policy,
            aggressive_retry_policy,
            queue_preferred_policy,
            cost_sensitive_policy,
        ]
        for factory in factories:
            result = factory().describe()
            assert isinstance(result, str) and len(result) > 0, f"{factory.__name__} describe() returned empty"


# ---------------------------------------------------------------------------
# Validation constraints
# ---------------------------------------------------------------------------

class TestValidationConstraints:
    def test_same_model_retries_must_be_less_than_total_attempts(self):
        with pytest.raises(Exception):
            FallbackPolicy(
                max_total_attempts=2,
                max_same_model_retries=2,  # equal — must fail
            )

    def test_same_model_retries_equal_to_total_attempts_rejected(self):
        with pytest.raises(Exception):
            FallbackPolicy(
                max_total_attempts=3,
                max_same_model_retries=3,
            )

    def test_valid_policy_with_zero_same_model_retries(self):
        p = FallbackPolicy(max_total_attempts=2, max_same_model_retries=0)
        assert p.max_total_attempts == 2

    def test_fail_fast_with_max_attempts_one_is_valid(self):
        p = FallbackPolicy(
            max_total_attempts=1,
            max_same_model_retries=0,
            fail_fast=True,
            rephrase=RephraseConfig(enabled=False, allowed_strategies=[RephraseStrategy.none]),
            model_switch=ModelSwitchConfig(enabled=False, max_models_to_try=1),
        )
        assert p.fail_fast is True
