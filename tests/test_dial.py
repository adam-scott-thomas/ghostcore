"""
Tests for ControlCore.registry.dial — filter_eligible_models()

Covers:
  - Disabled model excluded (DISABLED)
  - Deprecated model excluded (DEPRECATED)
  - Enabled model with matching intent included
  - Wrong intent excluded (INTENT_NOT_SUPPORTED)
  - Trust tier filtering (TRUST_INSUFFICIENT)
  - Multiple models filtered correctly
  - Context window too small (CONTEXT_TOO_SMALL)
  - Timeout too short (TIMEOUT_TOO_SHORT)
"""

from __future__ import annotations

import pytest

from ControlCore.registry.dial import (
    ExclusionReason,
    filter_eligible_models,
)
from ControlCore.registry.schema import ModelRegistry, TrustTier
from ControlCore.schemas import IntentClass, TrustTier as CallTrustTier

from tests.conftest import make_call, make_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_registry(*models):
    """Build a ModelRegistry from one or more ModelEntry objects."""
    return ModelRegistry(models=list(models))


# ---------------------------------------------------------------------------
# Disabled model
# ---------------------------------------------------------------------------

def test_disabled_model_excluded():
    model = make_model(alias="qwen:32b", enabled=False)
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" not in result.eligible_aliases
    assert result.get_exclusion_reason("qwen:32b") == ExclusionReason.DISABLED


def test_disabled_model_not_in_eligible_list():
    model = make_model(alias="qwen:32b", enabled=False)
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert result.has_eligible is False
    assert len(result.excluded) == 1
    assert result.excluded[0].alias == "qwen:32b"


# ---------------------------------------------------------------------------
# Deprecated model
# ---------------------------------------------------------------------------

def test_deprecated_model_excluded():
    model = make_model(alias="qwen:32b", deprecated=True)
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" not in result.eligible_aliases
    assert result.get_exclusion_reason("qwen:32b") == ExclusionReason.DEPRECATED


def test_deprecated_model_with_message():
    model = make_model(
        alias="qwen:32b",
        deprecated=True,
        deprecation_message="Use qwen:72b instead",
    )
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    exclusion = result.excluded[0]
    assert exclusion.reason == ExclusionReason.DEPRECATED
    assert "qwen:72b" in (exclusion.detail or "")


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------

def test_model_with_matching_intent_included():
    # supported_intents=[] means "supports all intents"
    model = make_model(alias="qwen:32b", supported_intents=[])
    registry = make_registry(model)
    call = make_call(intent_class=IntentClass.lookup)

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" in result.eligible_aliases
    assert result.has_eligible is True


def test_model_with_explicit_matching_intent_included():
    model = make_model(alias="qwen:32b", supported_intents=["lookup", "summarize"])
    registry = make_registry(model)
    call = make_call(intent_class=IntentClass.lookup)

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" in result.eligible_aliases


def test_wrong_intent_excluded():
    # Model only supports "summarize", call uses "lookup"
    model = make_model(alias="qwen:32b", supported_intents=["summarize"])
    registry = make_registry(model)
    call = make_call(intent_class=IntentClass.lookup)

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" not in result.eligible_aliases
    assert result.get_exclusion_reason("qwen:32b") == ExclusionReason.INTENT_NOT_SUPPORTED


def test_wrong_intent_excluded_detail_mentions_intent():
    model = make_model(alias="qwen:32b", supported_intents=["critique"])
    registry = make_registry(model)
    call = make_call(intent_class=IntentClass.draft)

    result = filter_eligible_models(call, registry)

    exc = result.excluded[0]
    assert exc.reason == ExclusionReason.INTENT_NOT_SUPPORTED
    assert "draft" in (exc.detail or "")


# ---------------------------------------------------------------------------
# Trust tier filtering
# ---------------------------------------------------------------------------

def test_trusted_call_excludes_standard_model():
    """A call requiring trusted tier must exclude standard-tier models."""
    model = make_model(alias="qwen:32b", trust_tier=TrustTier.standard)
    registry = make_registry(model)
    # Build a call that requires trusted tier
    call = make_call()
    # Manually patch target trust_tier to "trusted"
    from ControlCore.schemas import TrustTier as CallTT
    call.target.trust_tier = CallTT.trusted

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" not in result.eligible_aliases
    assert result.get_exclusion_reason("qwen:32b") == ExclusionReason.TRUST_INSUFFICIENT


def test_trusted_call_includes_trusted_model():
    model = make_model(alias="judge:trusted", trust_tier=TrustTier.trusted)
    registry = make_registry(model)
    call = make_call(target_alias="judge:trusted")
    from ControlCore.schemas import TrustTier as CallTT
    call.target.trust_tier = CallTT.trusted

    result = filter_eligible_models(call, registry)

    assert "judge:trusted" in result.eligible_aliases


def test_standard_call_includes_standard_model():
    model = make_model(alias="qwen:32b", trust_tier=TrustTier.standard)
    registry = make_registry(model)
    call = make_call()  # default trust_tier is standard

    result = filter_eligible_models(call, registry)

    assert "qwen:32b" in result.eligible_aliases


def test_standard_call_includes_trusted_model():
    """A standard-tier call should also accept a trusted-tier model (meets or exceeds requirement)."""
    model = make_model(alias="judge:trusted", trust_tier=TrustTier.trusted)
    registry = make_registry(model)
    call = make_call(target_alias="judge:trusted")
    # call.target.trust_tier stays standard

    result = filter_eligible_models(call, registry)

    assert "judge:trusted" in result.eligible_aliases


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

def test_context_window_too_small_excluded():
    # context_window=512 (minimum) with a ~200-char prompt (50 tokens) + 1024 buffer = 1074 required
    model = make_model(alias="tiny:model", context_window=512)
    registry = make_registry(model)
    # A prompt that clearly exceeds 512 - 1024 buffer headroom
    # Actually 512 < 1024 buffer alone → always excluded
    call = make_call(prompt="x")

    result = filter_eligible_models(call, registry)

    assert "tiny:model" not in result.eligible_aliases
    assert result.get_exclusion_reason("tiny:model") == ExclusionReason.CONTEXT_TOO_SMALL


def test_large_context_window_passes():
    model = make_model(alias="big:model", context_window=200000)
    registry = make_registry(model)
    call = make_call(prompt="What is the capital of France?")

    result = filter_eligible_models(call, registry)

    assert "big:model" in result.eligible_aliases


# ---------------------------------------------------------------------------
# Timeout filtering
# ---------------------------------------------------------------------------

def test_model_soft_timeout_exceeds_call_hard_timeout_excluded():
    """Model whose soft_ms > call hard_ms should be excluded."""
    from ControlCore.registry.schema import TimeoutDefaults
    model = make_model(
        alias="slow:model",
        timeouts=TimeoutDefaults(soft_ms=90000, hard_ms=180000),
    )
    registry = make_registry(model)
    # Default call hard_ms is 60000; model soft_ms=90000 > 60000
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert "slow:model" not in result.eligible_aliases
    assert result.get_exclusion_reason("slow:model") == ExclusionReason.TIMEOUT_TOO_SHORT


def test_model_soft_timeout_within_call_hard_timeout_passes():
    from ControlCore.registry.schema import TimeoutDefaults
    model = make_model(
        alias="fast:model",
        timeouts=TimeoutDefaults(soft_ms=15000, hard_ms=60000),
    )
    registry = make_registry(model)
    # Default call hard_ms=60000; model soft_ms=15000 ≤ 60000
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert "fast:model" in result.eligible_aliases


# ---------------------------------------------------------------------------
# Multiple models — mixed eligibility
# ---------------------------------------------------------------------------

def test_multiple_models_filtered_correctly():
    enabled_model = make_model(alias="good:model")
    disabled_model = make_model(alias="bad:model", enabled=False)
    deprecated_model = make_model(alias="old:model", deprecated=True)
    wrong_intent_model = make_model(alias="nope:model", supported_intents=["summarize"])

    registry = make_registry(enabled_model, disabled_model, deprecated_model, wrong_intent_model)
    call = make_call(intent_class=IntentClass.lookup)

    result = filter_eligible_models(call, registry)

    assert "good:model" in result.eligible_aliases
    assert "bad:model" not in result.eligible_aliases
    assert "old:model" not in result.eligible_aliases
    assert "nope:model" not in result.eligible_aliases

    assert len(result.eligible) == 1
    assert len(result.excluded) == 3


def test_all_models_eligible_in_permissive_registry():
    models = [
        make_model(alias="model-a", supported_intents=[]),
        make_model(alias="model-b", supported_intents=[]),
        make_model(alias="model-c", supported_intents=[]),
    ]
    registry = make_registry(*models)
    call = make_call(intent_class=IntentClass.lookup)

    result = filter_eligible_models(call, registry)

    assert len(result.eligible) == 3
    assert len(result.excluded) == 0
    assert result.has_eligible is True


def test_empty_registry_returns_empty_eligible():
    registry = make_registry()
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert result.has_eligible is False
    assert result.eligible == []
    assert result.excluded == []


# ---------------------------------------------------------------------------
# EligibilityResult properties
# ---------------------------------------------------------------------------

def test_eligible_aliases_property():
    models = [make_model(alias="a1:model"), make_model(alias="b2:model")]
    registry = make_registry(*models)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert set(result.eligible_aliases) == {"a1:model", "b2:model"}


def test_get_exclusion_reason_returns_none_for_eligible():
    model = make_model(alias="qwen:32b")
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert result.get_exclusion_reason("qwen:32b") is None


def test_get_exclusion_reason_returns_none_for_unknown():
    model = make_model(alias="qwen:32b", enabled=False)
    registry = make_registry(model)
    call = make_call()

    result = filter_eligible_models(call, registry)

    assert result.get_exclusion_reason("nonexistent:model") is None
