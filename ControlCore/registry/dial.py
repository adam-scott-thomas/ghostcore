"""
P2-03: Dial Eligibility Filter

Pure function that filters eligible models based on call requirements.
NO routing logic, NO execution, NO "best" selection.

Returns deterministic list of eligible models with exclusion reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ControlCore.schemas import ControlCoreCall, TrustTier as CallTrustTier
from ControlCore.registry.schema import ModelRegistry, ModelEntry, TrustTier


class ExclusionReason(str, Enum):
    """Reasons a model may be excluded from eligibility."""
    DISABLED = "disabled"                     # Model is not enabled
    DEPRECATED = "deprecated"                 # Model is deprecated
    INTENT_NOT_SUPPORTED = "intent_not_supported"   # Model doesn't support the intent
    TRUST_INSUFFICIENT = "trust_insufficient"       # Model trust tier too low
    CONTEXT_TOO_SMALL = "context_too_small"         # Context window too small for prompt
    TIMEOUT_TOO_SHORT = "timeout_too_short"         # Model timeouts exceed call timeouts


@dataclass
class ModelExclusion:
    """Record of why a model was excluded."""
    alias: str
    reason: ExclusionReason
    detail: Optional[str] = None


@dataclass
class EligibilityResult:
    """Result of dial eligibility filtering."""
    eligible: List[ModelEntry] = field(default_factory=list)
    excluded: List[ModelExclusion] = field(default_factory=list)

    @property
    def eligible_aliases(self) -> List[str]:
        """Get list of eligible model aliases."""
        return [m.alias for m in self.eligible]

    @property
    def has_eligible(self) -> bool:
        """Check if any models are eligible."""
        return len(self.eligible) > 0

    def get_exclusion_reason(self, alias: str) -> Optional[ExclusionReason]:
        """Get exclusion reason for a specific model."""
        for exc in self.excluded:
            if exc.alias == alias:
                return exc.reason
        return None


def _map_trust_tier(call_tier: CallTrustTier) -> TrustTier:
    """Map call schema TrustTier to registry TrustTier."""
    mapping = {
        CallTrustTier.trusted: TrustTier.trusted,
        CallTrustTier.standard: TrustTier.standard,
        CallTrustTier.untrusted: TrustTier.untrusted,
    }
    return mapping.get(call_tier, TrustTier.standard)


def _estimate_prompt_tokens(prompt: str) -> int:
    """
    Rough estimate of token count for a prompt.
    Conservative: ~4 chars per token.
    """
    return len(prompt) // 4 + 1


def filter_eligible_models(
    call: ControlCoreCall,
    registry: ModelRegistry,
    *,
    min_context_buffer: int = 1024,
) -> EligibilityResult:
    """
    Filter models eligible to handle a ControlCoreCall.

    This is a PURE FUNCTION with DETERMINISTIC results.
    It does NOT:
    - Choose a "best" model
    - Execute anything
    - Call any adapters
    - Make network requests

    Args:
        call: The validated ControlCoreCall
        registry: The model registry to filter
        min_context_buffer: Minimum tokens to reserve for output (default 1024)

    Returns:
        EligibilityResult with eligible models and exclusion reasons
    """
    result = EligibilityResult()

    # Extract call requirements
    required_intent = call.intent.cls.value
    required_trust = _map_trust_tier(call.target.trust_tier)
    call_soft_timeout = call.options.timeouts.soft_ms
    call_hard_timeout = call.options.timeouts.hard_ms

    # Estimate prompt size
    prompt_tokens = _estimate_prompt_tokens(call.prompt)
    for ctx in call.context:
        prompt_tokens += _estimate_prompt_tokens(ctx.content)

    # Filter each model
    for model in registry.models:
        # Check 1: Enabled
        if not model.enabled:
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.DISABLED,
                detail="Model is disabled",
            ))
            continue

        # Check 2: Not deprecated
        if model.deprecated:
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.DEPRECATED,
                detail=model.deprecation_message or "Model is deprecated",
            ))
            continue

        # Check 3: Intent support
        if not model.supports_intent(required_intent):
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.INTENT_NOT_SUPPORTED,
                detail=f"Model does not support intent '{required_intent}'. "
                       f"Supported: {model.supported_intents or 'all'}",
            ))
            continue

        # Check 4: Trust tier
        if not model.meets_trust_requirement(required_trust):
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.TRUST_INSUFFICIENT,
                detail=f"Model trust '{model.trust_tier.value}' < required '{required_trust.value}'",
            ))
            continue

        # Check 5: Context window
        required_context = prompt_tokens + min_context_buffer
        if model.context_window < required_context:
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.CONTEXT_TOO_SMALL,
                detail=f"Model context {model.context_window} < required {required_context} "
                       f"(prompt ~{prompt_tokens} + buffer {min_context_buffer})",
            ))
            continue

        # Check 6: Timeout constraints
        # Model's minimum required time should fit within call's hard timeout
        if model.timeouts.soft_ms > call_hard_timeout:
            result.excluded.append(ModelExclusion(
                alias=model.alias,
                reason=ExclusionReason.TIMEOUT_TOO_SHORT,
                detail=f"Model soft timeout {model.timeouts.soft_ms}ms > "
                       f"call hard timeout {call_hard_timeout}ms",
            ))
            continue

        # Model passed all checks
        result.eligible.append(model)

    return result


def filter_by_capability(
    models: List[ModelEntry],
    required_capabilities: List[str],
) -> List[ModelEntry]:
    """
    Additional filter: only models with ALL required capabilities.

    This is a helper for more specific filtering after eligibility.
    """
    from ControlCore.registry.schema import CapabilityTag

    required = set()
    for cap in required_capabilities:
        try:
            required.add(CapabilityTag(cap))
        except ValueError:
            continue  # Skip unknown capabilities

    if not required:
        return models

    return [
        m for m in models
        if all(m.has_capability(cap) for cap in required)
    ]


def filter_by_provider(
    models: List[ModelEntry],
    allowed_providers: List[str],
) -> List[ModelEntry]:
    """
    Additional filter: only models from specific providers.
    """
    from ControlCore.registry.schema import Provider

    allowed = set()
    for p in allowed_providers:
        try:
            allowed.add(Provider(p))
        except ValueError:
            continue

    if not allowed:
        return models

    return [m for m in models if m.provider in allowed]
