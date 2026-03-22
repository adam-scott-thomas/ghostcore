"""
P2-05: Fallback & Retry Policy Descriptor

Defines policy as DATA (schema), not execution code.
Auditable, overridable, attachable to ControlCoreCall or system default.

This module does NOT:
- Perform retries
- Inspect model outputs
- Trigger execution
- Make network calls
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class RephraseStrategy(str, Enum):
    """Allowed strategies for rephrasing prompts on retry."""
    neutral = "neutral"        # Keep same meaning, different wording
    narrower = "narrower"      # Reduce scope, more specific ask
    partial = "partial"        # Request partial/incremental response
    simplify = "simplify"      # Simpler language, shorter prompt
    none = "none"              # No rephrasing, exact retry


class ModelSwitchCondition(str, Enum):
    """Conditions that trigger switching to next model in routing order."""
    refusal = "refusal"              # Model refused the request
    timeout = "timeout"              # Request timed out
    low_confidence = "low_confidence"  # Model reported low confidence
    error = "error"                  # Adapter/model error
    rate_limit = "rate_limit"        # Rate limited by provider
    content_filter = "content_filter"  # Blocked by content filter


class QueueEscalationCondition(str, Enum):
    """Conditions that trigger returning job_id for async processing."""
    all_models_exhausted = "all_models_exhausted"  # Tried all models
    timeout_exceeded = "timeout_exceeded"          # Soft timeout exceeded
    explicit_queue = "explicit_queue"              # Caller requested queue
    high_latency_expected = "high_latency_expected"  # Model is known slow


class RetryTiming(BaseModel):
    """Timing configuration for retries."""
    model_config = ConfigDict(extra="forbid")

    initial_delay_ms: int = Field(100, ge=0, le=60000, description="Initial delay before first retry")
    max_delay_ms: int = Field(5000, ge=0, le=300000, description="Maximum delay between retries")
    backoff_multiplier: float = Field(2.0, ge=1.0, le=10.0, description="Exponential backoff multiplier")
    jitter: bool = Field(True, description="Add random jitter to delays")


class RephraseConfig(BaseModel):
    """Configuration for prompt rephrasing on retry."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether rephrasing is allowed")
    allowed_strategies: List[RephraseStrategy] = Field(
        default_factory=lambda: [RephraseStrategy.neutral, RephraseStrategy.narrower],
        description="Which rephrase strategies are permitted",
    )
    max_rephrase_attempts: int = Field(2, ge=0, le=5, description="Max times to rephrase same prompt")
    preserve_intent: bool = Field(True, description="Rephrasing must preserve original intent")

    @field_validator("allowed_strategies")
    @classmethod
    def validate_strategies(cls, v: List[RephraseStrategy]) -> List[RephraseStrategy]:
        if not v:
            # If empty, default to none (no rephrasing)
            return [RephraseStrategy.none]
        return v


class ModelSwitchConfig(BaseModel):
    """Configuration for switching between models."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether model switching is allowed")
    conditions: List[ModelSwitchCondition] = Field(
        default_factory=lambda: [
            ModelSwitchCondition.refusal,
            ModelSwitchCondition.timeout,
            ModelSwitchCondition.error,
        ],
        description="Conditions that trigger model switch",
    )
    max_models_to_try: int = Field(3, ge=1, le=10, description="Maximum models to try before giving up")
    cooldown_ms: int = Field(0, ge=0, le=60000, description="Delay before trying next model")


class QueueEscalationConfig(BaseModel):
    """Configuration for escalating to queue (returning job_id)."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether queue escalation is allowed")
    conditions: List[QueueEscalationCondition] = Field(
        default_factory=lambda: [
            QueueEscalationCondition.all_models_exhausted,
            QueueEscalationCondition.timeout_exceeded,
        ],
        description="Conditions that trigger queue escalation",
    )
    soft_timeout_threshold_ms: int = Field(
        5000, ge=1000, le=60000,
        description="Return job_id if processing exceeds this threshold",
    )


class FallbackPolicy(BaseModel):
    """
    Complete fallback and retry policy descriptor.

    This is DATA describing policy, not execution logic.
    Attach to ControlCoreCall.options or use as system default.

    Auditable: Every field has clear meaning
    Overridable: Call-level policy overrides system default
    """
    model_config = ConfigDict(extra="forbid")

    # Identity
    policy_id: Optional[str] = Field(None, max_length=64, description="Optional policy identifier")
    policy_version: str = Field("1.0.0", description="Policy schema version")

    # Retry limits
    max_total_attempts: int = Field(3, ge=1, le=10, description="Maximum total attempts across all strategies")
    max_same_model_retries: int = Field(1, ge=0, le=5, description="Max retries on same model before switching")

    # Timing
    timing: RetryTiming = Field(default_factory=RetryTiming)

    # Rephrase
    rephrase: RephraseConfig = Field(default_factory=RephraseConfig)

    # Model switching
    model_switch: ModelSwitchConfig = Field(default_factory=ModelSwitchConfig)

    # Queue escalation
    queue_escalation: QueueEscalationConfig = Field(default_factory=QueueEscalationConfig)

    # Behavior flags
    fail_fast: bool = Field(False, description="If true, don't retry - fail immediately on first error")
    preserve_partial_results: bool = Field(True, description="Keep partial results from failed attempts")
    audit_all_attempts: bool = Field(True, description="Record all attempts in result provenance")

    @model_validator(mode="after")
    def validate_consistency(self) -> "FallbackPolicy":
        # If fail_fast, other retry settings are moot but still valid
        if self.fail_fast and self.max_total_attempts > 1:
            # This is allowed but unusual - log it in description
            pass

        # max_same_model_retries shouldn't exceed max_total_attempts
        if self.max_same_model_retries >= self.max_total_attempts:
            raise ValueError(
                f"max_same_model_retries ({self.max_same_model_retries}) must be < "
                f"max_total_attempts ({self.max_total_attempts})"
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    def describe(self) -> str:
        """Human-readable description of the policy."""
        lines = [
            f"Fallback Policy (v{self.policy_version})",
            f"  Max attempts: {self.max_total_attempts}",
            f"  Same-model retries: {self.max_same_model_retries}",
            f"  Fail fast: {self.fail_fast}",
            "",
            "  Rephrase:",
            f"    Enabled: {self.rephrase.enabled}",
            f"    Strategies: {[s.value for s in self.rephrase.allowed_strategies]}",
            "",
            "  Model Switch:",
            f"    Enabled: {self.model_switch.enabled}",
            f"    Conditions: {[c.value for c in self.model_switch.conditions]}",
            f"    Max models: {self.model_switch.max_models_to_try}",
            "",
            "  Queue Escalation:",
            f"    Enabled: {self.queue_escalation.enabled}",
            f"    Conditions: {[c.value for c in self.queue_escalation.conditions]}",
            f"    Soft timeout: {self.queue_escalation.soft_timeout_threshold_ms}ms",
        ]
        return "\n".join(lines)


# --- Preset Policies ---

def default_policy() -> FallbackPolicy:
    """Default balanced policy."""
    return FallbackPolicy()


def aggressive_retry_policy() -> FallbackPolicy:
    """Policy that tries hard before failing."""
    return FallbackPolicy(
        policy_id="aggressive_retry",
        max_total_attempts=5,
        max_same_model_retries=2,
        rephrase=RephraseConfig(
            enabled=True,
            allowed_strategies=[
                RephraseStrategy.neutral,
                RephraseStrategy.narrower,
                RephraseStrategy.simplify,
            ],
            max_rephrase_attempts=3,
        ),
        model_switch=ModelSwitchConfig(
            enabled=True,
            max_models_to_try=5,
            conditions=[
                ModelSwitchCondition.refusal,
                ModelSwitchCondition.timeout,
                ModelSwitchCondition.error,
                ModelSwitchCondition.low_confidence,
            ],
        ),
    )


def fail_fast_policy() -> FallbackPolicy:
    """Policy that fails immediately on first error."""
    return FallbackPolicy(
        policy_id="fail_fast",
        max_total_attempts=1,
        max_same_model_retries=0,
        fail_fast=True,
        rephrase=RephraseConfig(enabled=False, allowed_strategies=[RephraseStrategy.none]),
        model_switch=ModelSwitchConfig(enabled=False, max_models_to_try=1),
    )


def queue_preferred_policy() -> FallbackPolicy:
    """Policy that prefers queueing over blocking."""
    return FallbackPolicy(
        policy_id="queue_preferred",
        max_total_attempts=2,
        max_same_model_retries=1,
        queue_escalation=QueueEscalationConfig(
            enabled=True,
            soft_timeout_threshold_ms=2000,  # Queue quickly
            conditions=[
                QueueEscalationCondition.timeout_exceeded,
                QueueEscalationCondition.high_latency_expected,
                QueueEscalationCondition.all_models_exhausted,
            ],
        ),
    )


def cost_sensitive_policy() -> FallbackPolicy:
    """Policy that minimizes cost by limiting retries."""
    return FallbackPolicy(
        policy_id="cost_sensitive",
        max_total_attempts=2,
        max_same_model_retries=0,
        rephrase=RephraseConfig(
            enabled=True,
            allowed_strategies=[RephraseStrategy.narrower],  # Only allow narrowing
            max_rephrase_attempts=1,
        ),
        model_switch=ModelSwitchConfig(
            enabled=True,
            max_models_to_try=2,  # Don't try too many
            conditions=[ModelSwitchCondition.refusal, ModelSwitchCondition.error],
        ),
    )


# Example policy as JSON for documentation
EXAMPLE_POLICY_JSON = """
{
  "policy_id": "example_balanced",
  "policy_version": "1.0.0",
  "max_total_attempts": 3,
  "max_same_model_retries": 1,
  "timing": {
    "initial_delay_ms": 100,
    "max_delay_ms": 5000,
    "backoff_multiplier": 2.0,
    "jitter": true
  },
  "rephrase": {
    "enabled": true,
    "allowed_strategies": ["neutral", "narrower"],
    "max_rephrase_attempts": 2,
    "preserve_intent": true
  },
  "model_switch": {
    "enabled": true,
    "conditions": ["refusal", "timeout", "error"],
    "max_models_to_try": 3,
    "cooldown_ms": 0
  },
  "queue_escalation": {
    "enabled": true,
    "conditions": ["all_models_exhausted", "timeout_exceeded"],
    "soft_timeout_threshold_ms": 5000
  },
  "fail_fast": false,
  "preserve_partial_results": true,
  "audit_all_attempts": true
}
"""
