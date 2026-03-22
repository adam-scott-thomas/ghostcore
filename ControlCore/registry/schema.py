"""
P2-01: Model Registry Schema

Defines callable targets without executing them.
Authority, not logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set
import re

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# Alias pattern: alphanumeric with colons, hyphens, underscores
ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9:_\-]{1,63}$")


class Provider(str, Enum):
    """Model provider type."""
    local = "local"           # Local model (ollama, llama.cpp, etc.)
    remote = "remote"         # Remote self-hosted
    api_hub = "api_hub"       # Third-party API (OpenAI, Anthropic, etc.)
    other = "other"           # Custom/unknown


class CapabilityTag(str, Enum):
    """
    Capability tags describing what a model can do.
    These are declarative - no execution logic.
    """
    summarize = "summarize"
    extract = "extract"
    reason = "reason"
    judge = "judge"
    translate = "translate"
    classify = "classify"
    draft = "draft"
    compare = "compare"
    critique = "critique"
    code = "code"
    math = "math"
    creative = "creative"
    factual = "factual"
    conversational = "conversational"


class TrustTier(str, Enum):
    """Trust tier for model reliability/safety."""
    trusted = "trusted"       # Verified, production-ready
    standard = "standard"     # Default, reasonable confidence
    untrusted = "untrusted"   # Experimental, use with caution


class TimeoutDefaults(BaseModel):
    """Default timeout configuration for a model."""
    model_config = ConfigDict(extra="forbid")

    soft_ms: int = Field(15000, ge=1000, le=600000, description="Soft timeout in ms")
    hard_ms: int = Field(60000, ge=1000, le=1200000, description="Hard timeout in ms")

    @model_validator(mode="after")
    def soft_le_hard(self) -> "TimeoutDefaults":
        if self.soft_ms > self.hard_ms:
            raise ValueError("soft_ms must be <= hard_ms")
        return self


class CostHints(BaseModel):
    """
    Cost metadata (informational only, no billing logic).
    All values are hints, not guarantees.
    """
    model_config = ConfigDict(extra="forbid")

    input_per_1k_tokens: Optional[float] = Field(None, ge=0, description="Cost per 1k input tokens (USD)")
    output_per_1k_tokens: Optional[float] = Field(None, ge=0, description="Cost per 1k output tokens (USD)")
    currency: str = Field("USD", description="Currency code")
    notes: Optional[str] = Field(None, max_length=256, description="Additional cost notes")


class ModelEntry(BaseModel):
    """
    A single model registry entry.

    Describes a callable target without any execution logic.
    This is declarative authority - "what can be called" not "how to call it".
    """
    model_config = ConfigDict(extra="forbid")

    # Identity
    alias: str = Field(..., description="Unique model alias (e.g., qwen:32b, mistral:7b)")
    display_name: Optional[str] = Field(None, max_length=128, description="Human-readable name")
    description: Optional[str] = Field(None, max_length=512, description="Model description")

    # Provider
    provider: Provider = Field(..., description="Provider type")
    provider_model_id: Optional[str] = Field(None, max_length=256, description="Provider's internal model ID")

    # Capabilities
    capability_tags: List[CapabilityTag] = Field(
        default_factory=list,
        description="What this model can do"
    )
    supported_intents: List[str] = Field(
        default_factory=list,
        description="IntentClass values this model handles well"
    )

    # Trust
    trust_tier: TrustTier = Field(TrustTier.standard, description="Trust level")

    # Constraints
    context_window: int = Field(4096, ge=512, le=10_000_000, description="Context window in tokens")
    max_output_tokens: Optional[int] = Field(None, ge=1, le=1_000_000, description="Max output tokens")

    # Timeouts
    timeouts: TimeoutDefaults = Field(default_factory=TimeoutDefaults)

    # Cost (metadata only)
    cost_hints: Optional[CostHints] = Field(None, description="Cost information (hints only)")

    # Availability
    enabled: bool = Field(True, description="Whether this model is available for dialing")
    deprecated: bool = Field(False, description="Whether this model is deprecated")
    deprecation_message: Optional[str] = Field(None, max_length=256)

    # Metadata
    version: Optional[str] = Field(None, max_length=64, description="Model version")
    tags: List[str] = Field(default_factory=list, description="Additional freeform tags")

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, v: str) -> str:
        if not ALIAS_PATTERN.match(v):
            raise ValueError(
                f"Invalid alias '{v}': must be lowercase alphanumeric with colons/hyphens/underscores, "
                "2-64 chars, starting with alphanumeric"
            )
        return v

    @field_validator("supported_intents")
    @classmethod
    def validate_intents(cls, v: List[str]) -> List[str]:
        from ControlCore.schemas import IntentClass
        valid_intents = {e.value for e in IntentClass}
        for intent in v:
            if intent not in valid_intents:
                raise ValueError(f"Invalid intent '{intent}'. Valid: {sorted(valid_intents)}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        if len(v) > 32:
            raise ValueError("Too many tags (max 32)")
        for tag in v:
            if len(tag) > 64:
                raise ValueError(f"Tag too long: {tag[:20]}...")
        return v

    def supports_intent(self, intent: str) -> bool:
        """Check if this model supports a given intent."""
        # If no intents specified, assume it supports all
        if not self.supported_intents:
            return True
        return intent in self.supported_intents

    def has_capability(self, cap: CapabilityTag) -> bool:
        """Check if this model has a capability."""
        return cap in self.capability_tags

    def meets_trust_requirement(self, required: TrustTier) -> bool:
        """Check if this model meets a trust requirement."""
        trust_order = {TrustTier.untrusted: 0, TrustTier.standard: 1, TrustTier.trusted: 2}
        return trust_order[self.trust_tier] >= trust_order[required]


class ModelRegistry(BaseModel):
    """
    The complete model registry.

    Read-only authority over what models exist and their capabilities.
    No routing logic, no execution code.
    """
    model_config = ConfigDict(extra="forbid")

    version: str = Field("1.0.0", description="Registry schema version")
    models: List[ModelEntry] = Field(default_factory=list, description="Registered models")

    # Internal index built after validation
    _by_alias: Dict[str, ModelEntry] = {}

    @model_validator(mode="after")
    def build_index_and_check_duplicates(self) -> "ModelRegistry":
        seen: Set[str] = set()
        self._by_alias = {}

        for model in self.models:
            if model.alias in seen:
                raise ValueError(f"Duplicate model alias: {model.alias}")
            seen.add(model.alias)
            self._by_alias[model.alias] = model

        return self

    def get(self, alias: str) -> Optional[ModelEntry]:
        """Get a model by alias."""
        return self._by_alias.get(alias)

    def list_aliases(self) -> List[str]:
        """List all model aliases."""
        return list(self._by_alias.keys())

    def list_enabled(self) -> List[ModelEntry]:
        """List all enabled (available) models."""
        return [m for m in self.models if m.enabled and not m.deprecated]

    def list_by_provider(self, provider: Provider) -> List[ModelEntry]:
        """List models by provider."""
        return [m for m in self.models if m.provider == provider]

    def list_by_capability(self, cap: CapabilityTag) -> List[ModelEntry]:
        """List models with a specific capability."""
        return [m for m in self.models if m.has_capability(cap)]

    def list_by_trust(self, min_trust: TrustTier) -> List[ModelEntry]:
        """List models meeting minimum trust requirement."""
        return [m for m in self.models if m.meets_trust_requirement(min_trust)]

    def __len__(self) -> int:
        return len(self.models)

    def __contains__(self, alias: str) -> bool:
        return alias in self._by_alias


# Example registry as JSON for documentation
EXAMPLE_REGISTRY_JSON = """
{
  "version": "1.0.0",
  "models": [
    {
      "alias": "qwen:32b",
      "display_name": "Qwen 32B",
      "description": "Alibaba's Qwen 32B parameter model",
      "provider": "local",
      "provider_model_id": "qwen2:32b",
      "capability_tags": ["summarize", "extract", "reason", "code"],
      "supported_intents": ["lookup", "summarize", "extract", "reason"],
      "trust_tier": "standard",
      "context_window": 32768,
      "max_output_tokens": 8192,
      "timeouts": {"soft_ms": 30000, "hard_ms": 120000},
      "enabled": true
    },
    {
      "alias": "mistral:7b",
      "display_name": "Mistral 7B",
      "provider": "local",
      "capability_tags": ["summarize", "draft", "conversational"],
      "supported_intents": ["lookup", "summarize", "draft"],
      "trust_tier": "standard",
      "context_window": 8192,
      "timeouts": {"soft_ms": 15000, "hard_ms": 60000},
      "enabled": true
    },
    {
      "alias": "judge:trusted",
      "display_name": "Trusted Judge Model",
      "description": "High-trust model for evaluation tasks",
      "provider": "api_hub",
      "provider_model_id": "claude-3-opus",
      "capability_tags": ["judge", "critique", "reason"],
      "supported_intents": ["critique", "reason", "compare"],
      "trust_tier": "trusted",
      "context_window": 200000,
      "max_output_tokens": 4096,
      "timeouts": {"soft_ms": 60000, "hard_ms": 300000},
      "cost_hints": {
        "input_per_1k_tokens": 0.015,
        "output_per_1k_tokens": 0.075
      },
      "enabled": true
    },
    {
      "alias": "experimental:llama3",
      "display_name": "Experimental Llama 3",
      "provider": "local",
      "capability_tags": ["draft", "creative"],
      "trust_tier": "untrusted",
      "context_window": 8192,
      "enabled": true,
      "deprecated": false
    }
  ]
}
"""
