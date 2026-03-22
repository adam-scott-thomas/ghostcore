"""
P3-01: Execution Adapter Interface (Contract Only)

Defines the strict contract that all execution adapters must follow.
This is a PURE INTERFACE - no execution logic, no side effects.

INVARIANTS:
1. execute() is the ONLY method that performs external work
2. Adapters receive ONLY validated ControlCoreCall objects
3. Adapters return ONLY AdapterResult objects
4. Adapters MUST NOT:
   - Write to persistent memory
   - Mutate any input state
   - Perform routing decisions
   - Retry internally
   - Access credentials beyond request scope
5. Adapters MUST:
   - Enforce timeouts strictly
   - Capture all output safely
   - Return partial results when possible
   - Include accurate provenance information
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from ControlCore.schemas import ControlCoreCall


class AdapterStatus(str, Enum):
    """Status of adapter execution."""
    success = "success"           # Completed successfully
    timeout = "timeout"           # Hard timeout exceeded
    soft_timeout = "soft_timeout" # Soft timeout exceeded, partial result
    error = "error"               # Adapter-level error
    refused = "refused"           # Model refused the request
    rate_limited = "rate_limited" # Rate limited by provider
    content_filtered = "content_filtered"  # Blocked by content filter


class AdapterConfig(BaseModel):
    """
    Read-only configuration for an adapter instance.

    Adapters receive this at construction time and MUST NOT modify it.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_name: str = Field(..., description="Unique adapter identifier")
    adapter_version: str = Field("1.0.0", description="Adapter implementation version")

    # Timeout overrides (if not set, use call-level or model-level)
    default_soft_timeout_ms: Optional[int] = Field(None, ge=1000, le=600000)
    default_hard_timeout_ms: Optional[int] = Field(None, ge=1000, le=600000)

    # Size limits
    max_input_bytes: int = Field(10_000_000, ge=1, le=100_000_000)
    max_output_bytes: int = Field(10_000_000, ge=1, le=100_000_000)

    # Provider-specific config (opaque to interface)
    provider_config: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AdapterTiming:
    """Timing information from adapter execution."""
    start_time: datetime
    end_time: datetime
    total_ms: int
    queue_ms: int = 0          # Time waiting before execution
    execution_ms: int = 0      # Actual execution time
    network_ms: int = 0        # Network latency (for remote adapters)

    @classmethod
    def create(cls, start: datetime, end: datetime, **kwargs) -> "AdapterTiming":
        """Create timing with calculated total."""
        total_ms = int((end - start).total_seconds() * 1000)
        return cls(
            start_time=start,
            end_time=end,
            total_ms=total_ms,
            **kwargs,
        )


@dataclass
class AdapterProvenance:
    """
    Provenance information from adapter execution.

    This is attached to results for auditability.
    """
    adapter_name: str
    adapter_version: str
    model_alias: str
    provider_model_id: Optional[str] = None
    timing: Optional[AdapterTiming] = None

    # Token usage (if available)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "model_alias": self.model_alias,
        }
        if self.provider_model_id:
            result["provider_model_id"] = self.provider_model_id
        if self.timing:
            result["timing"] = {
                "start_time": self.timing.start_time.isoformat(),
                "end_time": self.timing.end_time.isoformat(),
                "total_ms": self.timing.total_ms,
                "queue_ms": self.timing.queue_ms,
                "execution_ms": self.timing.execution_ms,
                "network_ms": self.timing.network_ms,
            }
        if self.input_tokens is not None:
            result["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            result["output_tokens"] = self.output_tokens
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class AdapterResult:
    """
    Normalized result from adapter execution.

    This is the ONLY return type from execute().
    All adapters must normalize their output to this format.
    """
    status: AdapterStatus

    # The raw response content (may be partial on timeout)
    content: Optional[str] = None

    # Structured response (if adapter supports it)
    structured: Optional[Dict[str, Any]] = None

    # Error information (if status is error)
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Refusal information (if status is refused)
    refusal_reason: Optional[str] = None

    # Whether this is a partial result
    is_partial: bool = False

    # Provenance for audit trail
    provenance: Optional[AdapterProvenance] = None

    # Model's confidence (if reported)
    model_confidence: Optional[float] = None

    # Raw provider response (for debugging, may be redacted)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == AdapterStatus.success

    @property
    def is_retriable(self) -> bool:
        """Check if this result suggests retry might help."""
        return self.status in (
            AdapterStatus.timeout,
            AdapterStatus.soft_timeout,
            AdapterStatus.rate_limited,
        )

    @property
    def should_switch_model(self) -> bool:
        """Check if this result suggests switching to another model."""
        return self.status in (
            AdapterStatus.refused,
            AdapterStatus.content_filtered,
            AdapterStatus.error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "status": self.status.value,
            "is_partial": self.is_partial,
        }
        if self.content is not None:
            result["content"] = self.content
        if self.structured is not None:
            result["structured"] = self.structured
        if self.error_message:
            result["error_message"] = self.error_message
        if self.error_code:
            result["error_code"] = self.error_code
        if self.refusal_reason:
            result["refusal_reason"] = self.refusal_reason
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        if self.model_confidence is not None:
            result["model_confidence"] = self.model_confidence
        return result


class ExecutionAdapter(ABC):
    """
    Abstract base class for execution adapters.

    INVARIANTS:
    -----------
    1. The ONLY external work happens in execute()
    2. execute() receives a VALIDATED ControlCoreCall
    3. execute() returns an AdapterResult (never raises for model errors)
    4. Adapters are STATELESS between calls
    5. Adapters MUST NOT:
       - Write to persistent storage
       - Mutate input objects
       - Make routing decisions
       - Retry internally
       - Cache responses
    6. Adapters MUST:
       - Enforce timeout limits strictly
       - Return partial results when possible
       - Include accurate provenance
       - Redact secrets from any logged output

    LIFECYCLE:
    ----------
    1. Adapter is instantiated with AdapterConfig
    2. can_handle() checks if adapter supports the model
    3. execute() performs the actual execution
    4. Result flows to caller for normalization/redaction

    THREAD SAFETY:
    --------------
    Adapters should be safe to use from multiple threads.
    No shared mutable state should exist.
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize adapter with read-only configuration.

        Args:
            config: Read-only adapter configuration
        """
        self._config = config

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration (read-only)."""
        return self._config

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._config.adapter_name

    @property
    def version(self) -> str:
        """Get adapter version."""
        return self._config.adapter_version

    @abstractmethod
    def can_handle(self, model_alias: str) -> bool:
        """
        Check if this adapter can handle the given model.

        This is a PURE check - no external calls, no side effects.

        Args:
            model_alias: The model alias to check

        Returns:
            True if this adapter can execute the model
        """
        pass

    @abstractmethod
    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """
        Execute the call against the model.

        This is the ONLY method that performs external work.

        GUARANTEES:
        - Returns AdapterResult (never raises for model errors)
        - Respects timeout limits strictly
        - Returns partial results when possible on timeout
        - Includes accurate provenance information

        MUST NOT:
        - Retry internally
        - Perform fallback to other models
        - Write to persistent storage
        - Mutate any input

        Args:
            call: Validated ControlCoreCall to execute
            model_alias: The model alias to use
            soft_timeout_ms: Soft timeout (return partial after this)
            hard_timeout_ms: Hard timeout (abort after this)

        Returns:
            AdapterResult with normalized response and provenance
        """
        pass

    def get_effective_timeouts(
        self,
        call: ControlCoreCall,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Get effective timeout values.

        Priority: explicit args > call options > config defaults > hardcoded

        Returns:
            Tuple of (soft_timeout_ms, hard_timeout_ms)
        """
        # Hardcoded absolute limits
        MAX_SOFT = 300000   # 5 minutes
        MAX_HARD = 600000   # 10 minutes
        DEFAULT_SOFT = 30000
        DEFAULT_HARD = 120000

        # Resolve soft timeout
        effective_soft = (
            soft_timeout_ms
            or (call.options.timeouts.soft_ms if call.options.timeouts else None)
            or self._config.default_soft_timeout_ms
            or DEFAULT_SOFT
        )

        # Resolve hard timeout
        effective_hard = (
            hard_timeout_ms
            or (call.options.timeouts.hard_ms if call.options.timeouts else None)
            or self._config.default_hard_timeout_ms
            or DEFAULT_HARD
        )

        # Enforce limits
        effective_soft = min(effective_soft, MAX_SOFT)
        effective_hard = min(effective_hard, MAX_HARD)

        # Soft must be <= hard
        if effective_soft > effective_hard:
            effective_soft = effective_hard

        return effective_soft, effective_hard

    def create_provenance(
        self,
        model_alias: str,
        timing: Optional[AdapterTiming] = None,
        **kwargs,
    ) -> AdapterProvenance:
        """
        Create provenance record for this adapter.

        Helper method for subclasses.
        """
        return AdapterProvenance(
            adapter_name=self.name,
            adapter_version=self.version,
            model_alias=model_alias,
            timing=timing,
            **kwargs,
        )


# Type alias for adapter factory functions
AdapterFactory = type[ExecutionAdapter]
