"""
P3-05: Adapter Wiring (No Intelligence)

Wires adapters into the daemon execution path:
- Routing + policy selects adapter
- Adapter executes exactly once
- Results flow through normalization, provenance, confidence, redaction

This module does NOT:
- Add planning or reasoning
- Make intelligent decisions
- Modify adapter behavior
- Cache results

It simply orchestrates the execution flow.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ControlCore.schemas import (
    ControlCoreCall,
    ControlCoreCallResult,
    Provenance,
    Confidence,
    CallError,
    ErrorCode,
    CallStatus,
    RedactionReport,
)
from ControlCore.bouncer import enforce_bouncer
from ControlCore.redaction import redact_text
from ControlCore.registry.schema import ModelEntry, ModelRegistry
from ControlCore.registry.dial import filter_eligible_models, EligibilityResult
from ControlCore.registry.routing import compute_routing_order, RoutingResult, RefusalHistory
from ControlCore.registry.fallback import FallbackPolicy, default_policy, ModelSwitchCondition
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterResult,
    AdapterStatus,
    AdapterConfig,
)
from ControlCore.observability import (
    get_or_create_trace,
    trace_span,
    with_trace_id,
    TracedLogger,
    record_call_start,
    record_call_end,
    record_adapter_call,
    record_routing_attempt,
    get_metrics,
    Metrics,
)
from ControlCore.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitOpenError,
    get_circuit_registry,
)

logger = TracedLogger(__name__)


class ExecutionOutcome(str, Enum):
    """Outcome of execution attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    QUEUED = "queued"
    FAILED = "failed"
    REFUSED = "refused"
    ALL_EXHAUSTED = "all_exhausted"


@dataclass
class ExecutionAttempt:
    """Record of a single execution attempt."""
    attempt_number: int
    model_alias: str
    adapter_name: str
    result: AdapterResult
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def succeeded(self) -> bool:
        return self.result.is_success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "model_alias": self.model_alias,
            "adapter_name": self.adapter_name,
            "status": self.result.status.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionTrace:
    """Complete trace of execution for audit trail."""
    call_id: str
    attempts: List[ExecutionAttempt] = field(default_factory=list)
    routing_result: Optional[RoutingResult] = None
    eligibility_result: Optional[EligibilityResult] = None
    policy: Optional[FallbackPolicy] = None
    outcome: ExecutionOutcome = ExecutionOutcome.FAILED
    final_result: Optional[AdapterResult] = None
    queued_job_id: Optional[str] = None

    def add_attempt(self, attempt: ExecutionAttempt) -> None:
        self.attempts.append(attempt)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "outcome": self.outcome.value,
            "total_attempts": self.total_attempts,
            "attempts": [a.to_dict() for a in self.attempts],
            "queued_job_id": self.queued_job_id,
        }


class AdapterRegistry:
    """
    Registry of available adapters.

    Maps providers/models to their adapters.
    """

    def __init__(self):
        self._adapters: List[ExecutionAdapter] = []
        self._default_adapter: Optional[ExecutionAdapter] = None

    def register(self, adapter: ExecutionAdapter) -> None:
        """Register an adapter."""
        self._adapters.append(adapter)

    def set_default(self, adapter: ExecutionAdapter) -> None:
        """Set the default adapter."""
        self._default_adapter = adapter

    def get_adapter_for_model(self, model_alias: str) -> Optional[ExecutionAdapter]:
        """Get adapter that can handle the model."""
        for adapter in self._adapters:
            if adapter.can_handle(model_alias):
                return adapter
        return self._default_adapter

    def list_adapters(self) -> List[str]:
        """List registered adapter names."""
        return [a.name for a in self._adapters]


class ExecutionEngine:
    """
    Main execution engine that wires everything together.

    Responsibilities:
    1. Accept validated ControlCoreCall
    2. Check eligibility via dial module
    3. Compute routing order
    4. Select adapter for top model
    5. Execute ONCE (no internal retry)
    6. Apply redaction
    7. Build result with provenance

    This class does NOT:
    - Perform retries (that's the caller's job using fallback policy)
    - Make intelligent decisions
    - Cache anything
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        adapter_registry: AdapterRegistry,
        refusal_history: Optional[RefusalHistory] = None,
        circuit_registry: Optional[CircuitBreakerRegistry] = None,
    ):
        self._model_registry = model_registry
        self._adapter_registry = adapter_registry
        self._refusal_history = refusal_history
        self._circuit_registry = circuit_registry or get_circuit_registry()

    async def execute_once(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
        skip_circuit_breaker: bool = False,
    ) -> Tuple[AdapterResult, Optional[ExecutionAdapter]]:
        """
        Execute call against a single model exactly once.

        Args:
            call: The validated call
            model_alias: Model to execute against
            soft_timeout_ms: Soft timeout
            hard_timeout_ms: Hard timeout
            skip_circuit_breaker: If True, bypass circuit breaker check

        Returns:
            Tuple of (AdapterResult, adapter used)
        """
        # Get adapter
        adapter = self._adapter_registry.get_adapter_for_model(model_alias)
        if not adapter:
            logger.warning("No adapter for model", model_alias=model_alias, call_id=call.call_id)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"No adapter available for model: {model_alias}",
                error_code="NO_ADAPTER",
            ), None

        # Check circuit breaker
        circuit = self._circuit_registry.get_circuit(adapter.name, model_alias)
        if not skip_circuit_breaker and not circuit.allow_request():
            logger.warning(
                "Circuit breaker open - failing fast",
                adapter=adapter.name,
                model=model_alias,
                call_id=call.call_id,
            )
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Circuit breaker open for {adapter.name}:{model_alias}",
                error_code="CIRCUIT_OPEN",
            ), adapter

        # Execute with tracing and metrics
        start_time = time.monotonic()
        with trace_span("adapter_execute", adapter=adapter.name, model=model_alias):
            logger.debug("Executing adapter", adapter=adapter.name, model=model_alias)

            result = await adapter.execute(
                call,
                model_alias,
                soft_timeout_ms=soft_timeout_ms,
                hard_timeout_ms=hard_timeout_ms,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            record_adapter_call(
                adapter_name=adapter.name,
                model_alias=model_alias,
                status=result.status.value,
                duration_ms=duration_ms,
            )

            # Update circuit breaker
            if result.is_success:
                circuit.record_success()
            else:
                circuit.record_failure(
                    is_timeout=(result.status == AdapterStatus.timeout),
                    is_rate_limit=(result.status == AdapterStatus.rate_limited),
                )

            logger.debug(
                "Adapter execution complete",
                adapter=adapter.name,
                model=model_alias,
                status=result.status.value,
                duration_ms=duration_ms,
            )

        return result, adapter

    async def execute_with_fallback(
        self,
        call: ControlCoreCall,
        policy: Optional[FallbackPolicy] = None,
        enqueue_callback: Optional[callable] = None,
    ) -> Tuple[ControlCoreCallResult, ExecutionTrace]:
        """
        Execute call with fallback policy.

        This method:
        1. Filters eligible models
        2. Computes routing order
        3. Tries models in order according to policy
        4. Returns result with full trace

        Args:
            call: Validated ControlCoreCall
            policy: Fallback policy (defaults to default_policy)
            enqueue_callback: Called if job should be queued, returns job_id

        Returns:
            Tuple of (ControlCoreCallResult, ExecutionTrace)
        """
        # Set up trace context using call_id
        with_trace_id(call.call_id)
        start_time = time.monotonic()

        policy = policy or default_policy()
        trace = ExecutionTrace(
            call_id=call.call_id,
            policy=policy,
        )

        # Record call start
        record_call_start(call.call_id)
        logger.info("Starting call execution", call_id=call.call_id, intent=call.intent.cls.value)

        # Check bouncer rules
        ok, bouncer_errors = enforce_bouncer(call)
        if not ok:
            duration_ms = (time.monotonic() - start_time) * 1000
            record_call_end(call.call_id, "failed", duration_ms)
            logger.warning("Call failed - bouncer violation", call_id=call.call_id, error=bouncer_errors[0].message)
            return self._build_error_result(
                call,
                error_message=f"Bouncer violation: {bouncer_errors[0].message}",
                error_code="BOUNCER_VIOLATION",
            ), trace

        # Filter eligible models
        eligibility = filter_eligible_models(call, self._model_registry)
        trace.eligibility_result = eligibility

        if not eligibility.has_eligible:
            duration_ms = (time.monotonic() - start_time) * 1000
            record_call_end(call.call_id, "failed", duration_ms)
            logger.warning("Call failed - no eligible models", call_id=call.call_id)
            return self._build_error_result(
                call,
                error_message="No eligible models found",
                error_code="NO_ELIGIBLE_MODELS",
            ), trace

        # Compute routing order
        routing = compute_routing_order(
            call,
            eligibility.eligible,
            refusal_history=self._refusal_history,
        )
        trace.routing_result = routing

        # Execute with policy
        attempt_number = 0
        models_tried: set[str] = set()
        current_model_retries = 0
        last_model_alias: Optional[str] = None

        for ranked_model in routing.ordered:
            # Check total attempts limit
            if attempt_number >= policy.max_total_attempts:
                break

            # Check models tried limit
            if len(models_tried) >= policy.model_switch.max_models_to_try:
                break

            model_alias = ranked_model.alias

            # Check same-model retry limit
            if model_alias == last_model_alias:
                current_model_retries += 1
                if current_model_retries > policy.max_same_model_retries:
                    continue
            else:
                current_model_retries = 0
                last_model_alias = model_alias

            models_tried.add(model_alias)
            attempt_number += 1

            # Execute
            result, adapter = await self.execute_once(
                call,
                model_alias,
                soft_timeout_ms=call.options.timeouts.soft_ms if call.options.timeouts else None,
                hard_timeout_ms=call.options.timeouts.hard_ms if call.options.timeouts else None,
            )

            # Record attempt
            attempt = ExecutionAttempt(
                attempt_number=attempt_number,
                model_alias=model_alias,
                adapter_name=adapter.name if adapter else "none",
                result=result,
            )
            trace.add_attempt(attempt)

            # Check result
            if result.is_success:
                trace.outcome = ExecutionOutcome.SUCCESS
                trace.final_result = result
                duration_ms = (time.monotonic() - start_time) * 1000
                record_call_end(call.call_id, "complete", duration_ms)
                record_routing_attempt(len(models_tried))
                logger.info("Call completed successfully", call_id=call.call_id, attempts=attempt_number, duration_ms=duration_ms)
                return self._build_success_result(call, result, trace), trace

            if result.status == AdapterStatus.soft_timeout and result.content:
                # Partial result
                trace.outcome = ExecutionOutcome.PARTIAL
                trace.final_result = result
                duration_ms = (time.monotonic() - start_time) * 1000
                record_call_end(call.call_id, "partial", duration_ms)
                record_routing_attempt(len(models_tried))
                logger.info("Call completed with partial result", call_id=call.call_id, attempts=attempt_number, duration_ms=duration_ms)
                return self._build_success_result(call, result, trace, is_partial=True), trace

            # Check if we should switch models
            if not self._should_switch_model(result.status, policy):
                # Stay on this model (will retry if retries available)
                continue

            # Move to next model
            last_model_alias = None  # Reset retry counter for new model

        # All attempts exhausted
        record_routing_attempt(len(models_tried))

        # Check if we should queue
        if policy.queue_escalation.enabled and enqueue_callback:
            job_id = await enqueue_callback(call)
            trace.outcome = ExecutionOutcome.QUEUED
            trace.queued_job_id = job_id
            duration_ms = (time.monotonic() - start_time) * 1000
            record_call_end(call.call_id, "queued", duration_ms)
            logger.info("Call queued for async processing", call_id=call.call_id, job_id=job_id, attempts=attempt_number)
            return self._build_queued_result(call, job_id, trace), trace

        # Final failure
        trace.outcome = ExecutionOutcome.ALL_EXHAUSTED
        last_attempt = trace.attempts[-1] if trace.attempts else None
        duration_ms = (time.monotonic() - start_time) * 1000
        record_call_end(call.call_id, "failed", duration_ms)
        logger.warning("Call failed - all models exhausted", call_id=call.call_id, attempts=attempt_number, models_tried=len(models_tried))
        return self._build_error_result(
            call,
            error_message=f"All {len(models_tried)} models exhausted after {attempt_number} attempts",
            error_code="ALL_MODELS_EXHAUSTED",
            last_result=last_attempt.result if last_attempt else None,
        ), trace

    def _should_switch_model(
        self,
        status: AdapterStatus,
        policy: FallbackPolicy,
    ) -> bool:
        """Check if the status indicates we should switch models."""
        if not policy.model_switch.enabled:
            return False

        status_to_condition = {
            AdapterStatus.refused: ModelSwitchCondition.refusal,
            AdapterStatus.timeout: ModelSwitchCondition.timeout,
            AdapterStatus.error: ModelSwitchCondition.error,
            AdapterStatus.rate_limited: ModelSwitchCondition.rate_limit,
            AdapterStatus.content_filtered: ModelSwitchCondition.content_filter,
        }

        condition = status_to_condition.get(status)
        if condition and condition in policy.model_switch.conditions:
            return True

        return False

    def _build_success_result(
        self,
        call: ControlCoreCall,
        adapter_result: AdapterResult,
        trace: ExecutionTrace,
        is_partial: bool = False,
    ) -> ControlCoreCallResult:
        """Build successful result with provenance and redaction."""
        # POST-MODEL REDACTION: Applied to model output, not prompt.
        # See redaction.py docstring for rationale.
        content = adapter_result.content or ""
        redaction_report = None

        if call.options.redaction.mode != "off":
            content, redaction_report = redact_text(content)

        # Build provenance
        adapter_prov = adapter_result.provenance
        provenance = Provenance(
            model_alias=adapter_prov.model_alias if adapter_prov else "unknown",
            started_at=adapter_prov.timing.start_time.isoformat() if adapter_prov and adapter_prov.timing else datetime.utcnow().isoformat(),
            finished_at=adapter_prov.timing.end_time.isoformat() if adapter_prov and adapter_prov.timing else datetime.utcnow().isoformat(),
            adapter=adapter_prov.adapter_name if adapter_prov else None,
            adapter_version=adapter_prov.adapter_version if adapter_prov else None,
        )

        # Build confidence
        confidence = Confidence()
        if adapter_result.model_confidence is not None:
            confidence = Confidence(self_reported=adapter_result.model_confidence)

        return ControlCoreCallResult(
            call_id=call.call_id,
            status=CallStatus.complete,
            partial=is_partial,
            answer=content,
            confidence=confidence,
            provenance=provenance,
            redaction=redaction_report or RedactionReport(),
        )

    def _build_error_result(
        self,
        call: ControlCoreCall,
        error_message: str,
        error_code: str,
        last_result: Optional[AdapterResult] = None,
    ) -> ControlCoreCallResult:
        """Build error result."""
        now = datetime.utcnow().isoformat()
        provenance = Provenance(
            model_alias="none",
            started_at=now,
            finished_at=now,
        )

        # Map error code to ErrorCode enum
        error_code_enum = ErrorCode.unknown
        if "ADAPTER" in error_code:
            error_code_enum = ErrorCode.adapter_error
        elif "TIMEOUT" in error_code:
            error_code_enum = ErrorCode.timeout
        elif "REFUSED" in error_code:
            error_code_enum = ErrorCode.refused
        elif "VALIDATION" in error_code or "BOUNCER" in error_code:
            error_code_enum = ErrorCode.validation_error
        elif "PERMISSION" in error_code:
            error_code_enum = ErrorCode.permission_denied

        return ControlCoreCallResult(
            call_id=call.call_id,
            status=CallStatus.failed,
            provenance=provenance,
            errors=[CallError(code=error_code_enum, message=error_message)],
        )

    def _build_queued_result(
        self,
        call: ControlCoreCall,
        job_id: str,
        trace: ExecutionTrace,
    ) -> ControlCoreCallResult:
        """Build queued result."""
        now = datetime.utcnow().isoformat()
        provenance = Provenance(
            model_alias="pending",
            started_at=now,
        )

        return ControlCoreCallResult(
            call_id=call.call_id,
            status=CallStatus.queued,
            job_id=job_id,
            provenance=provenance,
        )


# --- Integration with Daemon ---

async def execute_call(
    call: ControlCoreCall,
    model_registry: ModelRegistry,
    adapter_registry: AdapterRegistry,
    policy: Optional[FallbackPolicy] = None,
    refusal_history: Optional[RefusalHistory] = None,
    enqueue_callback: Optional[callable] = None,
) -> Tuple[ControlCoreCallResult, ExecutionTrace]:
    """
    Main entry point for executing a call.

    This is the function the daemon calls.
    """
    engine = ExecutionEngine(
        model_registry=model_registry,
        adapter_registry=adapter_registry,
        refusal_history=refusal_history,
    )

    return await engine.execute_with_fallback(
        call,
        policy=policy,
        enqueue_callback=enqueue_callback,
    )


def create_stub_adapter_registry() -> AdapterRegistry:
    """
    Create an adapter registry with stub adapters for testing.
    """
    from ControlCore.adapters.cpu import StubCPUAdapter, CPUAdapterConfig

    registry = AdapterRegistry()

    # Create stub adapter that handles all models
    config = CPUAdapterConfig(
        adapter_name="stub_cpu",
        adapter_version="1.0.0",
        allowed_entrypoints={},  # Empty - will use stub responses
    )

    class UniversalStubAdapter(StubCPUAdapter):
        """Stub adapter that handles any model."""
        def can_handle(self, model_alias: str) -> bool:
            return True

    stub = UniversalStubAdapter(
        config=config,
        stub_responses={},
        stub_delay_ms=50,
    )

    registry.register(stub)
    registry.set_default(stub)

    return registry
