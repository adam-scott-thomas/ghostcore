"""
Circuit Breaker for Adapter Execution

Prevents cascading failures by tracking adapter health and
failing fast when adapters are known to be unhealthy.

States:
- CLOSED: Normal operation, requests go through
- OPEN: Too many failures, requests fail immediately
- HALF_OPEN: Testing if adapter has recovered

This module integrates with the orchestration layer, NOT adapters.
Adapters remain unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Dict, Optional, Callable, Any

from ControlCore.observability import get_metrics, Metrics, TracedLogger

logger = TracedLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker."""
    # Failure threshold to open circuit
    failure_threshold: int = 5

    # Time window in seconds for counting failures
    failure_window_seconds: float = 60.0

    # Time to wait before testing recovery (half-open)
    recovery_timeout_seconds: float = 30.0

    # Number of successful calls in half-open to close circuit
    success_threshold: int = 2

    # Whether to count timeouts as failures
    count_timeouts_as_failures: bool = True

    # Whether to count rate limits as failures
    count_rate_limits_as_failures: bool = False


@dataclass
class CircuitStats:
    """Statistics for a circuit."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """
    Circuit breaker for a single adapter/model combination.

    Thread-safe implementation using locks.
    """

    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (e.g., "adapter:model")
            config: Circuit configuration
        """
        self.name = name
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._lock = Lock()
        self._failure_times: list[float] = []
        self._half_open_successes = 0
        self._stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._maybe_transition()
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        with self._lock:
            return CircuitStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_changed_at=self._stats.state_changed_at,
            )

    def is_open(self) -> bool:
        """Check if circuit is open (should fail fast)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request should proceed, False if should fail fast
        """
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                return True

        return True

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info("Circuit closed after recovery", circuit=self.name)

    def record_failure(self, is_timeout: bool = False, is_rate_limit: bool = False) -> None:
        """
        Record a failed call.

        Args:
            is_timeout: Whether failure was due to timeout
            is_rate_limit: Whether failure was due to rate limiting
        """
        # Check if we should count this failure
        if is_timeout and not self.config.count_timeouts_as_failures:
            return
        if is_rate_limit and not self.config.count_rate_limits_as_failures:
            return

        with self._lock:
            now = time.monotonic()
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = now

            # Add to failure window
            self._failure_times.append(now)
            self._prune_old_failures(now)

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning("Circuit reopened after half-open failure", circuit=self.name)
                return

            if self._state == CircuitState.CLOSED:
                # Check if we should open
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit opened due to failures",
                        circuit=self.name,
                        failures=len(self._failure_times),
                        threshold=self.config.failure_threshold,
                    )

    def _prune_old_failures(self, now: float) -> None:
        """Remove failures outside the window."""
        cutoff = now - self.config.failure_window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]

    def _maybe_transition(self) -> None:
        """Check if we should transition state based on time."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._stats.state_changed_at
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info("Circuit entering half-open state", circuit=self.name)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changed_at = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self._failure_times.clear()
            self._half_open_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0

        # Record metric
        get_metrics().gauge(
            Metrics.CIRCUIT_OPEN,
            1.0 if new_state == CircuitState.OPEN else 0.0,
            {"circuit": self.name},
        )

    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitStats()
            logger.info("Circuit manually reset", circuit=self.name)


class CircuitBreakerRegistry:
    """
    Registry of circuit breakers.

    Provides circuit breakers for adapter+model combinations.
    """

    def __init__(self, default_config: Optional[CircuitConfig] = None):
        """
        Initialize registry.

        Args:
            default_config: Default configuration for new circuits
        """
        self._default_config = default_config or CircuitConfig()
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get_circuit(
        self,
        adapter_name: str,
        model_alias: str,
        config: Optional[CircuitConfig] = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker for an adapter+model.

        Args:
            adapter_name: Name of the adapter
            model_alias: Model alias
            config: Optional custom config for this circuit

        Returns:
            CircuitBreaker instance
        """
        key = f"{adapter_name}:{model_alias}"

        with self._lock:
            if key not in self._circuits:
                self._circuits[key] = CircuitBreaker(
                    name=key,
                    config=config or self._default_config,
                )
            return self._circuits[key]

    def get_all_circuits(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        with self._lock:
            return dict(self._circuits)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for circuit in self._circuits.values():
                circuit.reset()

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        with self._lock:
            return [
                name for name, circuit in self._circuits.items()
                if circuit.is_open()
            ]


# Global registry instance
_circuit_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _circuit_registry
    if _circuit_registry is None:
        _circuit_registry = CircuitBreakerRegistry()
    return _circuit_registry


def set_circuit_registry(registry: CircuitBreakerRegistry) -> None:
    """Set the global circuit breaker registry."""
    global _circuit_registry
    _circuit_registry = registry


class CircuitOpenError(Exception):
    """Raised when a request is rejected due to open circuit."""

    def __init__(self, circuit_name: str):
        self.circuit_name = circuit_name
        super().__init__(f"Circuit '{circuit_name}' is open - failing fast")
