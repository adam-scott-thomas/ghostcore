"""
Observability: Metrics, Tracing, and Structured Logging

Provides:
- Metrics collection (counters, histograms, gauges)
- Trace context propagation (trace_id, span_id)
- Structured log correlation

This module uses a pluggable backend pattern:
- Default: In-memory metrics for testing
- Production: Prometheus, StatsD, or OpenTelemetry
"""

from __future__ import annotations

import contextvars
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog


# --- Trace Context ---

@dataclass
class TraceContext:
    """Distributed tracing context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)

    @classmethod
    def new_trace(cls) -> "TraceContext":
        """Create a new trace context."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )

    def new_span(self) -> "TraceContext":
        """Create a child span in this trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
        )

    def elapsed_ms(self) -> float:
        """Get elapsed time since span start."""
        return (time.monotonic() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


# Context variable for current trace
_current_trace: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    "current_trace", default=None
)


def get_current_trace() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _current_trace.get()


def get_or_create_trace() -> TraceContext:
    """Get current trace or create new one."""
    trace = _current_trace.get()
    if trace is None:
        trace = TraceContext.new_trace()
        _current_trace.set(trace)
    return trace


@contextmanager
def trace_span(name: str, **attributes):
    """
    Context manager for creating a traced span.

    Usage:
        with trace_span("execute_adapter", model="gpt-4") as span:
            # do work
            span.baggage["result"] = "success"
    """
    parent = get_current_trace()
    if parent:
        span = parent.new_span()
    else:
        span = TraceContext.new_trace()

    span.baggage["span_name"] = name
    span.baggage.update(attributes)

    token = _current_trace.set(span)
    try:
        yield span
    finally:
        _current_trace.reset(token)


def with_trace_id(trace_id: str):
    """Set a specific trace_id (e.g., from incoming request)."""
    trace = TraceContext(
        trace_id=trace_id,
        span_id=uuid.uuid4().hex[:16],
    )
    return _current_trace.set(trace)


# --- Metrics Interface ---

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """A single metric observation."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.COUNTER


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        pass

    @abstractmethod
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        pass

    @abstractmethod
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        pass

    @abstractmethod
    def get_metrics(self) -> List[MetricValue]:
        """Get all recorded metrics (for testing/debugging)."""
        pass


class InMemoryMetrics(MetricsBackend):
    """In-memory metrics backend for testing."""

    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[Tuple[str, tuple], float] = defaultdict(float)
        self._gauges: Dict[Tuple[str, tuple], float] = {}
        self._histograms: Dict[Tuple[str, tuple], List[float]] = defaultdict(list)

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = (name, self._labels_key(labels))
            self._counters[key] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = (name, self._labels_key(labels))
            self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = (name, self._labels_key(labels))
            self._histograms[key].append(value)

    def get_metrics(self) -> List[MetricValue]:
        with self._lock:
            result = []
            for (name, labels_tuple), value in self._counters.items():
                result.append(MetricValue(
                    name=name,
                    value=value,
                    labels=dict(labels_tuple),
                    metric_type=MetricType.COUNTER,
                ))
            for (name, labels_tuple), value in self._gauges.items():
                result.append(MetricValue(
                    name=name,
                    value=value,
                    labels=dict(labels_tuple),
                    metric_type=MetricType.GAUGE,
                ))
            for (name, labels_tuple), values in self._histograms.items():
                for v in values:
                    result.append(MetricValue(
                        name=name,
                        value=v,
                        labels=dict(labels_tuple),
                        metric_type=MetricType.HISTOGRAM,
                    ))
            return result

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value (for testing)."""
        with self._lock:
            key = (name, self._labels_key(labels))
            return self._counters.get(key, 0.0)

    def get_histogram_values(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[float]:
        """Get histogram values (for testing)."""
        with self._lock:
            key = (name, self._labels_key(labels))
            return list(self._histograms.get(key, []))

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global metrics instance
_metrics: MetricsBackend = InMemoryMetrics()


def get_metrics() -> MetricsBackend:
    """Get the global metrics backend."""
    return _metrics


def set_metrics(backend: MetricsBackend) -> None:
    """Set the global metrics backend."""
    global _metrics
    _metrics = backend


# --- Metric Names (Constants) ---

class Metrics:
    """Standard metric names for the ControlCore system."""
    # Execution metrics
    CALLS_TOTAL = "ControlCore_calls_total"
    CALLS_DURATION_MS = "ControlCore_calls_duration_ms"
    CALLS_SUCCESS = "ControlCore_calls_success_total"
    CALLS_FAILED = "ControlCore_calls_failed_total"
    CALLS_QUEUED = "ControlCore_calls_queued_total"

    # Adapter metrics
    ADAPTER_CALLS_TOTAL = "ControlCore_adapter_calls_total"
    ADAPTER_DURATION_MS = "ControlCore_adapter_duration_ms"
    ADAPTER_ERRORS = "ControlCore_adapter_errors_total"
    ADAPTER_TIMEOUTS = "ControlCore_adapter_timeouts_total"
    ADAPTER_REFUSALS = "ControlCore_adapter_refusals_total"

    # Routing metrics
    ROUTING_ATTEMPTS = "ControlCore_routing_attempts_total"
    MODELS_TRIED = "ControlCore_models_tried_total"

    # Circuit breaker metrics
    CIRCUIT_OPEN = "ControlCore_circuit_open"
    CIRCUIT_HALF_OPEN_ATTEMPTS = "ControlCore_circuit_half_open_attempts_total"

    # Resource metrics
    ACTIVE_EXECUTIONS = "ControlCore_active_executions"


# --- Convenience Functions ---

def record_call_start(call_id: str) -> None:
    """Record a call starting."""
    _metrics.increment(Metrics.CALLS_TOTAL)
    _metrics.gauge(Metrics.ACTIVE_EXECUTIONS, 1, {"call_id": call_id})


def record_call_end(call_id: str, status: str, duration_ms: float) -> None:
    """Record a call completing."""
    labels = {"status": status}
    _metrics.histogram(Metrics.CALLS_DURATION_MS, duration_ms, labels)

    if status == "complete":
        _metrics.increment(Metrics.CALLS_SUCCESS)
    elif status == "failed":
        _metrics.increment(Metrics.CALLS_FAILED)
    elif status == "queued":
        _metrics.increment(Metrics.CALLS_QUEUED)


def record_adapter_call(
    adapter_name: str,
    model_alias: str,
    status: str,
    duration_ms: float,
) -> None:
    """Record an adapter execution."""
    labels = {"adapter": adapter_name, "model": model_alias, "status": status}
    _metrics.increment(Metrics.ADAPTER_CALLS_TOTAL, labels=labels)
    _metrics.histogram(Metrics.ADAPTER_DURATION_MS, duration_ms, labels)

    if status == "error":
        _metrics.increment(Metrics.ADAPTER_ERRORS, labels={"adapter": adapter_name})
    elif status == "timeout":
        _metrics.increment(Metrics.ADAPTER_TIMEOUTS, labels={"adapter": adapter_name})
    elif status == "refused":
        _metrics.increment(Metrics.ADAPTER_REFUSALS, labels={"adapter": adapter_name})


def record_routing_attempt(models_tried: int) -> None:
    """Record a routing attempt."""
    _metrics.increment(Metrics.ROUTING_ATTEMPTS)
    _metrics.histogram(Metrics.MODELS_TRIED, models_tried)


# --- Structured Logging with Trace Context ---

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger that automatically includes trace context."""
    return structlog.get_logger(name)


def bind_trace_context(logger: structlog.BoundLogger) -> structlog.BoundLogger:
    """Bind current trace context to logger."""
    trace = get_current_trace()
    if trace:
        return logger.bind(
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            parent_span_id=trace.parent_span_id,
        )
    return logger


class TracedLogger:
    """Logger wrapper that automatically includes trace context."""

    def __init__(self, name: str):
        self._logger = structlog.get_logger(name)

    def _with_trace(self) -> structlog.BoundLogger:
        return bind_trace_context(self._logger)

    def debug(self, msg: str, **kwargs) -> None:
        self._with_trace().debug(msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._with_trace().info(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._with_trace().warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._with_trace().error(msg, **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        self._with_trace().exception(msg, **kwargs)


# --- Timer Context Manager ---

@contextmanager
def timed_operation(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager that times an operation and records it.

    Usage:
        with timed_operation("adapter_execute", {"adapter": "cpu"}):
            # do work
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        _metrics.histogram(name, duration_ms, labels)
