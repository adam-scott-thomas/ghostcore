"""In-memory job registry for tracking ControlCore calls."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ControlCore.schemas import (
    ControlCoreCall,
    ControlCoreCallResult,
    CallStatus,
    CallError,
    ErrorCode,
    Provenance,
    TrustTier,
    RedactionReport,
    NormalizationReport,
    Confidence,
)


class JobEntry:
    """Internal job entry with full state."""

    def __init__(self, call: ControlCoreCall):
        self.job_id: str = str(uuid4())
        self.call: ControlCoreCall = call
        self.status: CallStatus = CallStatus.queued
        self.created_at: datetime = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.answer: Optional[str] = None
        self.errors: List[CallError] = []
        self.redaction_report: RedactionReport = RedactionReport()
        self.normalization_report: NormalizationReport = NormalizationReport()

    def to_result(self) -> ControlCoreCallResult:
        """Convert to full ControlCoreCallResult."""
        now = datetime.utcnow()

        provenance = Provenance(
            model_alias=self.call.target.alias,
            trust_tier=self.call.target.trust_tier,
            started_at=self.started_at.isoformat() + "Z" if self.started_at else now.isoformat() + "Z",
            finished_at=self.completed_at.isoformat() + "Z" if self.completed_at else None,
        )

        return ControlCoreCallResult(
            call_id=self.call.call_id,
            status=self.status,
            job_id=self.job_id if self.status in (CallStatus.queued, CallStatus.running) else None,
            answer=self.answer,
            provenance=provenance,
            redaction=self.redaction_report,
            normalization=self.normalization_report,
            errors=self.errors,
            retryable=self.status == CallStatus.failed and any(e.code == ErrorCode.timeout for e in self.errors),
        )


class JobRegistry:
    """Thread-safe in-memory job registry."""

    def __init__(self, max_jobs: int = 10000):
        self._jobs: Dict[str, JobEntry] = {}
        self._job_order: List[str] = []
        self._lock = threading.RLock()
        self._max_jobs = max_jobs

    def create_job(self, call: ControlCoreCall) -> JobEntry:
        """Create a new job for a call."""
        with self._lock:
            while len(self._jobs) >= self._max_jobs and self._job_order:
                oldest_id = self._job_order.pop(0)
                self._jobs.pop(oldest_id, None)

            entry = JobEntry(call)
            self._jobs[entry.job_id] = entry
            self._job_order.append(entry.job_id)
            return entry

    def get_job(self, job_id: str) -> Optional[JobEntry]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_result(self, job_id: str) -> Optional[ControlCoreCallResult]:
        """Get full result for a job."""
        entry = self.get_job(job_id)
        return entry.to_result() if entry else None

    def mark_running(self, job_id: str) -> bool:
        """Mark a job as running."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry and entry.status == CallStatus.queued:
                entry.status = CallStatus.running
                entry.started_at = datetime.utcnow()
                return True
            return False

    def mark_complete(
        self,
        job_id: str,
        answer: str,
        redaction_report: Optional[RedactionReport] = None,
        normalization_report: Optional[NormalizationReport] = None,
    ) -> bool:
        """Mark a job as complete with result."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry and entry.status in (CallStatus.queued, CallStatus.running):
                entry.status = CallStatus.complete
                entry.completed_at = datetime.utcnow()
                entry.answer = answer
                if redaction_report:
                    entry.redaction_report = redaction_report
                if normalization_report:
                    entry.normalization_report = normalization_report
                return True
            return False

    def mark_failed(self, job_id: str, errors: List[CallError]) -> bool:
        """Mark a job as failed with errors."""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry and entry.status in (CallStatus.queued, CallStatus.running):
                entry.status = CallStatus.failed
                entry.completed_at = datetime.utcnow()
                entry.errors = errors
                return True
            return False

    def list_jobs(
        self,
        status: Optional[CallStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())

            if status is not None:
                jobs = [j for j in jobs if j.status == status]

            jobs.sort(key=lambda j: j.created_at, reverse=True)

            return [
                {
                    "job_id": j.job_id,
                    "call_id": j.call.call_id,
                    "status": j.status.value,
                    "created_at": j.created_at.isoformat() + "Z",
                }
                for j in jobs[:limit]
            ]

    def clear(self) -> int:
        """Clear all jobs. Returns count cleared."""
        with self._lock:
            count = len(self._jobs)
            self._jobs.clear()
            self._job_order.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            status_counts = {s: 0 for s in CallStatus}
            for entry in self._jobs.values():
                status_counts[entry.status] += 1

            return {
                "total_jobs": len(self._jobs),
                "max_jobs": self._max_jobs,
                "by_status": {s.value: c for s, c in status_counts.items()},
            }


_global_registry: Optional[JobRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    """Get or create the global job registry."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = JobRegistry()
        return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    with _registry_lock:
        _global_registry = None
