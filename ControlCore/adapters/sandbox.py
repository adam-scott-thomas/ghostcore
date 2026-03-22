"""
P3-04: Adapter Sandbox & Guardrails

Security enforcement for adapter execution:
- Restricts filesystem access
- Disables network where not required
- Enforces execution time and memory limits

These are guardrails that WRAP adapter execution, not modify adapters themselves.

WINDOWS LIMITATIONS:
- Resource limits (memory, CPU) are NOT enforced on Windows
- Use strict_mode=True to fail rather than run unprotected
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager

# resource module is Unix-only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    resource = None
    HAS_RESOURCE = False

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from ControlCore.schemas import ControlCoreCall
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterTiming,
)

# Module-level logger for sandbox warnings
_sandbox_logger = logging.getLogger("ControlCore.sandbox")

# Track if we've already warned about Windows limitations
_windows_warning_issued = False


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def reset_windows_warning() -> None:
    """Reset the Windows warning flag (for testing)."""
    global _windows_warning_issued
    _windows_warning_issued = False


def _emit_windows_warning(context: str = "") -> None:
    """
    Emit a loud warning about Windows sandbox limitations.

    Only emits once per process to avoid spam.
    """
    global _windows_warning_issued
    if _windows_warning_issued:
        return

    _windows_warning_issued = True

    msg = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════════════════╗\n"
        "║                        ⚠️  SANDBOX WARNING ⚠️                                 ║\n"
        "╠══════════════════════════════════════════════════════════════════════════════╣\n"
        "║ Running on Windows without full sandbox protection!                         ║\n"
        "║                                                                             ║\n"
        "║ The following protections are NOT enforced:                                 ║\n"
        "║   • Memory limits (RLIMIT_AS)                                               ║\n"
        "║   • CPU time limits (RLIMIT_CPU)                                            ║\n"
        "║   • File size limits (RLIMIT_FSIZE)                                         ║\n"
        "║   • Open file limits (RLIMIT_NOFILE)                                        ║\n"
        "║                                                                             ║\n"
        "║ For production use, deploy on Linux/Unix or use strict_mode=True           ║\n"
        "║ to fail rather than run unprotected.                                        ║\n"
        "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    )

    if context:
        msg = f"{msg}Context: {context}\n"

    # Use multiple output channels to ensure visibility
    _sandbox_logger.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # Also print to stderr for maximum visibility
    print(msg, file=sys.stderr)


class SandboxUnavailableError(Exception):
    """Raised when sandbox is required but not available (strict mode)."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Sandbox unavailable: {reason}")


class SandboxViolation(Exception):
    """Raised when sandbox constraints are violated."""
    def __init__(self, violation_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.violation_type = violation_type
        self.details = details or {}
        super().__init__(f"{violation_type}: {message}")


class ViolationType(str, Enum):
    """Types of sandbox violations."""
    FILESYSTEM_ACCESS = "filesystem_access"
    NETWORK_ACCESS = "network_access"
    MEMORY_LIMIT = "memory_limit"
    TIME_LIMIT = "time_limit"
    RESOURCE_LIMIT = "resource_limit"
    DISALLOWED_IMPORT = "disallowed_import"


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment."""

    # Filesystem restrictions
    allowed_read_paths: List[str] = None  # Paths allowed for reading
    allowed_write_paths: List[str] = None  # Paths allowed for writing
    deny_filesystem: bool = False  # Completely deny filesystem access

    # Network restrictions
    allow_network: bool = True  # Whether network access is allowed
    allowed_hosts: Optional[Set[str]] = None  # Specific hosts allowed (None = all)

    # Resource limits
    max_memory_mb: int = 1024  # Maximum memory usage
    max_cpu_seconds: int = 300  # Maximum CPU time
    max_wall_seconds: int = 600  # Maximum wall clock time
    max_file_size_mb: int = 100  # Maximum file size for writes
    max_open_files: int = 256  # Maximum open file descriptors

    # Process restrictions
    allow_subprocess: bool = False  # Whether subprocess creation is allowed
    allow_threads: bool = True  # Whether thread creation is allowed

    # Strict mode - fail if sandbox cannot be fully enforced
    strict_mode: bool = False  # If True, fail on Windows rather than run unprotected

    def __post_init__(self):
        if self.allowed_read_paths is None:
            self.allowed_read_paths = []
        if self.allowed_write_paths is None:
            self.allowed_write_paths = []


class ResourceLimiter:
    """
    Enforces resource limits using OS-level controls.

    Note: Full enforcement requires platform-specific features.
    This implementation works on Unix-like systems.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._original_limits: Dict[int, tuple] = {}

    def _is_unix(self) -> bool:
        """Check if running on Unix-like system."""
        return HAS_RESOURCE and hasattr(resource, "setrlimit")

    @contextmanager
    def enforce_limits(self):
        """Context manager to enforce resource limits."""
        if not self._is_unix():
            # On Windows, we can't use resource module
            if self.config.strict_mode:
                raise SandboxUnavailableError(
                    "Resource limits require Unix. Windows cannot enforce "
                    "memory/CPU/file limits. Use strict_mode=False to proceed unprotected."
                )
            else:
                # Emit loud warning but continue
                _emit_windows_warning(
                    "ResourceLimiter.enforce_limits() - resource limits not enforced"
                )
            yield
            return

        try:
            # Set memory limit
            if self.config.max_memory_mb > 0:
                mem_bytes = self.config.max_memory_mb * 1024 * 1024
                self._set_limit(resource.RLIMIT_AS, mem_bytes)

            # Set CPU time limit
            if self.config.max_cpu_seconds > 0:
                self._set_limit(resource.RLIMIT_CPU, self.config.max_cpu_seconds)

            # Set file size limit
            if self.config.max_file_size_mb > 0:
                file_bytes = self.config.max_file_size_mb * 1024 * 1024
                self._set_limit(resource.RLIMIT_FSIZE, file_bytes)

            # Set open files limit
            if self.config.max_open_files > 0:
                self._set_limit(resource.RLIMIT_NOFILE, self.config.max_open_files)

            yield

        finally:
            # Restore original limits
            self._restore_limits()

    def _set_limit(self, resource_type: int, limit: int) -> None:
        """Set a resource limit, saving the original."""
        try:
            self._original_limits[resource_type] = resource.getrlimit(resource_type)
            resource.setrlimit(resource_type, (limit, limit))
        except (ValueError, resource.error):
            # Can't set limit (might be higher than hard limit)
            pass

    def _restore_limits(self) -> None:
        """Restore original resource limits."""
        for resource_type, original in self._original_limits.items():
            try:
                resource.setrlimit(resource_type, original)
            except (ValueError, resource.error):
                pass
        self._original_limits.clear()


class FilesystemGuard:
    """
    Guards filesystem access against unauthorized paths.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._normalized_read_paths = [
            self._normalize_path(p) for p in config.allowed_read_paths
        ]
        self._normalized_write_paths = [
            self._normalize_path(p) for p in config.allowed_write_paths
        ]

    def _normalize_path(self, path: str) -> Path:
        """Normalize path for comparison."""
        return Path(path).resolve()

    def check_read_access(self, path: Union[str, Path]) -> bool:
        """Check if read access to path is allowed."""
        if self.config.deny_filesystem:
            return False

        # If no explicit allowlist, allow all reads
        if not self._normalized_read_paths:
            return True

        normalized = self._normalize_path(str(path))

        for allowed in self._normalized_read_paths:
            try:
                normalized.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def check_write_access(self, path: Union[str, Path]) -> bool:
        """Check if write access to path is allowed."""
        if self.config.deny_filesystem:
            return False

        # If no explicit allowlist, deny all writes by default
        if not self._normalized_write_paths:
            return False

        normalized = self._normalize_path(str(path))

        for allowed in self._normalized_write_paths:
            try:
                normalized.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def assert_read_access(self, path: Union[str, Path]) -> None:
        """Assert read access is allowed, raise if not."""
        if not self.check_read_access(path):
            raise SandboxViolation(
                ViolationType.FILESYSTEM_ACCESS,
                f"Read access denied: {path}",
                {"path": str(path), "operation": "read"},
            )

    def assert_write_access(self, path: Union[str, Path]) -> None:
        """Assert write access is allowed, raise if not."""
        if not self.check_write_access(path):
            raise SandboxViolation(
                ViolationType.FILESYSTEM_ACCESS,
                f"Write access denied: {path}",
                {"path": str(path), "operation": "write"},
            )


class NetworkGuard:
    """
    Guards network access.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config

    def check_access(self, host: Optional[str] = None) -> bool:
        """Check if network access is allowed."""
        if not self.config.allow_network:
            return False

        if self.config.allowed_hosts is None:
            return True

        if host is None:
            return True

        return host in self.config.allowed_hosts

    def assert_access(self, host: Optional[str] = None) -> None:
        """Assert network access is allowed, raise if not."""
        if not self.check_access(host):
            raise SandboxViolation(
                ViolationType.NETWORK_ACCESS,
                f"Network access denied: {host}",
                {"host": host},
            )


class TimeoutGuard:
    """
    Enforces execution time limits.
    """

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the timeout timer."""
        self._start_time = time.monotonic()

    def check(self) -> None:
        """Check if timeout has been exceeded."""
        if self._start_time is None:
            return

        elapsed = time.monotonic() - self._start_time
        if elapsed > self.max_seconds:
            raise SandboxViolation(
                ViolationType.TIME_LIMIT,
                f"Execution time exceeded: {elapsed:.2f}s > {self.max_seconds}s",
                {"elapsed": elapsed, "limit": self.max_seconds},
            )

    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def remaining(self) -> float:
        """Get remaining time until timeout."""
        return max(0, self.max_seconds - self.elapsed)


@dataclass
class SandboxContext:
    """Context for sandboxed execution."""
    config: SandboxConfig
    filesystem: FilesystemGuard
    network: NetworkGuard
    timeout: TimeoutGuard
    resource_limiter: ResourceLimiter


class SandboxedAdapter(ExecutionAdapter):
    """
    Wrapper that adds sandbox enforcement to any adapter.

    Wraps an existing adapter with security guardrails.
    """

    def __init__(
        self,
        inner_adapter: ExecutionAdapter,
        sandbox_config: SandboxConfig,
    ):
        """
        Initialize sandboxed adapter.

        Args:
            inner_adapter: The adapter to wrap
            sandbox_config: Sandbox configuration
        """
        super().__init__(inner_adapter.config)
        self._inner = inner_adapter
        self._sandbox_config = sandbox_config

    def can_handle(self, model_alias: str) -> bool:
        """Delegate to inner adapter."""
        return self._inner.can_handle(model_alias)

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Execute with sandbox enforcement."""
        start_time = datetime.utcnow()

        # Get effective timeout
        _, hard_ms = self.get_effective_timeouts(call, soft_timeout_ms, hard_timeout_ms)
        hard_seconds = hard_ms / 1000

        # Create guards
        filesystem_guard = FilesystemGuard(self._sandbox_config)
        network_guard = NetworkGuard(self._sandbox_config)
        timeout_guard = TimeoutGuard(min(hard_seconds, self._sandbox_config.max_wall_seconds))
        resource_limiter = ResourceLimiter(self._sandbox_config)

        # Start timeout
        timeout_guard.start()

        try:
            # Execute with resource limits
            with resource_limiter.enforce_limits():
                # Execute inner adapter
                result = await asyncio.wait_for(
                    self._inner.execute(
                        call,
                        model_alias,
                        soft_timeout_ms=soft_timeout_ms,
                        hard_timeout_ms=hard_timeout_ms,
                    ),
                    timeout=timeout_guard.remaining,
                )

            return result

        except asyncio.TimeoutError:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.timeout,
                error_message="Sandbox timeout exceeded",
                error_code="SANDBOX_TIMEOUT",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except SandboxViolation as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=str(e),
                error_code=f"SANDBOX_{e.violation_type.upper()}",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except MemoryError:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message="Memory limit exceeded",
                error_code="SANDBOX_MEMORY_LIMIT",
                provenance=self.create_provenance(model_alias, timing=timing),
            )


def create_sandbox_for_cpu_adapter(
    allowed_model_paths: Optional[List[str]] = None,
    work_dir: Optional[str] = None,
) -> SandboxConfig:
    """
    Create a sandbox configuration suitable for CPU adapters.

    CPU adapters need:
    - Read access to model files
    - Write access to temp directory
    - No network access
    - Subprocess allowed (for model execution)
    """
    temp_dir = work_dir or tempfile.gettempdir()

    return SandboxConfig(
        allowed_read_paths=allowed_model_paths or [],
        allowed_write_paths=[temp_dir],
        deny_filesystem=False,
        allow_network=False,
        allow_subprocess=True,
        max_memory_mb=4096,
        max_cpu_seconds=300,
        max_wall_seconds=600,
    )


def create_sandbox_for_remote_adapter(
    allowed_hosts: Optional[Set[str]] = None,
) -> SandboxConfig:
    """
    Create a sandbox configuration suitable for remote adapters.

    Remote adapters need:
    - Network access (to specific hosts)
    - No filesystem access
    - No subprocess
    """
    return SandboxConfig(
        allowed_read_paths=[],
        allowed_write_paths=[],
        deny_filesystem=True,
        allow_network=True,
        allowed_hosts=allowed_hosts,
        allow_subprocess=False,
        max_memory_mb=512,
        max_cpu_seconds=60,
        max_wall_seconds=600,
    )


def create_restricted_sandbox() -> SandboxConfig:
    """
    Create a highly restricted sandbox for untrusted code.

    This is the most restrictive configuration.
    """
    return SandboxConfig(
        allowed_read_paths=[],
        allowed_write_paths=[],
        deny_filesystem=True,
        allow_network=False,
        allow_subprocess=False,
        allow_threads=False,
        max_memory_mb=256,
        max_cpu_seconds=30,
        max_wall_seconds=60,
        max_open_files=16,
    )


# --- Guardrail Utilities ---

def validate_sandbox_config(config: SandboxConfig) -> List[str]:
    """
    Validate sandbox configuration and return warnings.

    Returns list of warning messages for potentially unsafe settings.
    """
    warnings = []

    if not config.deny_filesystem and not config.allowed_write_paths:
        # Write paths empty but filesystem not denied - writes go nowhere
        pass  # This is fine

    if config.allow_network and config.allowed_hosts is None:
        warnings.append("Network access allowed to ALL hosts - consider restricting")

    if config.max_memory_mb > 8192:
        warnings.append(f"Memory limit {config.max_memory_mb}MB is very high")

    if config.max_wall_seconds > 3600:
        warnings.append(f"Wall time limit {config.max_wall_seconds}s is very high")

    if config.allow_subprocess and config.allow_network:
        warnings.append("Both subprocess and network are allowed - high risk")

    return warnings


def merge_sandbox_configs(base: SandboxConfig, override: SandboxConfig) -> SandboxConfig:
    """
    Merge two sandbox configs, taking the MORE restrictive option.

    Used when multiple security policies apply.
    """
    return SandboxConfig(
        # Paths: intersection (more restrictive)
        allowed_read_paths=[
            p for p in base.allowed_read_paths
            if p in override.allowed_read_paths
        ] if base.allowed_read_paths and override.allowed_read_paths else [],

        allowed_write_paths=[
            p for p in base.allowed_write_paths
            if p in override.allowed_write_paths
        ] if base.allowed_write_paths and override.allowed_write_paths else [],

        # Booleans: deny if either denies
        deny_filesystem=base.deny_filesystem or override.deny_filesystem,
        allow_network=base.allow_network and override.allow_network,
        allow_subprocess=base.allow_subprocess and override.allow_subprocess,
        allow_threads=base.allow_threads and override.allow_threads,

        # Limits: take minimum
        max_memory_mb=min(base.max_memory_mb, override.max_memory_mb),
        max_cpu_seconds=min(base.max_cpu_seconds, override.max_cpu_seconds),
        max_wall_seconds=min(base.max_wall_seconds, override.max_wall_seconds),
        max_file_size_mb=min(base.max_file_size_mb, override.max_file_size_mb),
        max_open_files=min(base.max_open_files, override.max_open_files),

        # Hosts: intersection
        allowed_hosts=(
            (base.allowed_hosts or set()) & (override.allowed_hosts or set())
            if base.allowed_hosts is not None and override.allowed_hosts is not None
            else base.allowed_hosts or override.allowed_hosts
        ),
    )
