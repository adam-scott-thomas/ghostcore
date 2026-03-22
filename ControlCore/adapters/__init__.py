"""Execution adapters - interface and implementations."""

from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterProvenance,
    AdapterTiming,
)
from ControlCore.adapters.cpu import (
    CPUAdapter,
    CPUAdapterConfig,
    StubCPUAdapter,
)
from ControlCore.adapters.remote import (
    RemoteAPIAdapter,
    RemoteAdapterConfig,
    CredentialProvider,
    MockRemoteAdapter,
)
from ControlCore.adapters.sandbox import (
    SandboxConfig,
    SandboxViolation,
    ViolationType,
    SandboxedAdapter,
    FilesystemGuard,
    NetworkGuard,
    TimeoutGuard,
    ResourceLimiter,
    create_sandbox_for_cpu_adapter,
    create_sandbox_for_remote_adapter,
    create_restricted_sandbox,
    validate_sandbox_config,
    merge_sandbox_configs,
)
from ControlCore.adapters.executor import (
    ExecutionEngine,
    ExecutionOutcome,
    ExecutionAttempt,
    ExecutionTrace,
    AdapterRegistry,
    execute_call,
    create_stub_adapter_registry,
)

__all__ = [
    # Interface
    "ExecutionAdapter",
    "AdapterConfig",
    "AdapterResult",
    "AdapterStatus",
    "AdapterProvenance",
    "AdapterTiming",
    # CPU Adapter
    "CPUAdapter",
    "CPUAdapterConfig",
    "StubCPUAdapter",
    # Remote Adapter
    "RemoteAPIAdapter",
    "RemoteAdapterConfig",
    "CredentialProvider",
    "MockRemoteAdapter",
    # Sandbox
    "SandboxConfig",
    "SandboxViolation",
    "ViolationType",
    "SandboxedAdapter",
    "FilesystemGuard",
    "NetworkGuard",
    "TimeoutGuard",
    "ResourceLimiter",
    "create_sandbox_for_cpu_adapter",
    "create_sandbox_for_remote_adapter",
    "create_restricted_sandbox",
    "validate_sandbox_config",
    "merge_sandbox_configs",
    # Executor
    "ExecutionEngine",
    "ExecutionOutcome",
    "ExecutionAttempt",
    "ExecutionTrace",
    "AdapterRegistry",
    "execute_call",
    "create_stub_adapter_registry",
]
