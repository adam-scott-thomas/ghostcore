"""
P3-02: CPU Adapter (Reference Implementation)

Executes local models via controlled subprocess or library call.
Enforces timeouts strictly and captures output safely.

This adapter:
- Runs models in a sandboxed subprocess
- Accepts only explicit entrypoints (no auto-discovery)
- Enforces strict timeout limits
- Captures stdout/stderr safely
- Returns partial results on timeout when possible

This adapter does NOT:
- Auto-discover models
- Execute arbitrary code
- Retry internally
- Cache responses
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ControlCore.schemas import ControlCoreCall
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterProvenance,
    AdapterTiming,
)


class CPUAdapterConfig(AdapterConfig):
    """Configuration for CPU adapter."""

    # Allowed model entrypoints (alias -> command/path)
    allowed_entrypoints: Dict[str, str] = {}

    # Python executable to use for Python-based models
    python_executable: str = sys.executable

    # Working directory for subprocess
    work_dir: Optional[str] = None

    # Environment variables to pass (allowlist)
    allowed_env_vars: List[str] = ["PATH", "HOME", "TEMP", "TMP"]

    # Memory limit in MB (0 = no limit)
    memory_limit_mb: int = 0

    # Whether to capture stderr
    capture_stderr: bool = True

    # Max output size before truncation
    max_output_chars: int = 1_000_000


class CPUAdapter(ExecutionAdapter):
    """
    CPU execution adapter for local models.

    Executes models via subprocess with strict sandboxing.
    """

    def __init__(self, config: CPUAdapterConfig):
        """Initialize CPU adapter."""
        super().__init__(config)
        self._cpu_config = config

    def can_handle(self, model_alias: str) -> bool:
        """
        Check if this adapter can handle the model.

        Only handles models with explicit entrypoints configured.
        """
        return model_alias in self._cpu_config.allowed_entrypoints

    def _build_env(self) -> Dict[str, str]:
        """Build sanitized environment for subprocess."""
        env = {}
        for var in self._cpu_config.allowed_env_vars:
            if var in os.environ:
                env[var] = os.environ[var]
        return env

    def _build_input_payload(self, call: ControlCoreCall) -> Dict[str, Any]:
        """Build input payload for subprocess."""
        return {
            "call_id": call.call_id,
            "prompt": call.prompt,
            "context": [
                {"role": ctx.role.value, "content": ctx.content}
                for ctx in call.context
            ],
            "intent": call.intent.cls.value,
            "params": {
                "temperature": call.params.temperature if call.params else None,
                "top_p": call.params.top_p if call.params else None,
                "seed": call.params.seed if call.params else None,
            },
        }

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """
        Execute call via subprocess.

        The subprocess receives input as JSON on stdin and writes
        output as JSON to stdout.
        """
        start_time = datetime.utcnow()

        # Validate model is allowed
        if not self.can_handle(model_alias):
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Model '{model_alias}' not in allowed entrypoints",
                error_code="MODEL_NOT_ALLOWED",
                provenance=self.create_provenance(model_alias),
            )

        # Get timeouts
        soft_ms, hard_ms = self.get_effective_timeouts(call, soft_timeout_ms, hard_timeout_ms)
        soft_seconds = soft_ms / 1000
        hard_seconds = hard_ms / 1000

        # Get entrypoint
        entrypoint = self._cpu_config.allowed_entrypoints[model_alias]

        # Build input
        input_payload = self._build_input_payload(call)
        input_json = json.dumps(input_payload)

        # Check input size
        if len(input_json.encode()) > self._cpu_config.max_input_bytes:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message="Input exceeds maximum size limit",
                error_code="INPUT_TOO_LARGE",
                provenance=self.create_provenance(model_alias),
            )

        # Execute subprocess
        try:
            result = await self._run_subprocess(
                entrypoint=entrypoint,
                input_json=input_json,
                soft_timeout=soft_seconds,
                hard_timeout=hard_seconds,
                model_alias=model_alias,
                start_time=start_time,
            )
            return result

        except Exception as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Adapter error: {str(e)}",
                error_code="ADAPTER_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

    async def _run_subprocess(
        self,
        entrypoint: str,
        input_json: str,
        soft_timeout: float,
        hard_timeout: float,
        model_alias: str,
        start_time: datetime,
    ) -> AdapterResult:
        """Run the subprocess with timeout handling."""
        # Build command
        if entrypoint.endswith(".py"):
            cmd = [self._cpu_config.python_executable, entrypoint]
        else:
            cmd = [entrypoint]

        # Build environment
        env = self._build_env()

        # Working directory
        cwd = self._cpu_config.work_dir or os.getcwd()

        # Track partial output
        partial_output = []
        soft_timeout_triggered = False
        execution_start = datetime.utcnow()

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if self._cpu_config.capture_stderr else None,
                env=env,
                cwd=cwd,
            )

            # Write input
            if process.stdin:
                process.stdin.write(input_json.encode())
                await process.stdin.drain()
                process.stdin.close()

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=hard_timeout,
                )
            except asyncio.TimeoutError:
                # Hard timeout - kill process
                process.kill()
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=1.0,  # Grace period
                    )
                except asyncio.TimeoutError:
                    stdout, stderr = b"", b""

                end_time = datetime.utcnow()
                timing = AdapterTiming.create(
                    start_time, end_time,
                    execution_ms=int((end_time - execution_start).total_seconds() * 1000),
                )

                # Try to parse partial output
                partial_content = self._try_parse_partial(stdout)

                return AdapterResult(
                    status=AdapterStatus.timeout,
                    content=partial_content,
                    is_partial=bool(partial_content),
                    error_message="Hard timeout exceeded",
                    error_code="HARD_TIMEOUT",
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            end_time = datetime.utcnow()
            execution_ms = int((end_time - execution_start).total_seconds() * 1000)
            timing = AdapterTiming.create(start_time, end_time, execution_ms=execution_ms)

            # Check for soft timeout (process completed but took too long)
            if execution_ms > soft_timeout * 1000:
                soft_timeout_triggered = True

            # Parse output
            return self._parse_subprocess_output(
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                model_alias=model_alias,
                timing=timing,
                soft_timeout_triggered=soft_timeout_triggered,
            )

        except FileNotFoundError:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Entrypoint not found: {entrypoint}",
                error_code="ENTRYPOINT_NOT_FOUND",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except PermissionError:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Permission denied for entrypoint: {entrypoint}",
                error_code="PERMISSION_DENIED",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

    def _try_parse_partial(self, stdout: bytes) -> Optional[str]:
        """Try to extract partial content from stdout."""
        if not stdout:
            return None

        try:
            decoded = stdout.decode("utf-8", errors="replace")
            # Truncate if too long
            if len(decoded) > self._cpu_config.max_output_chars:
                decoded = decoded[:self._cpu_config.max_output_chars] + "... [truncated]"
            return decoded
        except Exception:
            return None

    def _parse_subprocess_output(
        self,
        stdout: bytes,
        stderr: Optional[bytes],
        return_code: Optional[int],
        model_alias: str,
        timing: AdapterTiming,
        soft_timeout_triggered: bool,
    ) -> AdapterResult:
        """Parse subprocess output into AdapterResult."""
        # Check return code
        if return_code != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
            # Check for refusal indicators
            if return_code == 77:  # Convention: 77 = refusal
                return AdapterResult(
                    status=AdapterStatus.refused,
                    refusal_reason=error_msg,
                    provenance=self.create_provenance(model_alias, timing=timing),
                )
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Process exited with code {return_code}: {error_msg}",
                error_code=f"EXIT_{return_code}",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        # Parse stdout as JSON
        if not stdout:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message="No output from subprocess",
                error_code="NO_OUTPUT",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        try:
            decoded = stdout.decode("utf-8")
            # Truncate if too long
            if len(decoded) > self._cpu_config.max_output_chars:
                decoded = decoded[:self._cpu_config.max_output_chars]

            # Try to parse as JSON
            try:
                output_data = json.loads(decoded)
                content = output_data.get("content") or output_data.get("response") or decoded
                structured = output_data if isinstance(output_data, dict) else None
                input_tokens = output_data.get("input_tokens")
                output_tokens = output_data.get("output_tokens")
                confidence = output_data.get("confidence")
            except json.JSONDecodeError:
                # Plain text output
                content = decoded
                structured = None
                input_tokens = None
                output_tokens = None
                confidence = None

            status = AdapterStatus.soft_timeout if soft_timeout_triggered else AdapterStatus.success

            return AdapterResult(
                status=status,
                content=content,
                structured=structured,
                is_partial=soft_timeout_triggered,
                model_confidence=confidence,
                provenance=self.create_provenance(
                    model_alias,
                    timing=timing,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        except UnicodeDecodeError:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message="Output is not valid UTF-8",
                error_code="INVALID_OUTPUT_ENCODING",
                provenance=self.create_provenance(model_alias, timing=timing),
            )


class StubCPUAdapter(CPUAdapter):
    """
    Stub CPU adapter for testing.

    Returns predefined responses without actual subprocess execution.
    """

    def __init__(
        self,
        config: CPUAdapterConfig,
        stub_responses: Optional[Dict[str, str]] = None,
        stub_delay_ms: int = 100,
    ):
        """Initialize stub adapter."""
        super().__init__(config)
        self._stub_responses = stub_responses or {}
        self._stub_delay_ms = stub_delay_ms

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Return stub response."""
        start_time = datetime.utcnow()

        # Simulate processing delay
        await asyncio.sleep(self._stub_delay_ms / 1000)

        end_time = datetime.utcnow()
        timing = AdapterTiming.create(
            start_time, end_time,
            execution_ms=self._stub_delay_ms,
        )

        # Get stub response
        response = self._stub_responses.get(
            model_alias,
            f"Stub response for {model_alias}: {call.prompt[:50]}..."
        )

        return AdapterResult(
            status=AdapterStatus.success,
            content=response,
            provenance=self.create_provenance(model_alias, timing=timing),
        )
