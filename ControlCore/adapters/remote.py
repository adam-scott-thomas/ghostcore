"""
P3-03: Remote API Adapter (Read-Only Calls)

Calls external LLM APIs using explicit credentials.
Normalizes responses into adapter output format.

This adapter:
- Calls external APIs with explicit credentials
- Enforces request and response size limits
- Normalizes responses to AdapterResult format
- Redacts secrets from logs

This adapter does NOT:
- Store credentials in memory beyond request scope
- Retry internally
- Perform fallback
- Cache responses
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse

import httpx

from ControlCore.schemas import ControlCoreCall
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterTiming,
)


class RemoteAdapterConfig(AdapterConfig):
    """Configuration for remote API adapter."""

    # API endpoint configuration (alias -> endpoint URL)
    endpoints: Dict[str, str] = {}

    # Request size limits
    max_request_bytes: int = 10_000_000  # 10MB

    # Response size limits
    max_response_bytes: int = 50_000_000  # 50MB

    # Connection settings
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 300.0

    # SSL verification
    verify_ssl: bool = True

    # Headers to always include (NOT for secrets)
    default_headers: Dict[str, str] = {}

    # Patterns for secrets to redact in logs
    secret_patterns: List[str] = [
        r"sk-[a-zA-Z0-9]+",           # OpenAI-style keys
        r"key-[a-zA-Z0-9]+",          # Generic API keys
        r"Bearer\s+[a-zA-Z0-9\-_.]+", # Bearer tokens
        r"api[_-]?key[=:]\s*\S+",     # Various API key formats
    ]


class CredentialProvider:
    """
    Provides credentials for API requests.

    Credentials are retrieved fresh for each request and
    NEVER stored in adapter state.
    """

    def __init__(self, credential_getter: Callable[[str], Optional[str]]):
        """
        Initialize credential provider.

        Args:
            credential_getter: Function that takes model_alias and returns
                               the credential string (e.g., API key)
        """
        self._getter = credential_getter

    def get_credential(self, model_alias: str) -> Optional[str]:
        """Get credential for model (fresh each time)."""
        return self._getter(model_alias)


class RemoteAPIAdapter(ExecutionAdapter):
    """
    Remote API execution adapter.

    Calls external LLM APIs with credential handling and response normalization.
    """

    def __init__(
        self,
        config: RemoteAdapterConfig,
        credential_provider: Optional[CredentialProvider] = None,
    ):
        """Initialize remote adapter."""
        super().__init__(config)
        self._remote_config = config
        self._credential_provider = credential_provider
        self._secret_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in config.secret_patterns
        ]

    def can_handle(self, model_alias: str) -> bool:
        """Check if this adapter can handle the model."""
        return model_alias in self._remote_config.endpoints

    def _redact_secrets(self, text: str) -> str:
        """Redact secrets from text for safe logging."""
        result = text
        for pattern in self._secret_patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def _build_request_payload(
        self,
        call: ControlCoreCall,
        model_alias: str,
    ) -> Dict[str, Any]:
        """Build request payload for API."""
        # Build messages array
        messages = []

        # Add context messages
        for ctx in call.context:
            messages.append({
                "role": ctx.role.value,
                "content": ctx.content,
            })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": call.prompt,
        })

        payload = {
            "messages": messages,
        }

        # Add optional parameters
        if call.options.params:
            if call.options.params.temperature is not None:
                payload["temperature"] = call.options.params.temperature
            if call.options.params.max_tokens is not None:
                payload["max_tokens"] = call.options.params.max_tokens
            if call.options.params.seed is not None:
                payload["seed"] = call.options.params.seed
            if call.options.params.stop_sequences:
                payload["stop"] = call.options.params.stop_sequences

        return payload

    def _build_headers(self, credential: Optional[str]) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._remote_config.default_headers,
        }

        if credential:
            # Support both Bearer token and API key styles
            if credential.startswith("Bearer "):
                headers["Authorization"] = credential
            else:
                headers["Authorization"] = f"Bearer {credential}"

        return headers

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Execute call via remote API."""
        start_time = datetime.utcnow()

        # Validate model is configured
        if not self.can_handle(model_alias):
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Model '{model_alias}' not configured for remote adapter",
                error_code="MODEL_NOT_CONFIGURED",
                provenance=self.create_provenance(model_alias),
            )

        # Get endpoint
        endpoint = self._remote_config.endpoints[model_alias]

        # Get credential (fresh for this request)
        credential = None
        if self._credential_provider:
            credential = self._credential_provider.get_credential(model_alias)

        # Get timeouts
        soft_ms, hard_ms = self.get_effective_timeouts(call, soft_timeout_ms, hard_timeout_ms)

        # Build request
        try:
            payload = self._build_request_payload(call, model_alias)
            payload_json = json.dumps(payload)

            # Check request size
            if len(payload_json.encode()) > self._remote_config.max_request_bytes:
                return AdapterResult(
                    status=AdapterStatus.error,
                    error_message="Request exceeds maximum size limit",
                    error_code="REQUEST_TOO_LARGE",
                    provenance=self.create_provenance(model_alias),
                )

        except Exception as e:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Failed to build request: {str(e)}",
                error_code="REQUEST_BUILD_ERROR",
                provenance=self.create_provenance(model_alias),
            )

        # Build headers
        headers = self._build_headers(credential)

        # Execute request
        network_start = datetime.utcnow()

        try:
            async with httpx.AsyncClient(
                verify=self._remote_config.verify_ssl,
                timeout=httpx.Timeout(
                    connect=self._remote_config.connect_timeout_seconds,
                    read=hard_ms / 1000,  # Use hard timeout for read
                    write=30.0,
                    pool=10.0,
                ),
            ) as client:
                response = await client.post(
                    endpoint,
                    content=payload_json,
                    headers=headers,
                )

            network_end = datetime.utcnow()
            network_ms = int((network_end - network_start).total_seconds() * 1000)

            end_time = datetime.utcnow()
            timing = AdapterTiming.create(
                start_time, end_time,
                network_ms=network_ms,
                execution_ms=network_ms,
            )

            # Check response size
            content_length = len(response.content)
            if content_length > self._remote_config.max_response_bytes:
                return AdapterResult(
                    status=AdapterStatus.error,
                    error_message="Response exceeds maximum size limit",
                    error_code="RESPONSE_TOO_LARGE",
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            # Parse response
            return self._parse_response(
                response=response,
                model_alias=model_alias,
                timing=timing,
                soft_timeout_ms=soft_ms,
            )

        except httpx.TimeoutException:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.timeout,
                error_message="Request timed out",
                error_code="TIMEOUT",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except httpx.ConnectError as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Connection failed: {self._redact_secrets(str(e))}",
                error_code="CONNECTION_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except httpx.HTTPStatusError as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return self._handle_http_error(e, model_alias, timing)

        except Exception as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Request failed: {self._redact_secrets(str(e))}",
                error_code="REQUEST_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        model_alias: str,
        timing: AdapterTiming,
    ) -> AdapterResult:
        """Handle HTTP errors and map to appropriate status."""
        status_code = error.response.status_code

        if status_code == 429:
            return AdapterResult(
                status=AdapterStatus.rate_limited,
                error_message="Rate limited by provider",
                error_code="RATE_LIMITED",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        if status_code == 401 or status_code == 403:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message="Authentication failed",
                error_code="AUTH_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        if status_code >= 500:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Provider error: {status_code}",
                error_code=f"PROVIDER_ERROR_{status_code}",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        return AdapterResult(
            status=AdapterStatus.error,
            error_message=f"HTTP error: {status_code}",
            error_code=f"HTTP_{status_code}",
            provenance=self.create_provenance(model_alias, timing=timing),
        )

    def _parse_response(
        self,
        response: httpx.Response,
        model_alias: str,
        timing: AdapterTiming,
        soft_timeout_ms: int,
    ) -> AdapterResult:
        """Parse API response into AdapterResult."""
        # Check for soft timeout
        soft_timeout_triggered = timing.total_ms > soft_timeout_ms

        # Handle non-2xx responses
        if response.status_code >= 400:
            return self._handle_http_error(
                httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                ),
                model_alias,
                timing,
            )

        try:
            data = response.json()
        except json.JSONDecodeError:
            # Try plain text
            content = response.text
            return AdapterResult(
                status=AdapterStatus.soft_timeout if soft_timeout_triggered else AdapterStatus.success,
                content=content,
                is_partial=soft_timeout_triggered,
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        # Extract content from various API formats
        content = self._extract_content(data)
        input_tokens = self._extract_input_tokens(data)
        output_tokens = self._extract_output_tokens(data)
        confidence = self._extract_confidence(data)

        # Check for refusal indicators
        if self._is_refusal(data):
            return AdapterResult(
                status=AdapterStatus.refused,
                content=content,
                refusal_reason=self._extract_refusal_reason(data),
                provenance=self.create_provenance(
                    model_alias, timing=timing,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        # Check for content filter
        if self._is_content_filtered(data):
            return AdapterResult(
                status=AdapterStatus.content_filtered,
                error_message="Content was filtered by provider",
                error_code="CONTENT_FILTERED",
                provenance=self.create_provenance(
                    model_alias, timing=timing,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        status = AdapterStatus.soft_timeout if soft_timeout_triggered else AdapterStatus.success

        return AdapterResult(
            status=status,
            content=content,
            structured=data,
            is_partial=soft_timeout_triggered,
            model_confidence=confidence,
            provenance=self.create_provenance(
                model_alias, timing=timing,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            raw_response=self._redact_response(data),
        )

    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract content from various API response formats."""
        # OpenAI format
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice:
                return choice["message"].get("content")
            if "text" in choice:
                return choice["text"]

        # Anthropic format
        if "content" in data and isinstance(data["content"], list):
            texts = [c.get("text", "") for c in data["content"] if c.get("type") == "text"]
            return "".join(texts) if texts else None

        # Direct content
        if "content" in data and isinstance(data["content"], str):
            return data["content"]

        # Response field
        if "response" in data:
            return data["response"]

        # Output field
        if "output" in data:
            return data["output"]

        return None

    def _extract_input_tokens(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract input token count."""
        usage = data.get("usage", {})
        return usage.get("prompt_tokens") or usage.get("input_tokens")

    def _extract_output_tokens(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract output token count."""
        usage = data.get("usage", {})
        return usage.get("completion_tokens") or usage.get("output_tokens")

    def _extract_confidence(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract model confidence if available."""
        return data.get("confidence")

    def _is_refusal(self, data: Dict[str, Any]) -> bool:
        """Check if response indicates refusal."""
        # Check for explicit refusal flag
        if data.get("refusal") or data.get("refused"):
            return True

        # Check finish reason
        if "choices" in data and data["choices"]:
            finish_reason = data["choices"][0].get("finish_reason")
            if finish_reason in ("content_filter", "refusal"):
                return True

        return False

    def _extract_refusal_reason(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract refusal reason if available."""
        return data.get("refusal_reason") or data.get("refusal_message")

    def _is_content_filtered(self, data: Dict[str, Any]) -> bool:
        """Check if content was filtered."""
        if "choices" in data and data["choices"]:
            finish_reason = data["choices"][0].get("finish_reason")
            if finish_reason == "content_filter":
                return True

        # Check for content filter flag
        if data.get("content_filter_results"):
            results = data["content_filter_results"]
            if isinstance(results, dict):
                for category, result in results.items():
                    if result.get("filtered"):
                        return True

        return False

    def _redact_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from response for storage."""
        # Deep copy and redact
        import copy
        redacted = copy.deepcopy(data)

        # Remove any credential-like fields
        sensitive_keys = ["api_key", "key", "token", "secret", "credential", "password"]
        def redact_dict(d: Dict[str, Any]) -> None:
            for key in list(d.keys()):
                if any(s in key.lower() for s in sensitive_keys):
                    d[key] = "[REDACTED]"
                elif isinstance(d[key], dict):
                    redact_dict(d[key])
                elif isinstance(d[key], str):
                    d[key] = self._redact_secrets(d[key])

        redact_dict(redacted)
        return redacted


class MockRemoteAdapter(RemoteAPIAdapter):
    """
    Mock remote adapter for testing.

    Returns predefined responses without making actual HTTP requests.
    """

    def __init__(
        self,
        config: RemoteAdapterConfig,
        mock_responses: Optional[Dict[str, Dict[str, Any]]] = None,
        mock_delay_ms: int = 100,
    ):
        """Initialize mock adapter."""
        super().__init__(config)
        self._mock_responses = mock_responses or {}
        self._mock_delay_ms = mock_delay_ms

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Return mock response."""
        start_time = datetime.utcnow()

        # Simulate network delay
        await asyncio.sleep(self._mock_delay_ms / 1000)

        end_time = datetime.utcnow()
        timing = AdapterTiming.create(
            start_time, end_time,
            network_ms=self._mock_delay_ms,
            execution_ms=self._mock_delay_ms,
        )

        # Get mock response
        mock_data = self._mock_responses.get(model_alias, {
            "choices": [{
                "message": {
                    "content": f"Mock response for {model_alias}: {call.prompt[:50]}..."
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(call.prompt) // 4,
                "completion_tokens": 50,
            },
        })

        # Parse as if it were a real response
        content = self._extract_content(mock_data)
        input_tokens = self._extract_input_tokens(mock_data)
        output_tokens = self._extract_output_tokens(mock_data)

        return AdapterResult(
            status=AdapterStatus.success,
            content=content,
            structured=mock_data,
            provenance=self.create_provenance(
                model_alias, timing=timing,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
