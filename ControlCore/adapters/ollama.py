"""
Ollama Adapter - Local LLM execution via Ollama API.

Connects to Ollama server (default: http://localhost:11434) for local model inference.
Supports all models imported into Ollama.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import httpx

from ControlCore.schemas import ControlCoreCall
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterTiming,
)


class OllamaAdapterConfig(AdapterConfig):
    """Configuration for Ollama adapter."""

    # Ollama server URL
    base_url: str = "http://localhost:11434"

    # Models this adapter handles (empty = all available)
    handled_models: Set[str] = set()

    # Model alias to Ollama model name mapping
    # e.g., {"qwen-coder": "qwen25-coder-32b:latest"}
    model_mapping: Dict[str, str] = {}

    # Default generation options
    default_options: Dict[str, Any] = {}

    # Stream responses (for partial results on timeout)
    stream: bool = True

    # Connection settings
    connect_timeout: float = 10.0
    read_timeout: float = 300.0


class OllamaAdapter(ExecutionAdapter):
    """
    Ollama execution adapter for local models.

    Calls the Ollama HTTP API to run local LLMs.
    """

    def __init__(self, config: OllamaAdapterConfig):
        """Initialize Ollama adapter."""
        super().__init__(config)
        self._ollama_config = config
        self._available_models: Optional[Set[str]] = None

    async def _fetch_available_models(self) -> Set[str]:
        """Fetch list of available models from Ollama."""
        if self._available_models is not None:
            return self._available_models

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._ollama_config.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = {
                        m["name"] for m in data.get("models", [])
                    }
                    return self._available_models
        except Exception:
            pass

        return set()

    def _resolve_model_name(self, model_alias: str) -> str:
        """Resolve alias to Ollama model name."""
        # Check explicit mapping first
        if model_alias in self._ollama_config.model_mapping:
            return self._ollama_config.model_mapping[model_alias]

        # Try direct match with :latest suffix
        if not ":" in model_alias:
            return f"{model_alias}:latest"

        return model_alias

    def can_handle(self, model_alias: str) -> bool:
        """Check if this adapter can handle the model."""
        # If specific models configured, check those
        if self._ollama_config.handled_models:
            return model_alias in self._ollama_config.handled_models

        # Check mapping
        if model_alias in self._ollama_config.model_mapping:
            return True

        # For local models, we assume we can handle if alias looks like ollama format
        # Real check happens at execution time
        return True

    def _build_request_payload(
        self,
        call: ControlCoreCall,
        ollama_model: str,
    ) -> Dict[str, Any]:
        """Build Ollama API request payload."""
        # Build the prompt with context
        full_prompt = ""

        # Add context parts
        for ctx in call.context:
            full_prompt += f"{ctx.content}\n\n"

        # Add main prompt
        full_prompt += call.prompt

        payload = {
            "model": ollama_model,
            "prompt": full_prompt,
            "stream": self._ollama_config.stream,
        }

        # Add generation options
        options = dict(self._ollama_config.default_options)

        if call.params:
            if call.params.temperature is not None:
                options["temperature"] = call.params.temperature
            if call.params.top_p is not None:
                options["top_p"] = call.params.top_p
            if call.params.seed is not None:
                options["seed"] = call.params.seed

        if options:
            payload["options"] = options

        return payload

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Execute call via Ollama API."""
        start_time = datetime.utcnow()

        # Resolve model name
        ollama_model = self._resolve_model_name(model_alias)

        # Get timeouts
        soft_ms, hard_ms = self.get_effective_timeouts(call, soft_timeout_ms, hard_timeout_ms)

        # Build request
        payload = self._build_request_payload(call, ollama_model)

        # Execute request
        collected_response = []
        input_tokens = 0
        output_tokens = 0

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._ollama_config.connect_timeout,
                    read=hard_ms / 1000,
                    write=30.0,
                    pool=10.0,
                ),
            ) as client:
                if self._ollama_config.stream:
                    # Streaming request
                    async with client.stream(
                        "POST",
                        f"{self._ollama_config.base_url}/api/generate",
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            end_time = datetime.utcnow()
                            timing = AdapterTiming.create(start_time, end_time)
                            error_text = await response.aread()
                            return AdapterResult(
                                status=AdapterStatus.error,
                                error_message=f"Ollama error {response.status_code}: {error_text.decode()[:200]}",
                                error_code=f"OLLAMA_{response.status_code}",
                                provenance=self.create_provenance(model_alias, timing=timing),
                            )

                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    collected_response.append(chunk["response"])
                                if chunk.get("done"):
                                    # Final chunk has stats
                                    input_tokens = chunk.get("prompt_eval_count", 0)
                                    output_tokens = chunk.get("eval_count", 0)
                            except json.JSONDecodeError:
                                continue
                else:
                    # Non-streaming request
                    response = await client.post(
                        f"{self._ollama_config.base_url}/api/generate",
                        json=payload,
                    )

                    if response.status_code != 200:
                        end_time = datetime.utcnow()
                        timing = AdapterTiming.create(start_time, end_time)
                        return AdapterResult(
                            status=AdapterStatus.error,
                            error_message=f"Ollama error {response.status_code}: {response.text[:200]}",
                            error_code=f"OLLAMA_{response.status_code}",
                            provenance=self.create_provenance(model_alias, timing=timing),
                        )

                    data = response.json()
                    collected_response.append(data.get("response", ""))
                    input_tokens = data.get("prompt_eval_count", 0)
                    output_tokens = data.get("eval_count", 0)

            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)

            content = "".join(collected_response)

            return AdapterResult(
                status=AdapterStatus.success,
                content=content,
                provenance=self.create_provenance(
                    model_alias,
                    timing=timing,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_model_id=ollama_model,
                ),
            )

        except httpx.TimeoutException:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)

            # Return partial content if we have any
            partial_content = "".join(collected_response) if collected_response else None

            return AdapterResult(
                status=AdapterStatus.timeout if not partial_content else AdapterStatus.soft_timeout,
                content=partial_content,
                is_partial=bool(partial_content),
                error_message="Request timed out",
                error_code="TIMEOUT",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except httpx.ConnectError as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Cannot connect to Ollama at {self._ollama_config.base_url}: {e}",
                error_code="CONNECTION_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )

        except Exception as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Ollama adapter error: {str(e)}",
                error_code="ADAPTER_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )


def create_ollama_adapter(
    base_url: str = "http://localhost:11434",
    model_mapping: Optional[Dict[str, str]] = None,
) -> OllamaAdapter:
    """
    Create an Ollama adapter with standard configuration.

    Args:
        base_url: Ollama server URL
        model_mapping: Optional alias to model name mapping

    Returns:
        Configured OllamaAdapter
    """
    config = OllamaAdapterConfig(
        adapter_name="ollama",
        adapter_version="1.0.0",
        base_url=base_url,
        model_mapping=model_mapping or {},
        default_options={
            "num_ctx": 8192,  # Context window
        },
    )
    return OllamaAdapter(config)
