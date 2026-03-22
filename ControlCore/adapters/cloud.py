"""
Cloud LLM Adapters - OpenAI, Anthropic, xAI (Grok), Google (Gemini), and more.

Unified adapter system for major cloud LLM providers.
Each provider has its own class but shares common infrastructure.
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

import httpx

from ControlCore.schemas import ControlCoreCall
from ControlCore.adapters.interface import (
    ExecutionAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
    AdapterTiming,
)


class CloudProvider(str, Enum):
    """Supported cloud LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"  # Grok
    GOOGLE = "google"  # Gemini
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"  # Together.ai
    GROQ = "groq"  # Groq (fast inference)
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"


# Provider endpoint configurations
PROVIDER_ENDPOINTS = {
    CloudProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
    CloudProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
    CloudProvider.XAI: "https://api.x.ai/v1/chat/completions",
    CloudProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    CloudProvider.MISTRAL: "https://api.mistral.ai/v1/chat/completions",
    CloudProvider.COHERE: "https://api.cohere.ai/v1/chat",
    CloudProvider.TOGETHER: "https://api.together.xyz/v1/chat/completions",
    CloudProvider.GROQ: "https://api.groq.com/openai/v1/chat/completions",
    CloudProvider.DEEPSEEK: "https://api.deepseek.com/v1/chat/completions",
    CloudProvider.PERPLEXITY: "https://api.perplexity.ai/chat/completions",
}

# Environment variable names for API keys
PROVIDER_ENV_KEYS = {
    CloudProvider.OPENAI: "OPENAI_API_KEY",
    CloudProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    CloudProvider.XAI: "XAI_API_KEY",
    CloudProvider.GOOGLE: "GOOGLE_API_KEY",
    CloudProvider.MISTRAL: "MISTRAL_API_KEY",
    CloudProvider.COHERE: "COHERE_API_KEY",
    CloudProvider.TOGETHER: "TOGETHER_API_KEY",
    CloudProvider.GROQ: "GROQ_API_KEY",
    CloudProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    CloudProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
}


class CloudAdapterConfig(AdapterConfig):
    """Configuration for cloud adapters."""

    # Provider
    provider: CloudProvider

    # API key (if not using env var)
    api_key: Optional[str] = None

    # Custom endpoint override
    endpoint_override: Optional[str] = None

    # Models this adapter handles
    handled_models: Set[str] = set()

    # Model alias to provider model ID mapping
    model_mapping: Dict[str, str] = {}

    # Default parameters
    default_max_tokens: int = 4096
    default_temperature: float = 0.7

    # Connection settings
    connect_timeout: float = 10.0
    read_timeout: float = 300.0


class CloudAdapter(ExecutionAdapter):
    """
    Base cloud adapter with provider-specific subclasses.

    Handles common logic for API calls, credential management, and response parsing.
    """

    def __init__(self, config: CloudAdapterConfig):
        super().__init__(config)
        self._cloud_config = config

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self._cloud_config.api_key:
            return self._cloud_config.api_key

        env_key = PROVIDER_ENV_KEYS.get(self._cloud_config.provider)
        if env_key:
            return os.environ.get(env_key)

        return None

    def _get_endpoint(self, model_id: Optional[str] = None) -> str:
        """Get API endpoint for this provider."""
        if self._cloud_config.endpoint_override:
            return self._cloud_config.endpoint_override

        endpoint = PROVIDER_ENDPOINTS.get(self._cloud_config.provider, "")

        # Google needs model in URL
        if self._cloud_config.provider == CloudProvider.GOOGLE and model_id:
            endpoint = endpoint.format(model=model_id)

        return endpoint

    def _resolve_model_id(self, model_alias: str) -> str:
        """Resolve alias to provider model ID."""
        if model_alias in self._cloud_config.model_mapping:
            return self._cloud_config.model_mapping[model_alias]
        return model_alias

    def can_handle(self, model_alias: str) -> bool:
        """Check if this adapter can handle the model."""
        if self._cloud_config.handled_models:
            return model_alias in self._cloud_config.handled_models
        if model_alias in self._cloud_config.model_mapping:
            return True
        return False

    @abstractmethod
    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Build headers and payload for the request. Returns (headers, payload)."""
        pass

    @abstractmethod
    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        """Parse response. Returns (content, input_tokens, output_tokens)."""
        pass

    def _check_refusal(self, data: Dict[str, Any]) -> Optional[str]:
        """Check if response indicates refusal. Returns reason or None."""
        return None

    async def execute(
        self,
        call: ControlCoreCall,
        model_alias: str,
        *,
        soft_timeout_ms: Optional[int] = None,
        hard_timeout_ms: Optional[int] = None,
    ) -> AdapterResult:
        """Execute call via cloud API."""
        start_time = datetime.utcnow()

        # Check API key
        api_key = self._get_api_key()
        if not api_key:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"No API key for {self._cloud_config.provider.value}",
                error_code="NO_API_KEY",
                provenance=self.create_provenance(model_alias),
            )

        # Resolve model
        model_id = self._resolve_model_id(model_alias)

        # Get endpoint
        endpoint = self._get_endpoint(model_id)

        # Get timeouts
        _, hard_ms = self.get_effective_timeouts(call, soft_timeout_ms, hard_timeout_ms)

        # Build request
        try:
            headers, payload = self._build_request(call, model_id)
        except Exception as e:
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Failed to build request: {e}",
                error_code="REQUEST_BUILD_ERROR",
                provenance=self.create_provenance(model_alias),
            )

        # Execute request
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._cloud_config.connect_timeout,
                    read=hard_ms / 1000,
                    write=30.0,
                    pool=10.0,
                ),
            ) as client:
                response = await client.post(endpoint, headers=headers, json=payload)

            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)

            # Handle errors
            if response.status_code == 429:
                return AdapterResult(
                    status=AdapterStatus.rate_limited,
                    error_message="Rate limited",
                    error_code="RATE_LIMITED",
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            if response.status_code == 401 or response.status_code == 403:
                return AdapterResult(
                    status=AdapterStatus.error,
                    error_message="Authentication failed",
                    error_code="AUTH_ERROR",
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            if response.status_code >= 400:
                error_text = response.text[:500]
                return AdapterResult(
                    status=AdapterStatus.error,
                    error_message=f"API error {response.status_code}: {error_text}",
                    error_code=f"API_{response.status_code}",
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            # Parse response
            data = response.json()

            # Check for refusal
            refusal_reason = self._check_refusal(data)
            if refusal_reason:
                return AdapterResult(
                    status=AdapterStatus.refused,
                    refusal_reason=refusal_reason,
                    provenance=self.create_provenance(model_alias, timing=timing),
                )

            # Extract content
            content, input_tokens, output_tokens = self._parse_response(data)

            return AdapterResult(
                status=AdapterStatus.success,
                content=content,
                structured=data,
                provenance=self.create_provenance(
                    model_alias,
                    timing=timing,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_model_id=model_id,
                ),
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

        except Exception as e:
            end_time = datetime.utcnow()
            timing = AdapterTiming.create(start_time, end_time)
            return AdapterResult(
                status=AdapterStatus.error,
                error_message=f"Request failed: {str(e)}",
                error_code="REQUEST_ERROR",
                provenance=self.create_provenance(model_alias, timing=timing),
            )


# =============================================================================
# OpenAI Adapter (GPT-4, GPT-4o, o1, etc.)
# =============================================================================

class OpenAIAdapter(CloudAdapter):
    """OpenAI API adapter for GPT models."""

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build messages
        messages = []
        for ctx in call.context:
            messages.append({"role": "user", "content": ctx.content})
        messages.append({"role": "user", "content": call.prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self._cloud_config.default_max_tokens,
        }

        if call.params:
            if call.params.temperature is not None:
                payload["temperature"] = call.params.temperature
            if call.params.seed is not None:
                payload["seed"] = call.params.seed

        return headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        choices = data.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return content, input_tokens, output_tokens

    def _check_refusal(self, data: Dict[str, Any]) -> Optional[str]:
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            if message.get("refusal"):
                return message.get("refusal")
            # Check finish reason
            if choices[0].get("finish_reason") == "content_filter":
                return "Content filtered"
        return None


# =============================================================================
# Anthropic Adapter (Claude)
# =============================================================================

class AnthropicAdapter(CloudAdapter):
    """Anthropic API adapter for Claude models."""

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Build messages
        messages = []
        for ctx in call.context:
            messages.append({"role": "user", "content": ctx.content})
        messages.append({"role": "user", "content": call.prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self._cloud_config.default_max_tokens,
        }

        if call.params and call.params.temperature is not None:
            payload["temperature"] = call.params.temperature

        return headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return content, input_tokens, output_tokens

    def _check_refusal(self, data: Dict[str, Any]) -> Optional[str]:
        if data.get("stop_reason") == "end_turn":
            # Check content for refusal patterns
            content_blocks = data.get("content", [])
            for block in content_blocks:
                if block.get("type") == "text":
                    text = block.get("text", "").lower()
                    if "i cannot" in text or "i'm not able to" in text:
                        return "Model declined request"
        return None


# =============================================================================
# xAI Adapter (Grok)
# =============================================================================

class XAIAdapter(CloudAdapter):
    """xAI API adapter for Grok models."""

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build messages (OpenAI-compatible format)
        messages = []
        for ctx in call.context:
            messages.append({"role": "user", "content": ctx.content})
        messages.append({"role": "user", "content": call.prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self._cloud_config.default_max_tokens,
        }

        if call.params:
            if call.params.temperature is not None:
                payload["temperature"] = call.params.temperature

        return headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        # xAI uses OpenAI-compatible format
        choices = data.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return content, input_tokens, output_tokens


# =============================================================================
# Google Adapter (Gemini)
# =============================================================================

class GoogleAdapter(CloudAdapter):
    """Google API adapter for Gemini models."""

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "Content-Type": "application/json",
        }

        # Build contents (Google format)
        parts = []
        for ctx in call.context:
            parts.append({"text": ctx.content})
        parts.append({"text": call.prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": self._cloud_config.default_max_tokens,
            },
        }

        if call.params and call.params.temperature is not None:
            payload["generationConfig"]["temperature"] = call.params.temperature

        return headers, payload

    def _get_endpoint(self, model_id: Optional[str] = None) -> str:
        """Google needs API key as query param and model in URL."""
        api_key = self._get_api_key()
        base = PROVIDER_ENDPOINTS[CloudProvider.GOOGLE].format(model=model_id or "gemini-pro")
        return f"{base}?key={api_key}"

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                content += part.get("text", "")

        # Google doesn't always return token counts
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return content, input_tokens, output_tokens

    def _check_refusal(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = data.get("candidates", [])
        if candidates:
            finish_reason = candidates[0].get("finishReason")
            if finish_reason == "SAFETY":
                return "Content blocked by safety filters"
        return None


# =============================================================================
# OpenAI-Compatible Adapter (Groq, Together, Mistral, DeepSeek, Perplexity)
# =============================================================================

class OpenAICompatibleAdapter(CloudAdapter):
    """
    Adapter for providers with OpenAI-compatible APIs.

    Works with: Groq, Together.ai, Mistral, DeepSeek, Perplexity, etc.
    """

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build messages
        messages = []
        for ctx in call.context:
            messages.append({"role": "user", "content": ctx.content})
        messages.append({"role": "user", "content": call.prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self._cloud_config.default_max_tokens,
        }

        if call.params:
            if call.params.temperature is not None:
                payload["temperature"] = call.params.temperature
            if call.params.seed is not None:
                payload["seed"] = call.params.seed

        return headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        choices = data.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return content, input_tokens, output_tokens


# =============================================================================
# Cohere Adapter
# =============================================================================

class CohereAdapter(CloudAdapter):
    """Cohere API adapter."""

    def _build_request(
        self,
        call: ControlCoreCall,
        model_id: str,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Combine context and prompt
        full_message = ""
        for ctx in call.context:
            full_message += ctx.content + "\n\n"
        full_message += call.prompt

        payload = {
            "model": model_id,
            "message": full_message,
            "max_tokens": self._cloud_config.default_max_tokens,
        }

        if call.params and call.params.temperature is not None:
            payload["temperature"] = call.params.temperature

        return headers, payload

    def _parse_response(self, data: Dict[str, Any]) -> tuple[str, int, int]:
        content = data.get("text", "")

        # Cohere token format
        input_tokens = data.get("meta", {}).get("tokens", {}).get("input_tokens", 0)
        output_tokens = data.get("meta", {}).get("tokens", {}).get("output_tokens", 0)

        return content, input_tokens, output_tokens


# =============================================================================
# Factory Functions
# =============================================================================

def create_openai_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAIAdapter:
    """Create OpenAI adapter."""
    config = CloudAdapterConfig(
        adapter_name="openai",
        adapter_version="1.0.0",
        provider=CloudProvider.OPENAI,
        api_key=api_key,
        model_mapping=model_mapping or {
            "gpt4": "gpt-4",
            "gpt4o": "gpt-4o",
            "gpt4o-mini": "gpt-4o-mini",
            "gpt4-turbo": "gpt-4-turbo",
            "o1": "o1",
            "o1-mini": "o1-mini",
            "o1-preview": "o1-preview",
        },
        handled_models={
            "gpt4", "gpt4o", "gpt4o-mini", "gpt4-turbo",
            "o1", "o1-mini", "o1-preview",
            "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        },
    )
    return OpenAIAdapter(config)


def create_anthropic_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> AnthropicAdapter:
    """Create Anthropic adapter."""
    config = CloudAdapterConfig(
        adapter_name="anthropic",
        adapter_version="1.0.0",
        provider=CloudProvider.ANTHROPIC,
        api_key=api_key,
        model_mapping=model_mapping or {
            "claude": "claude-sonnet-4-20250514",
            "claude-sonnet": "claude-sonnet-4-20250514",
            "claude-opus": "claude-opus-4-20250514",
            "claude-haiku": "claude-3-5-haiku-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        },
        handled_models={
            "claude", "claude-sonnet", "claude-opus", "claude-haiku",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        },
    )
    return AnthropicAdapter(config)


def create_xai_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> XAIAdapter:
    """Create xAI (Grok) adapter."""
    config = CloudAdapterConfig(
        adapter_name="xai",
        adapter_version="1.0.0",
        provider=CloudProvider.XAI,
        api_key=api_key,
        model_mapping=model_mapping or {
            "grok": "grok-beta",
            "grok-beta": "grok-beta",
            "grok-2": "grok-2",
            "grok-2-mini": "grok-2-mini",
        },
        handled_models={"grok", "grok-beta", "grok-2", "grok-2-mini"},
    )
    return XAIAdapter(config)


def create_google_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> GoogleAdapter:
    """Create Google (Gemini) adapter."""
    config = CloudAdapterConfig(
        adapter_name="google",
        adapter_version="1.0.0",
        provider=CloudProvider.GOOGLE,
        api_key=api_key,
        model_mapping=model_mapping or {
            "gemini": "gemini-1.5-pro",
            "gemini-pro": "gemini-1.5-pro",
            "gemini-flash": "gemini-1.5-flash",
            "gemini-2": "gemini-2.0-flash-exp",
            "gemini-2-flash": "gemini-2.0-flash-exp",
        },
        handled_models={"gemini", "gemini-pro", "gemini-flash", "gemini-2", "gemini-2-flash"},
    )
    return GoogleAdapter(config)


def create_groq_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAICompatibleAdapter:
    """Create Groq adapter (fast inference)."""
    config = CloudAdapterConfig(
        adapter_name="groq",
        adapter_version="1.0.0",
        provider=CloudProvider.GROQ,
        api_key=api_key,
        model_mapping=model_mapping or {
            "llama-70b": "llama-3.3-70b-versatile",
            "llama-8b": "llama-3.1-8b-instant",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma2-9b-it",
        },
        handled_models={"llama-70b", "llama-8b", "mixtral", "gemma"},
    )
    return OpenAICompatibleAdapter(config)


def create_together_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAICompatibleAdapter:
    """Create Together.ai adapter."""
    config = CloudAdapterConfig(
        adapter_name="together",
        adapter_version="1.0.0",
        provider=CloudProvider.TOGETHER,
        api_key=api_key,
        model_mapping=model_mapping or {
            "llama-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "qwen-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        },
        handled_models={"llama-405b", "qwen-72b", "deepseek-v3"},
    )
    return OpenAICompatibleAdapter(config)


def create_mistral_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAICompatibleAdapter:
    """Create Mistral adapter."""
    config = CloudAdapterConfig(
        adapter_name="mistral",
        adapter_version="1.0.0",
        provider=CloudProvider.MISTRAL,
        api_key=api_key,
        model_mapping=model_mapping or {
            "mistral-large": "mistral-large-latest",
            "mistral-medium": "mistral-medium-latest",
            "mistral-small": "mistral-small-latest",
            "codestral": "codestral-latest",
        },
        handled_models={"mistral-large", "mistral-medium", "mistral-small", "codestral"},
    )
    return OpenAICompatibleAdapter(config)


def create_deepseek_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAICompatibleAdapter:
    """Create DeepSeek adapter."""
    config = CloudAdapterConfig(
        adapter_name="deepseek",
        adapter_version="1.0.0",
        provider=CloudProvider.DEEPSEEK,
        api_key=api_key,
        model_mapping=model_mapping or {
            "deepseek": "deepseek-chat",
            "deepseek-chat": "deepseek-chat",
            "deepseek-coder": "deepseek-coder",
            "deepseek-reasoner": "deepseek-reasoner",
        },
        handled_models={"deepseek", "deepseek-chat", "deepseek-coder", "deepseek-reasoner"},
    )
    return OpenAICompatibleAdapter(config)


def create_perplexity_adapter(
    api_key: Optional[str] = None,
    model_mapping: Optional[Dict[str, str]] = None,
) -> OpenAICompatibleAdapter:
    """Create Perplexity adapter (search-augmented)."""
    config = CloudAdapterConfig(
        adapter_name="perplexity",
        adapter_version="1.0.0",
        provider=CloudProvider.PERPLEXITY,
        api_key=api_key,
        model_mapping=model_mapping or {
            "pplx-online": "llama-3.1-sonar-huge-128k-online",
            "pplx-sonar": "llama-3.1-sonar-large-128k-chat",
        },
        handled_models={"pplx-online", "pplx-sonar"},
    )
    return OpenAICompatibleAdapter(config)


# =============================================================================
# All-in-One Factory
# =============================================================================

def create_all_cloud_adapters() -> Dict[str, CloudAdapter]:
    """
    Create all cloud adapters.

    Returns dict of provider name -> adapter.
    API keys should be set in environment variables.
    """
    return {
        "openai": create_openai_adapter(),
        "anthropic": create_anthropic_adapter(),
        "xai": create_xai_adapter(),
        "google": create_google_adapter(),
        "groq": create_groq_adapter(),
        "together": create_together_adapter(),
        "mistral": create_mistral_adapter(),
        "deepseek": create_deepseek_adapter(),
        "perplexity": create_perplexity_adapter(),
    }
