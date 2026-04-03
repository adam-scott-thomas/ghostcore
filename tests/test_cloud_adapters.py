"""
Tests for cloud LLM adapters.

Covers: OpenAI, Anthropic, Google, xAI, and all OpenAI-compatible providers
(Groq, Together, Mistral, DeepSeek, Perplexity).
Includes streaming (SSE) tests for providers that support it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from unittest.mock import patch

import httpx
import pytest
import respx

from ControlCore.adapters.cloud import (
    CloudAdapter,
    CloudAdapterConfig,
    CloudProvider,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    XAIAdapter,
    OpenAICompatibleAdapter,
    create_openai_adapter,
    create_anthropic_adapter,
    create_google_adapter,
    create_xai_adapter,
    create_groq_adapter,
    create_together_adapter,
    create_mistral_adapter,
    create_deepseek_adapter,
    create_perplexity_adapter,
    create_all_cloud_adapters,
    PROVIDER_ENDPOINTS,
)
from ControlCore.adapters.interface import AdapterStatus
from tests.conftest import make_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_success_body(content: str = "Hello!") -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": content, "refusal": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _anthropic_success_body(content: str = "Hello!") -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def _google_success_body(content: str = "Hello!") -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": content}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 4},
    }


def _openai_sse_bytes(
    chunks: List[str],
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> bytes:
    """Build OpenAI-format SSE byte stream from content chunks."""
    lines = []
    for chunk_text in chunks:
        data = {"choices": [{"delta": {"content": chunk_text}}]}
        lines.append(f"data: {json.dumps(data)}\n\n")
    # Final chunk with usage
    final = {"choices": [{}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
    lines.append(f"data: {json.dumps(final)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _anthropic_sse_bytes(
    chunks: List[str],
    input_tokens: int = 12,
    output_tokens: int = 8,
) -> bytes:
    """Build Anthropic-format SSE byte stream from content chunks."""
    lines = []
    # message_start with input usage
    msg_start = {"type": "message_start", "message": {"usage": {"input_tokens": input_tokens}}}
    lines.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n")
    # content blocks
    for chunk_text in chunks:
        data = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": chunk_text}}
        lines.append(f"event: content_block_delta\ndata: {json.dumps(data)}\n\n")
    # message_delta with output usage
    msg_delta = {"type": "message_delta", "usage": {"output_tokens": output_tokens}}
    lines.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n")
    lines.append("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
    return "".join(lines).encode()


# ---------------------------------------------------------------------------
# OpenAI Adapter
# ---------------------------------------------------------------------------

class TestOpenAICanHandle:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test")

    def test_known_aliases(self):
        for alias in ("gpt4", "gpt4o", "gpt4o-mini", "gpt4-turbo", "o1", "o1-mini", "o1-preview"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_raw_model_ids(self):
        for alias in ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_unknown_model(self):
        assert not self.adapter.can_handle("claude-sonnet")
        assert not self.adapter.can_handle("gemini")
        assert not self.adapter.can_handle("llama-70b")


class TestOpenAIBuildRequest:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test-key")

    def test_bearer_auth_header(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        headers, _ = self.adapter._build_request(call, "gpt-4o")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_payload_model_field(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        _, payload = self.adapter._build_request(call, "gpt-4o")
        assert payload["model"] == "gpt-4o"

    def test_payload_messages_include_prompt(self):
        call = make_call(target_alias="gpt4o", prompt="What is 2+2?")
        _, payload = self.adapter._build_request(call, "gpt-4o")
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is 2+2?"

    def test_payload_has_max_tokens(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        _, payload = self.adapter._build_request(call, "gpt-4o")
        assert "max_tokens" in payload
        assert payload["max_tokens"] == 4096


class TestOpenAIParseResponse:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test")

    def test_extracts_content(self):
        content, _, _ = self.adapter._parse_response(_openai_success_body("Answer: 4"))
        assert content == "Answer: 4"

    def test_extracts_token_counts(self):
        _, input_tokens, output_tokens = self.adapter._parse_response(_openai_success_body())
        assert input_tokens == 10
        assert output_tokens == 5

    def test_empty_choices(self):
        content, inp, out = self.adapter._parse_response({"choices": [], "usage": {}})
        assert content == ""
        assert inp == 0
        assert out == 0


class TestOpenAICheckRefusal:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test")

    def test_no_refusal_for_normal_response(self):
        result = self.adapter._check_refusal(_openai_success_body())
        assert result is None

    def test_detects_content_filter_finish_reason(self):
        data = {
            "choices": [{"message": {"content": "", "refusal": None}, "finish_reason": "content_filter"}],
            "usage": {},
        }
        result = self.adapter._check_refusal(data)
        assert result == "Content filtered"

    def test_detects_explicit_refusal_field(self):
        data = {
            "choices": [{"message": {"content": "", "refusal": "I can't help with that."}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = self.adapter._check_refusal(data)
        assert result == "I can't help with that."


class TestOpenAIExecute:
    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test")
        self.endpoint = PROVIDER_ENDPOINTS[CloudProvider.OPENAI]

    async def test_success_200(self):
        """OpenAI streams by default — mock SSE response."""
        call = make_call(target_alias="gpt4o", prompt="Hello")
        sse = _openai_sse_bytes(["Hi", " there"], prompt_tokens=10, completion_tokens=5)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.success
        assert result.content == "Hi there"
        assert result.provenance is not None
        assert result.provenance.input_tokens == 10
        assert result.provenance.output_tokens == 5

    async def test_rate_limit_429(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(429, text="rate limited")
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.rate_limited
        assert result.error_code == "RATE_LIMITED"

    async def test_no_api_key_returns_error(self):
        adapter = create_openai_adapter(api_key=None)
        # Ensure the env var is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            call = make_call(target_alias="gpt4o", prompt="Hello")
            result = await adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.error
        assert result.error_code == "NO_API_KEY"
        assert "openai" in result.error_message.lower()

    async def test_auth_error_401(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.error
        assert result.error_code == "AUTH_ERROR"

    async def test_content_filter_returns_refused_nonstream(self):
        """Content filter detection requires non-streaming path (refusal is in JSON body)."""
        config = CloudAdapterConfig(
            adapter_name="openai",
            adapter_version="1.0.0",
            provider=CloudProvider.OPENAI,
            api_key="sk-test",
            stream=False,
            model_mapping={"gpt4o": "gpt-4o"},
            handled_models={"gpt4o"},
        )
        adapter = OpenAIAdapter(config)
        call = make_call(target_alias="gpt4o", prompt="Hello")
        body = {
            "choices": [{"message": {"content": "", "refusal": None}, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        with respx.mock(assert_all_called=True) as mock:
            mock.post(PROVIDER_ENDPOINTS[CloudProvider.OPENAI]).mock(
                return_value=httpx.Response(200, json=body)
            )
            result = await adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.refused


# ---------------------------------------------------------------------------
# Anthropic Adapter
# ---------------------------------------------------------------------------

class TestAnthropicCanHandle:
    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-test")

    def test_known_aliases(self):
        for alias in ("claude", "claude-sonnet", "claude-opus", "claude-haiku",
                      "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_unknown_model(self):
        assert not self.adapter.can_handle("gpt4")
        assert not self.adapter.can_handle("gemini")


class TestAnthropicBuildRequest:
    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-key")

    def test_x_api_key_header(self):
        call = make_call(target_alias="claude", prompt="Hello")
        headers, _ = self.adapter._build_request(call, "claude-sonnet-4-20250514")
        assert headers["x-api-key"] == "sk-ant-key"
        assert "Authorization" not in headers

    def test_anthropic_version_header(self):
        call = make_call(target_alias="claude", prompt="Hello")
        headers, _ = self.adapter._build_request(call, "claude-sonnet-4-20250514")
        assert headers["anthropic-version"] == "2023-06-01"

    def test_payload_model_and_messages(self):
        call = make_call(target_alias="claude", prompt="Test prompt")
        _, payload = self.adapter._build_request(call, "claude-sonnet-4-20250514")
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["messages"][-1]["content"] == "Test prompt"


class TestAnthropicParseResponse:
    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-test")

    def test_extracts_text_from_content_blocks(self):
        content, inp, out = self.adapter._parse_response(_anthropic_success_body("The answer is 42"))
        assert content == "The answer is 42"
        assert inp == 12
        assert out == 8

    def test_concatenates_multiple_text_blocks(self):
        data = {
            "content": [
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        content, _, _ = self.adapter._parse_response(data)
        assert content == "Part 1. Part 2."

    def test_skips_non_text_blocks(self):
        data = {
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "foo", "input": {}},
                {"type": "text", "text": "Done."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        content, _, _ = self.adapter._parse_response(data)
        assert content == "Done."


class TestAnthropicExecute:
    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-test")
        self.endpoint = PROVIDER_ENDPOINTS[CloudProvider.ANTHROPIC]

    async def test_success_200(self):
        """Anthropic streams by default — mock SSE response."""
        call = make_call(target_alias="claude", prompt="Hello")
        sse = _anthropic_sse_bytes(["Got", " it"], input_tokens=12, output_tokens=8)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "claude")

        assert result.status == AdapterStatus.success
        assert result.content == "Got it"

    async def test_no_api_key_returns_error(self):
        adapter = create_anthropic_adapter(api_key=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            call = make_call(target_alias="claude", prompt="Hello")
            result = await adapter.execute(call, "claude")

        assert result.status == AdapterStatus.error
        assert result.error_code == "NO_API_KEY"

    async def test_rate_limit_429(self):
        call = make_call(target_alias="claude", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(429, text="Too Many Requests")
            )
            result = await self.adapter.execute(call, "claude")

        assert result.status == AdapterStatus.rate_limited


# ---------------------------------------------------------------------------
# Google Adapter
# ---------------------------------------------------------------------------

class TestGoogleBuildRequest:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="google-test-key")

    def test_contents_parts_format(self):
        call = make_call(target_alias="gemini", prompt="Hello Google")
        _, payload = self.adapter._build_request(call, "gemini-1.5-pro")
        assert "contents" in payload
        contents = payload["contents"]
        assert isinstance(contents, list)
        assert len(contents) == 1
        parts = contents[0]["parts"]
        assert isinstance(parts, list)
        # Prompt should be the last part
        assert parts[-1]["text"] == "Hello Google"

    def test_generation_config_present(self):
        call = make_call(target_alias="gemini", prompt="Hello")
        _, payload = self.adapter._build_request(call, "gemini-1.5-pro")
        assert "generationConfig" in payload
        assert "maxOutputTokens" in payload["generationConfig"]

    def test_no_auth_header_in_build_request(self):
        call = make_call(target_alias="gemini", prompt="Hello")
        headers, _ = self.adapter._build_request(call, "gemini-1.5-pro")
        # Google uses API key in URL, not headers
        assert "Authorization" not in headers
        assert "x-api-key" not in headers


class TestGoogleGetEndpoint:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="google-test-key")

    def test_includes_model_in_url(self):
        endpoint = self.adapter._get_endpoint("gemini-1.5-pro")
        assert "gemini-1.5-pro" in endpoint

    def test_includes_api_key_as_query_param(self):
        endpoint = self.adapter._get_endpoint("gemini-1.5-pro")
        assert "key=google-test-key" in endpoint

    def test_url_structure(self):
        endpoint = self.adapter._get_endpoint("gemini-1.5-flash")
        assert "generativelanguage.googleapis.com" in endpoint
        assert "gemini-1.5-flash" in endpoint
        assert "generateContent" in endpoint


class TestGoogleParseResponse:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="google-test-key")

    def test_extracts_text_from_candidates(self):
        content, inp, out = self.adapter._parse_response(_google_success_body("Gemini says hi"))
        assert content == "Gemini says hi"
        assert inp == 7
        assert out == 4

    def test_empty_candidates(self):
        content, inp, out = self.adapter._parse_response({"candidates": [], "usageMetadata": {}})
        assert content == ""


class TestGoogleCheckRefusal:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="google-test-key")

    def test_detects_safety_finish_reason(self):
        data = {
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
        }
        result = self.adapter._check_refusal(data)
        assert result == "Content blocked by safety filters"

    def test_no_refusal_for_stop(self):
        result = self.adapter._check_refusal(_google_success_body())
        assert result is None

    def test_no_refusal_for_empty_candidates(self):
        result = self.adapter._check_refusal({"candidates": []})
        assert result is None


class TestGoogleExecute:
    def setup_method(self):
        self.adapter = create_google_adapter(api_key="google-test-key")

    def _google_endpoint_pattern(self, model_id: str = "gemini-1.5-pro") -> str:
        """Build the exact Google endpoint URL for mocking."""
        return self.adapter._get_endpoint(model_id)

    async def test_success_200(self):
        call = make_call(target_alias="gemini", prompt="Hello")
        # Resolve exactly what endpoint execute() will use: gemini -> gemini-1.5-pro
        endpoint = self._google_endpoint_pattern("gemini-1.5-pro")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(endpoint).mock(
                return_value=httpx.Response(200, json=_google_success_body("Gemini response"))
            )
            result = await self.adapter.execute(call, "gemini")

        assert result.status == AdapterStatus.success
        assert result.content == "Gemini response"

    async def test_safety_refusal_returns_refused(self):
        call = make_call(target_alias="gemini", prompt="Hello")
        endpoint = self._google_endpoint_pattern("gemini-1.5-pro")
        body = {
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0},
        }
        with respx.mock(assert_all_called=True) as mock:
            mock.post(endpoint).mock(
                return_value=httpx.Response(200, json=body)
            )
            result = await self.adapter.execute(call, "gemini")

        assert result.status == AdapterStatus.refused
        assert result.refusal_reason == "Content blocked by safety filters"


# ---------------------------------------------------------------------------
# OpenAI-Compatible Adapters (Groq, Together, Mistral, DeepSeek, Perplexity)
# ---------------------------------------------------------------------------

class TestGroqAdapter:
    def setup_method(self):
        self.adapter = create_groq_adapter(api_key="groq-test-key")

    def test_can_handle_groq_models(self):
        for alias in ("llama-70b", "llama-8b", "mixtral", "gemma"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_cannot_handle_other_models(self):
        assert not self.adapter.can_handle("gpt4")
        assert not self.adapter.can_handle("claude")

    def test_model_mapping_resolves(self):
        assert self.adapter._resolve_model_id("llama-70b") == "llama-3.3-70b-versatile"
        assert self.adapter._resolve_model_id("llama-8b") == "llama-3.1-8b-instant"
        assert self.adapter._resolve_model_id("mixtral") == "mixtral-8x7b-32768"
        assert self.adapter._resolve_model_id("gemma") == "gemma2-9b-it"


class TestTogetherAdapter:
    def setup_method(self):
        self.adapter = create_together_adapter(api_key="together-test-key")

    def test_can_handle_together_models(self):
        for alias in ("llama-405b", "qwen-72b", "deepseek-v3"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_model_mapping_resolves(self):
        assert "Meta-Llama" in self.adapter._resolve_model_id("llama-405b")
        assert "Qwen" in self.adapter._resolve_model_id("qwen-72b")
        assert "DeepSeek" in self.adapter._resolve_model_id("deepseek-v3")


class TestMistralAdapter:
    def setup_method(self):
        self.adapter = create_mistral_adapter(api_key="mistral-test-key")

    def test_can_handle_mistral_models(self):
        for alias in ("mistral-large", "mistral-medium", "mistral-small", "codestral"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_model_mapping_resolves(self):
        assert self.adapter._resolve_model_id("mistral-large") == "mistral-large-latest"
        assert self.adapter._resolve_model_id("codestral") == "codestral-latest"


class TestDeepSeekAdapter:
    def setup_method(self):
        self.adapter = create_deepseek_adapter(api_key="deepseek-test-key")

    def test_can_handle_deepseek_models(self):
        for alias in ("deepseek", "deepseek-chat", "deepseek-coder", "deepseek-reasoner"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_model_mapping_resolves(self):
        assert self.adapter._resolve_model_id("deepseek") == "deepseek-chat"
        assert self.adapter._resolve_model_id("deepseek-reasoner") == "deepseek-reasoner"


class TestPerplexityAdapter:
    def setup_method(self):
        self.adapter = create_perplexity_adapter(api_key="pplx-test-key")

    def test_can_handle_perplexity_models(self):
        for alias in ("pplx-online", "pplx-sonar"):
            assert self.adapter.can_handle(alias), f"should handle {alias}"

    def test_model_mapping_resolves(self):
        assert "sonar-huge" in self.adapter._resolve_model_id("pplx-online")
        assert "sonar-large" in self.adapter._resolve_model_id("pplx-sonar")


class TestOpenAICompatibleBuildRequest:
    """Shared request-building behaviour for all OAI-compatible adapters."""

    def setup_method(self):
        self.adapter = create_groq_adapter(api_key="groq-key")

    def test_bearer_auth_header(self):
        call = make_call(target_alias="llama-70b", prompt="Hello")
        headers, _ = self.adapter._build_request(call, "llama-3.3-70b-versatile")
        assert headers["Authorization"] == "Bearer groq-key"

    def test_payload_model_and_messages(self):
        call = make_call(target_alias="llama-70b", prompt="Test")
        _, payload = self.adapter._build_request(call, "llama-3.3-70b-versatile")
        assert payload["model"] == "llama-3.3-70b-versatile"
        assert payload["messages"][-1]["content"] == "Test"


class TestOpenAICompatibleExecute:
    def setup_method(self):
        self.adapter = create_groq_adapter(api_key="groq-key")
        self.endpoint = PROVIDER_ENDPOINTS[CloudProvider.GROQ]

    async def test_success_200(self):
        """Groq (OpenAI-compatible) streams by default — mock SSE response."""
        call = make_call(target_alias="llama-70b", prompt="Hello")
        sse = _openai_sse_bytes(["Fast", " answer"], prompt_tokens=10, completion_tokens=5)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "llama-70b")

        assert result.status == AdapterStatus.success
        assert result.content == "Fast answer"

    async def test_rate_limit_429(self):
        call = make_call(target_alias="llama-70b", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(429, text="Rate limited")
            )
            result = await self.adapter.execute(call, "llama-70b")

        assert result.status == AdapterStatus.rate_limited


# ---------------------------------------------------------------------------
# SSE Streaming Tests
# ---------------------------------------------------------------------------

class TestOpenAIStreaming:
    """Verify OpenAI SSE streaming collects content and usage correctly."""

    def setup_method(self):
        self.adapter = create_openai_adapter(api_key="sk-test")
        self.endpoint = PROVIDER_ENDPOINTS[CloudProvider.OPENAI]

    async def test_multi_chunk_content(self):
        call = make_call(target_alias="gpt4o", prompt="Tell me a story")
        sse = _openai_sse_bytes(
            ["Once", " upon", " a", " time"],
            prompt_tokens=15,
            completion_tokens=20,
        )
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.success
        assert result.content == "Once upon a time"
        assert result.provenance.input_tokens == 15
        assert result.provenance.output_tokens == 20

    async def test_single_chunk(self):
        call = make_call(target_alias="gpt4o", prompt="Hi")
        sse = _openai_sse_bytes(["Hello!"], prompt_tokens=5, completion_tokens=1)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.success
        assert result.content == "Hello!"

    async def test_stream_rate_limit_429(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(429, text="rate limited")
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.rate_limited

    async def test_stream_auth_error_401(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.error
        assert result.error_code == "AUTH_ERROR"

    async def test_stream_server_error_500(self):
        call = make_call(target_alias="gpt4o", prompt="Hello")
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(500, text="Internal Server Error")
            )
            result = await self.adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.error
        assert result.error_code == "API_500"


class TestAnthropicStreaming:
    """Verify Anthropic SSE streaming collects content and usage correctly."""

    def setup_method(self):
        self.adapter = create_anthropic_adapter(api_key="sk-ant-test")
        self.endpoint = PROVIDER_ENDPOINTS[CloudProvider.ANTHROPIC]

    async def test_multi_chunk_content(self):
        call = make_call(target_alias="claude", prompt="Tell me about AI")
        sse = _anthropic_sse_bytes(
            ["Artificial", " intelligence", " is", " fascinating"],
            input_tokens=20,
            output_tokens=30,
        )
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "claude")

        assert result.status == AdapterStatus.success
        assert result.content == "Artificial intelligence is fascinating"
        assert result.provenance.input_tokens == 20
        assert result.provenance.output_tokens == 30

    async def test_single_chunk(self):
        call = make_call(target_alias="claude", prompt="Hi")
        sse = _anthropic_sse_bytes(["Hello!"], input_tokens=5, output_tokens=1)
        with respx.mock(assert_all_called=True) as mock:
            mock.post(self.endpoint).mock(
                return_value=httpx.Response(200, content=sse)
            )
            result = await self.adapter.execute(call, "claude")

        assert result.status == AdapterStatus.success
        assert result.content == "Hello!"


class TestStreamingTimeout:
    """Verify that streaming returns partial content on timeout."""

    async def test_timeout_with_partial_content(self):
        """Simulate timeout mid-stream: adapter should return partial content."""
        adapter = create_openai_adapter(api_key="sk-test")
        endpoint = PROVIDER_ENDPOINTS[CloudProvider.OPENAI]
        call = make_call(target_alias="gpt4o", prompt="Hello")

        from contextlib import asynccontextmanager
        import unittest.mock as um

        class MockResponse:
            status_code = 200
            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"Partial"}}]}'
                yield ''
                yield 'data: {"choices":[{"delta":{"content":" content"}}]}'
                yield ''
                raise httpx.ReadTimeout("read timed out")
            async def aclose(self):
                pass

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockResponse()

        with um.patch.object(httpx.AsyncClient, "stream", mock_stream):
            result = await adapter.execute(call, "gpt4o")

        # ReadTimeout during iteration should propagate up and be caught
        # by the except httpx.TimeoutException handler, returning partial content
        assert result.status == AdapterStatus.soft_timeout
        assert result.content == "Partial content"
        assert result.is_partial is True

    async def test_timeout_with_no_content(self):
        """Timeout before any content arrives should return hard timeout."""
        adapter = create_openai_adapter(api_key="sk-test")
        call = make_call(target_alias="gpt4o", prompt="Hello")

        from contextlib import asynccontextmanager
        import unittest.mock as um

        class MockResponse:
            status_code = 200
            async def aiter_lines(self):
                raise httpx.ReadTimeout("read timed out")
                yield  # make this an async generator
            async def aclose(self):
                pass

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockResponse()

        with um.patch.object(httpx.AsyncClient, "stream", mock_stream):
            result = await adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.timeout
        assert result.content is None


class TestNonStreamingPathPreserved:
    """Verify that the non-streaming code path works correctly when stream=False."""

    async def test_openai_nonstream_success(self):
        config = CloudAdapterConfig(
            adapter_name="openai",
            adapter_version="1.0.0",
            provider=CloudProvider.OPENAI,
            api_key="sk-test",
            stream=False,
            model_mapping={"gpt4o": "gpt-4o"},
            handled_models={"gpt4o"},
        )
        adapter = OpenAIAdapter(config)
        endpoint = PROVIDER_ENDPOINTS[CloudProvider.OPENAI]
        call = make_call(target_alias="gpt4o", prompt="Hello")

        with respx.mock(assert_all_called=True) as mock:
            mock.post(endpoint).mock(
                return_value=httpx.Response(200, json=_openai_success_body("Non-stream reply"))
            )
            result = await adapter.execute(call, "gpt4o")

        assert result.status == AdapterStatus.success
        assert result.content == "Non-stream reply"
        assert result.structured is not None  # Non-streaming path returns structured data
        assert result.provenance.input_tokens == 10
        assert result.provenance.output_tokens == 5

    async def test_anthropic_nonstream_success(self):
        config = CloudAdapterConfig(
            adapter_name="anthropic",
            adapter_version="1.0.0",
            provider=CloudProvider.ANTHROPIC,
            api_key="sk-ant-test",
            stream=False,
            model_mapping={"claude": "claude-sonnet-4-20250514"},
            handled_models={"claude"},
        )
        adapter = AnthropicAdapter(config)
        endpoint = PROVIDER_ENDPOINTS[CloudProvider.ANTHROPIC]
        call = make_call(target_alias="claude", prompt="Hello")

        with respx.mock(assert_all_called=True) as mock:
            mock.post(endpoint).mock(
                return_value=httpx.Response(200, json=_anthropic_success_body("Non-stream Claude"))
            )
            result = await adapter.execute(call, "claude")

        assert result.status == AdapterStatus.success
        assert result.content == "Non-stream Claude"
        assert result.structured is not None

    async def test_google_always_nonstream(self):
        """Google adapter should always use non-streaming path."""
        adapter = create_google_adapter(api_key="google-test-key")
        endpoint = adapter._get_endpoint("gemini-1.5-pro")
        call = make_call(target_alias="gemini", prompt="Hello")

        with respx.mock(assert_all_called=True) as mock:
            mock.post(endpoint).mock(
                return_value=httpx.Response(200, json=_google_success_body("Google reply"))
            )
            result = await adapter.execute(call, "gemini")

        assert result.status == AdapterStatus.success
        assert result.content == "Google reply"
        assert result.structured is not None


class TestStreamingSupport:
    """Verify _supports_streaming() correctly identifies streaming-capable adapters."""

    def test_openai_supports_streaming(self):
        adapter = create_openai_adapter(api_key="sk-test")
        assert adapter._supports_streaming() is True

    def test_anthropic_supports_streaming(self):
        adapter = create_anthropic_adapter(api_key="sk-ant-test")
        assert adapter._supports_streaming() is True

    def test_xai_supports_streaming(self):
        adapter = create_xai_adapter(api_key="xai-test")
        assert adapter._supports_streaming() is True

    def test_groq_supports_streaming(self):
        adapter = create_groq_adapter(api_key="groq-test")
        assert adapter._supports_streaming() is True

    def test_google_does_not_support_streaming(self):
        adapter = create_google_adapter(api_key="google-test")
        # Google has stream=False in config and no _parse_stream_chunk override
        assert adapter._supports_streaming() is False

    def test_cohere_does_not_support_streaming(self):
        """CohereAdapter has no _parse_stream_chunk override."""
        from ControlCore.adapters.cloud import CohereAdapter
        config = CloudAdapterConfig(
            adapter_name="cohere",
            adapter_version="1.0.0",
            provider=CloudProvider.COHERE,
            api_key="co-test",
            stream=False,
        )
        adapter = CohereAdapter(config)
        assert adapter._supports_streaming() is False


class TestParseStreamChunk:
    """Unit tests for _parse_stream_chunk on each provider."""

    def test_openai_extracts_content(self):
        adapter = create_openai_adapter(api_key="sk-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        result = adapter._parse_stream_chunk('data: {"choices":[{"delta":{"content":"Hello"}}]}')
        assert result == "Hello"

    def test_openai_skips_done(self):
        adapter = create_openai_adapter(api_key="sk-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        result = adapter._parse_stream_chunk("data: [DONE]")
        assert result is None

    def test_openai_skips_non_data_lines(self):
        adapter = create_openai_adapter(api_key="sk-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        assert adapter._parse_stream_chunk("event: message") is None
        assert adapter._parse_stream_chunk("") is None
        assert adapter._parse_stream_chunk(": comment") is None

    def test_openai_captures_usage(self):
        adapter = create_openai_adapter(api_key="sk-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        adapter._parse_stream_chunk(
            'data: {"choices":[{}],"usage":{"prompt_tokens":42,"completion_tokens":17}}'
        )
        assert adapter._stream_usage["input_tokens"] == 42
        assert adapter._stream_usage["output_tokens"] == 17

    def test_anthropic_extracts_text_delta(self):
        adapter = create_anthropic_adapter(api_key="sk-ant-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        line = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}'
        result = adapter._parse_stream_chunk(line)
        assert result == "world"

    def test_anthropic_captures_output_tokens(self):
        adapter = create_anthropic_adapter(api_key="sk-ant-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        adapter._parse_stream_chunk(
            'data: {"type":"message_delta","usage":{"output_tokens":25}}'
        )
        assert adapter._stream_usage["output_tokens"] == 25

    def test_anthropic_captures_input_tokens(self):
        adapter = create_anthropic_adapter(api_key="sk-ant-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        adapter._parse_stream_chunk(
            'data: {"type":"message_start","message":{"usage":{"input_tokens":33}}}'
        )
        assert adapter._stream_usage["input_tokens"] == 33

    def test_anthropic_skips_event_lines(self):
        adapter = create_anthropic_adapter(api_key="sk-ant-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        assert adapter._parse_stream_chunk("event: content_block_delta") is None

    def test_openai_compatible_extracts_content(self):
        adapter = create_groq_adapter(api_key="groq-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        result = adapter._parse_stream_chunk('data: {"choices":[{"delta":{"content":"fast"}}]}')
        assert result == "fast"

    def test_xai_extracts_content(self):
        adapter = create_xai_adapter(api_key="xai-test")
        adapter._stream_usage = {"input_tokens": 0, "output_tokens": 0}
        result = adapter._parse_stream_chunk('data: {"choices":[{"delta":{"content":"grok"}}]}')
        assert result == "grok"


# ---------------------------------------------------------------------------
# Factory: create_all_cloud_adapters
# ---------------------------------------------------------------------------

class TestCreateAllCloudAdapters:
    def test_returns_nine_adapters(self):
        adapters = create_all_cloud_adapters()
        assert len(adapters) == 9

    def test_expected_provider_keys(self):
        adapters = create_all_cloud_adapters()
        expected_keys = {
            "openai", "anthropic", "xai", "google",
            "groq", "together", "mistral", "deepseek", "perplexity",
        }
        assert set(adapters.keys()) == expected_keys

    def test_adapter_types(self):
        adapters = create_all_cloud_adapters()
        assert isinstance(adapters["openai"], OpenAIAdapter)
        assert isinstance(adapters["anthropic"], AnthropicAdapter)
        assert isinstance(adapters["google"], GoogleAdapter)
        assert isinstance(adapters["xai"], XAIAdapter)
        # OAI-compatible providers
        for key in ("groq", "together", "mistral", "deepseek", "perplexity"):
            assert isinstance(adapters[key], OpenAICompatibleAdapter), f"{key} should be OpenAICompatibleAdapter"

    def test_all_adapters_have_correct_names(self):
        adapters = create_all_cloud_adapters()
        for key, adapter in adapters.items():
            assert adapter.name == key, f"adapter '{key}' has name '{adapter.name}'"
