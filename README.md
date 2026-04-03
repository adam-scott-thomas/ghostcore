# ghostrouter

The LLM router that learns. Intelligent routing across 10+ providers, with fallback, circuit breakers, budget tracking, and post-model redaction.

## Why

Every project hardcodes model names and API clients. When a model is slow, rate-limited, or refuses a request, the call just fails. ghostrouter routes each call to the best available model automatically — with circuit breakers, fallback chains, and post-model redaction baked in.

## Install

```bash
pip install ghostrouter
```

## Quick Start

```python
from ControlCore.config import initialize_controlcore
from ControlCore.adapters.executor import execute_call
from ControlCore.schemas import ControlCoreCall, CallerIdentity, CallIntent, CallTarget

# Initialize registries (reads env vars for API keys)
config, model_registry, adapter_registry = initialize_controlcore()

# Build a call
call = ControlCoreCall(
    caller=CallerIdentity(handle="my-app", account_id="00000000-0000-0000-0000-000000000000"),
    intent=CallIntent(cls="lookup"),
    target=CallTarget(type="model", alias="claude"),
    prompt="What is the capital of France?",
)

# Execute with automatic fallback
import asyncio
result, trace = asyncio.run(execute_call(call, model_registry, adapter_registry))
print(result.answer)
```

Or boot as a daemon and call over HTTP:

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## Architecture

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **Routing** — scores eligible models by trust tier, cost, and latency history
2. **Eligibility** — filters models by intent, verbosity, and determinism requirements
3. **Fallback** — tries models in order; switches on timeout, error, refusal, or rate limit
4. **Adapter** — thin shim per provider; no shared state between providers
5. **Redaction** — applied to model output (not the prompt); strips leaked secrets, emails, phone numbers

## Providers

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4, GPT-4o, o1, o1-mini |
| Anthropic | Claude Sonnet, Opus, Haiku |
| Google | Gemini 1.5 Pro/Flash, Gemini 2.0 |
| xAI | Grok, Grok-2 |
| Mistral | Mistral Large/Medium/Small, Codestral |
| Groq | Llama-70B, Llama-8B, Mixtral, Gemma |
| Together | Llama-405B, Qwen-72B, DeepSeek-V3 |
| DeepSeek | DeepSeek Chat, Coder, Reasoner |
| Perplexity | Sonar (search-augmented) |
| Ollama | Any local model |

Set the relevant `*_API_KEY` environment variables to enable each provider.

## Key Features

- **Circuit breakers** — automatically open on repeated failures, recover with half-open probing
- **Trust tiers** — models tagged with trust levels; calls can require a minimum tier
- **Post-model redaction** — output is scanned and sanitized before returning to caller
- **Execution traces** — every call records which models were tried, timings, and outcomes
- **Structured schemas** — Pydantic v2 throughout; strict validation at every boundary
- **Async-native** — built on httpx + asyncio; runs in Starlette for low overhead

## Spine Integration (optional)

ghostrouter supports [maelspine](https://github.com/adam-scott-thomas/maelspine) for zero-import access to registries across a larger application:

```python
from ControlCore.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/call` | Submit a call |
| GET | `/result/{job_id}` | Poll for result |
| GET | `/health` | Health + job stats |
| GET | `/jobs` | List recent jobs |

## Part of the GhostLogic Stack

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## License

MIT — Copyright 2026 Adam Thomas / GhostLogic LLC
