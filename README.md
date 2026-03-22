# ControlCore

Structured LLM call orchestration daemon and CLI.

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Start the daemon

```bash
ControlCore serve
```

The daemon binds to `localhost:8265` by default.

### Submit a call (sugar mode)

```bash
ControlCore run claude "What is kubernetes?"
```

### Submit a call (explicit mode)

```bash
ControlCore call \
  --target claude \
  --prompt "Explain recursion" \
  --intent lookup \
  --verbosity standard \
  --determinism best_effort \
  --timeout 15000
```

### Get result

```bash
ControlCore result <job_id>

# Poll until complete
ControlCore result <job_id> --poll
```

### Check health

```bash
ControlCore health
```

### List jobs

```bash
ControlCore jobs
ControlCore jobs --status complete --limit 10
```

## CLI Flags

| Flag | Schema Field | Values |
|------|--------------|--------|
| `--target`, `-t` | `target.alias` | Model alias |
| `--prompt`, `-p` | `prompt` | The prompt text |
| `--intent`, `-i` | `intent.class` | `lookup`, `summarize`, `extract`, `compare`, `draft`, `classify`, `reason`, `critique`, `translate`, `unknown` |
| `--verbosity`, `-v` | `options.verbosity` | `minimal`, `standard`, `full` |
| `--determinism`, `-d` | `options.determinism` | `strict`, `best_effort`, `off` |
| `--timeout` | `options.timeouts.soft_ms` | Timeout in milliseconds |

## API Endpoints

### POST /call

Submit a call for processing.

**Request:**
```json
{
  "caller": {"handle": "user", "account_id": "00000000-0000-0000-0000-000000000000"},
  "intent": {"class": "lookup"},
  "target": {"type": "model", "alias": "claude"},
  "prompt": "Hello world",
  "options": {
    "verbosity": "standard",
    "determinism": "best_effort",
    "timeouts": {"soft_ms": 15000, "hard_ms": 60000}
  }
}
```

**Response (200):**
```json
{
  "schema_version": "1.0.0",
  "call_id": "uuid",
  "status": "complete",
  "answer": "Response content",
  "provenance": {
    "model_alias": "claude",
    "trust_tier": "standard",
    "started_at": "2024-01-01T00:00:00Z"
  },
  "redaction": {"performed": false, "items": []},
  "errors": []
}
```

**Response (202 - Queued):**
```json
{
  "call_id": "uuid",
  "job_id": "uuid",
  "status": "running",
  "message": "Job queued for processing. Poll /result/{job_id} for results."
}
```

### GET /result/{job_id}

Get result for a job.

### GET /health

Health check with job statistics.

### GET /jobs

List jobs with optional status filter.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   CLI       │    │ Assist/     │    │   HTTP      │     │
│  │ (ControlCore)  │───▶│ Normalize   │───▶│   Client    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         DAEMON                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   HTTP      │    │  Validation │    │  Redaction  │     │
│  │   Server    │───▶│  (strict)   │───▶│  Pipeline   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          ▼                                  │
│                  ┌─────────────┐                           │
│                  │ Job Registry│                           │
│                  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** Assist/normalize logic runs CLIENT-SIDE ONLY. The daemon enforces strict validation.

## Redaction

By default, sensitive content is auto-redacted:
- API keys (`sk_*`, `rk_*`, `pk_*`)
- Bearer tokens
- Email addresses
- Phone numbers

To disable redaction, you must explicitly acknowledge the risk with all three phrases:
- `INCLUDE_SENSITIVE_DATA`
- `NO_REDACTION_ACKNOWLEDGED`
- `I_UNDERSTAND_AND_ACCEPT_RISK`

## Error Types

| Code | Description |
|------|-------------|
| `validation_error` | Input validation failed |
| `permission_denied` | Operation not permitted |
| `adapter_error` | Adapter/model error |
| `timeout` | Operation timed out |
| `refused` | Request refused |
| `unknown` | Unknown error |

## Testing

```bash
pytest
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run daemon in development
ControlCore serve --host 127.0.0.1 --port 8265
```
