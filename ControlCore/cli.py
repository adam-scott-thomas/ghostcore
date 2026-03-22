"""
ControlCore CLI - Command-line interface for the ControlCore daemon.

Subcommands:
- serve: Start the daemon
- call: Submit a call to the daemon
- result: Get result for a job
- run: Sugar mode shorthand
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import httpx

from ControlCore.schemas import Verbosity, Determinism, IntentClass, CallStatus
from ControlCore.normalize import assist_normalize_user_input, validate_candidates_strict
from ControlCore.bouncer import OVERRIDE_PHRASES_REQUIRED


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8265


def get_base_url(host: str, port: int) -> str:
    """Get base URL for daemon."""
    return f"http://{host}:{port}"


def format_error(error: dict[str, Any]) -> str:
    """Format an error for display."""
    code = error.get("code", "unknown")
    message = error.get("message", "Unknown error")
    details = error.get("details", {})

    result = f"[{code}] {message}"
    if details:
        result += f"\n  Details: {json.dumps(details, indent=2)}"
    return result


def format_result(result: dict[str, Any]) -> str:
    """Format a result for display."""
    lines = []

    lines.append(f"Call ID: {result.get('call_id', 'N/A')}")
    lines.append(f"Status: {result.get('status', 'N/A')}")
    if result.get("job_id"):
        lines.append(f"Job ID: {result['job_id']}")
    lines.append("")

    if result.get("answer"):
        lines.append("Answer:")
        lines.append("-" * 40)
        lines.append(result["answer"])
        lines.append("-" * 40)

    if result.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for err in result["errors"]:
            lines.append(f"  - {format_error(err)}")

    # Provenance
    prov = result.get("provenance", {})
    lines.append("")
    lines.append("Provenance:")
    lines.append(f"  Model: {prov.get('model_alias', 'N/A')}")
    lines.append(f"  Trust Tier: {prov.get('trust_tier', 'N/A')}")
    lines.append(f"  Started: {prov.get('started_at', 'N/A')}")

    # Redaction
    redaction = result.get("redaction", {})
    if redaction.get("performed"):
        lines.append("")
        lines.append("Redaction:")
        lines.append(f"  Performed: {redaction.get('performed')}")
        for item in redaction.get("items", []):
            lines.append(f"    - {item['kind']}: {item['count']}")

    return "\n".join(lines)


@click.group()
@click.version_option(version="0.1.0", prog_name="ControlCore")
def main():
    """ControlCore CLI - Structured LLM call orchestration."""
    pass


@main.command()
@click.option("--host", default=DEFAULT_HOST, help="Host to bind to")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port to bind to")
def serve(host: str, port: int):
    """Start the ControlCore daemon."""
    click.echo(f"Starting ControlCore daemon on {host}:{port}")
    click.echo("Press Ctrl+C to stop")

    from ControlCore.daemon import run_server
    run_server(host=host, port=port)


@main.command()
@click.option("--host", default=DEFAULT_HOST, help="Daemon host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Daemon port")
@click.option("--target", "-t", required=True, help="Target model alias")
@click.option("--prompt", "-p", required=True, help="Prompt to send")
@click.option("--intent", "-i", type=click.Choice([e.value for e in IntentClass]), default="unknown")
@click.option("--verbosity", "-v", type=click.Choice([e.value for e in Verbosity]), default="standard")
@click.option("--determinism", "-d", type=click.Choice([e.value for e in Determinism]), default="best_effort")
@click.option("--timeout", default=15000, type=int, help="Soft timeout in milliseconds")
@click.option("--handle", default="cli_user", help="Caller handle")
@click.option("--account-id", default="00000000-0000-0000-0000-000000000000", help="Account ID")
@click.option("--json-output", is_flag=True, help="Output raw JSON")
def call(
    host: str,
    port: int,
    target: str,
    prompt: str,
    intent: str,
    verbosity: str,
    determinism: str,
    timeout: int,
    handle: str,
    account_id: str,
    json_output: bool,
):
    """Submit a call to the ControlCore daemon."""
    base_url = get_base_url(host, port)

    # Build raw payload
    raw = {
        "caller": {"handle": handle, "account_id": account_id},
        "intent": {"class": intent},
        "target": {"type": "model", "alias": target},
        "prompt": prompt,
        "options": {
            "verbosity": verbosity,
            "determinism": determinism,
            "timeouts": {"soft_ms": timeout},
        },
    }

    # Normalize via canonical path (same as `run` command)
    # Invariant: all execution inputs are normalized before routing
    candidates, norm_report = assist_normalize_user_input(raw, allow_variants=False, max_variants=1)

    if not candidates:
        click.echo("Error: Failed to normalize input", err=True)
        sys.exit(1)

    try:
        calls = validate_candidates_strict(candidates)
        payload = calls[0].model_dump(mode="json", by_alias=True)
    except Exception as e:
        click.echo(f"Error: Failed to compile call: {e}", err=True)
        sys.exit(1)

    try:
        with httpx.Client(timeout=timeout / 1000 + 30) as client:
            response = client.post(f"{base_url}/call", json=payload)

        data = response.json()

        if json_output:
            click.echo(json.dumps(data, indent=2))
        elif response.status_code >= 400:
            if "errors" in data:
                click.echo("Errors:", err=True)
                for error in data["errors"]:
                    click.echo(f"  - {format_error(error)}", err=True)
            sys.exit(1)
        else:
            click.echo(format_result(data))

    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to daemon at {base_url}", err=True)
        click.echo("Is the daemon running? Start it with: ControlCore serve", err=True)
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo("Error: Request timed out", err=True)
        sys.exit(1)


@main.command()
@click.argument("job_id")
@click.option("--host", default=DEFAULT_HOST, help="Daemon host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Daemon port")
@click.option("--json-output", is_flag=True, help="Output raw JSON")
@click.option("--poll", is_flag=True, help="Poll until complete")
@click.option("--poll-interval", default=1.0, type=float, help="Poll interval in seconds")
def result(job_id: str, host: str, port: int, json_output: bool, poll: bool, poll_interval: float):
    """Get result for a job."""
    import time

    base_url = get_base_url(host, port)

    try:
        with httpx.Client(timeout=30) as client:
            while True:
                response = client.get(f"{base_url}/result/{job_id}")
                data = response.json()

                if response.status_code == 404:
                    click.echo(f"Error: Job not found: {job_id}", err=True)
                    sys.exit(1)

                if response.status_code >= 400:
                    if "errors" in data:
                        for err in data["errors"]:
                            click.echo(format_error(err), err=True)
                    sys.exit(1)

                status = data.get("status", "")

                if poll and status in ("queued", "running"):
                    click.echo(f"Status: {status}, polling in {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue

                if json_output:
                    click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(format_result(data))
                break

    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to daemon at {base_url}", err=True)
        sys.exit(1)


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--host", default=DEFAULT_HOST, help="Daemon host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Daemon port")
@click.option("--json-output", is_flag=True, help="Output raw JSON")
def run(model: str, prompt: str, host: str, port: int, json_output: bool):
    """
    Sugar mode: ControlCore run <model> "<prompt>"

    Uses client-side assist layer to compile down to strict ControlCoreCall.
    """
    base_url = get_base_url(host, port)

    # Client-side assist layer (never runs in daemon)
    raw = {
        "prompt": prompt,
        "target": {"type": "model", "alias": model},
        "caller": {"handle": "cli_user", "account_id": "00000000-0000-0000-0000-000000000000"},
    }

    candidates, norm_report = assist_normalize_user_input(raw, allow_variants=False, max_variants=1)

    if not candidates:
        click.echo("Error: Failed to normalize input", err=True)
        sys.exit(1)

    try:
        calls = validate_candidates_strict(candidates)
        payload = calls[0].model_dump(mode="json", by_alias=True)
    except Exception as e:
        click.echo(f"Error: Failed to compile call: {e}", err=True)
        sys.exit(1)

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(f"{base_url}/call", json=payload)

        data = response.json()

        if json_output:
            click.echo(json.dumps(data, indent=2))
        elif response.status_code >= 400:
            if "errors" in data:
                click.echo("Errors:", err=True)
                for error in data["errors"]:
                    click.echo(f"  - {format_error(error)}", err=True)
            sys.exit(1)
        else:
            click.echo(format_result(data))

    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to daemon at {base_url}", err=True)
        click.echo("Is the daemon running? Start it with: ControlCore serve", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default=DEFAULT_HOST, help="Daemon host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Daemon port")
def health(host: str, port: int):
    """Check daemon health."""
    base_url = get_base_url(host, port)

    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{base_url}/health")

        data = response.json()
        click.echo(f"Status: {data.get('status', 'unknown')}")
        click.echo(f"Version: {data.get('version', 'unknown')}")
        click.echo(f"Started: {data.get('started_at', 'unknown')}")

        jobs = data.get("jobs", {})
        click.echo(f"Jobs: {jobs.get('total_jobs', 0)} total")
        by_status = jobs.get("by_status", {})
        for status, count in by_status.items():
            if count > 0:
                click.echo(f"  {status}: {count}")

    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to daemon at {base_url}", err=True)
        click.echo("Is the daemon running? Start it with: ControlCore serve", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default=DEFAULT_HOST, help="Daemon host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Daemon port")
@click.option("--status", "-s", type=click.Choice([e.value for e in CallStatus]), help="Filter by status")
@click.option("--limit", "-l", default=10, type=int, help="Max results")
def jobs(host: str, port: int, status: str | None, limit: int):
    """List jobs."""
    base_url = get_base_url(host, port)

    params = {"limit": str(limit)}
    if status:
        params["status"] = status

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{base_url}/jobs", params=params)

        data = response.json()

        if response.status_code >= 400:
            if "errors" in data:
                for err in data["errors"]:
                    click.echo(format_error(err), err=True)
            sys.exit(1)

        jobs_list = data.get("jobs", [])
        click.echo(f"Jobs ({data.get('count', 0)} returned):")
        click.echo("-" * 60)

        for job in jobs_list:
            click.echo(f"  {job['job_id']} | {job['status']} | {job['created_at']}")

    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to daemon at {base_url}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
