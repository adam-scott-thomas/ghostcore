"""
ControlCore daemon - HTTP server for processing LLM calls.

CRITICAL: The daemon MUST NEVER run assist logic.
All input must be strictly validated ControlCoreCall payloads.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ControlCore.schemas import (
    ControlCoreCall,
    CallStatus,
    CallError,
    ErrorCode,
)
from ControlCore.bouncer import enforce_bouncer
from ControlCore.job_registry import get_registry
from ControlCore.adapters.executor import (
    execute_call,
    AdapterRegistry,
    create_stub_adapter_registry,
)
from ControlCore.registry.loader import (
    get_global_registry,
    set_global_registry,
    load_registry_from_dict,
)
from ControlCore.registry.schema import ModelRegistry


# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ControlCore.daemon")

# Soft timeout for immediate response (ms)
SOFT_TIMEOUT_MS = 5000

# Daemon state
_daemon_state: dict[str, Any] = {
    "started_at": None,
    "shutting_down": False,
    "model_registry": None,
    "adapter_registry": None,
}


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")


def json_response(data: dict[str, Any], status_code: int = 200) -> JSONResponse:
    """Create a JSON response with proper serialization."""
    return JSONResponse(
        content=json.loads(json.dumps(data, default=json_serial)),
        status_code=status_code,
    )


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    registry = get_registry()
    stats = registry.stats()

    return json_response({
        "status": "healthy" if not _daemon_state["shutting_down"] else "shutting_down",
        "version": "0.1.0",
        "started_at": _daemon_state["started_at"],
        "jobs": stats,
    })


async def post_call(request: Request) -> JSONResponse:
    """
    POST /call - Submit a ControlCore call.

    CRITICAL: No assist logic. Strict validation only.
    """
    log = logger.bind(endpoint="POST /call")

    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        log.warning("invalid_json", error=str(e))
        return json_response({
            "errors": [{"code": "validation_error", "message": f"Invalid JSON: {e}"}],
        }, status_code=400)

    # Strict validation
    try:
        call = ControlCoreCall.model_validate(body)
    except Exception as e:
        log.warning("validation_failed", error=str(e))
        return json_response({
            "errors": [{"code": "validation_error", "message": str(e)}],
        }, status_code=400)

    # Bouncer enforcement
    ok, bouncer_errors = enforce_bouncer(call)
    if not ok:
        log.warning("bouncer_violation", error_count=len(bouncer_errors))
        return json_response({
            "errors": [e.model_dump() for e in bouncer_errors],
        }, status_code=400)

    # Create job
    registry = get_registry()
    job = registry.create_job(call)
    log.info("job_created", job_id=job.job_id, target=call.target.alias)

    # Mark job as running
    start_time = time.time()
    registry.mark_running(job.job_id)

    # Execute call via executor
    model_registry = _daemon_state["model_registry"]
    adapter_registry = _daemon_state["adapter_registry"]

    try:
        exec_result, trace = await execute_call(
            call,
            model_registry=model_registry,
            adapter_registry=adapter_registry,
        )
    except Exception as e:
        log.error("execution_error", job_id=job.job_id, error=str(e))
        registry.mark_failed(job.job_id, [
            CallError(code=ErrorCode.unknown, message=f"Execution error: {e}")
        ])
        result = registry.get_job_result(job.job_id)
        return json_response(result.model_dump(mode="json"), status_code=500)

    elapsed_ms = (time.time() - start_time) * 1000

    # Update job based on execution result
    if exec_result.status == CallStatus.complete:
        registry.mark_complete(
            job.job_id,
            exec_result.answer or "",
            exec_result.redaction,
        )
        log.info("job_complete", job_id=job.job_id, elapsed_ms=elapsed_ms)
    elif exec_result.status == CallStatus.failed:
        registry.mark_failed(job.job_id, exec_result.errors or [])
        log.warning("job_failed", job_id=job.job_id, elapsed_ms=elapsed_ms)
    elif exec_result.status == CallStatus.queued:
        log.info("job_queued", job_id=job.job_id, elapsed_ms=elapsed_ms)
        return json_response({
            "call_id": call.call_id,
            "job_id": job.job_id,
            "status": CallStatus.running.value,
            "message": "Job queued for processing. Poll /result/{job_id} for results.",
        }, status_code=202)

    result = registry.get_job_result(job.job_id)
    status_code = 200 if exec_result.status == CallStatus.complete else 500

    return json_response(result.model_dump(mode="json"), status_code=status_code)


async def get_result(request: Request) -> JSONResponse:
    """GET /result/{job_id} - Get result for a job."""
    log = logger.bind(endpoint="GET /result")

    job_id = request.path_params.get("job_id", "")

    registry = get_registry()
    result = registry.get_job_result(job_id)

    if result is None:
        log.warning("job_not_found", job_id=job_id)
        return json_response({
            "errors": [{"code": "validation_error", "message": f"Job not found: {job_id}"}],
        }, status_code=404)

    log.info("result_retrieved", job_id=job_id, status=result.status.value)
    return json_response(result.model_dump(mode="json"))


async def list_jobs(request: Request) -> JSONResponse:
    """GET /jobs - List all jobs."""
    registry = get_registry()

    status_filter = request.query_params.get("status")
    limit = int(request.query_params.get("limit", "100"))

    status = None
    if status_filter:
        try:
            status = CallStatus(status_filter)
        except ValueError:
            return json_response({
                "errors": [{"code": "validation_error", "message": f"Invalid status: {status_filter}"}],
            }, status_code=400)

    jobs = registry.list_jobs(status=status, limit=limit)
    return json_response({
        "jobs": jobs,
        "count": len(jobs),
    })


def create_app() -> Starlette:
    """Create the Starlette application."""

    @asynccontextmanager
    async def lifespan(app: Starlette):
        _daemon_state["started_at"] = datetime.utcnow().isoformat() + "Z"
        _daemon_state["shutting_down"] = False

        # Initialize model registry (use global if set, else create empty)
        model_registry = get_global_registry()
        if model_registry is None:
            model_registry = load_registry_from_dict({"version": "1.0.0", "models": []})
            set_global_registry(model_registry)
        _daemon_state["model_registry"] = model_registry

        # Initialize adapter registry with stub adapters
        _daemon_state["adapter_registry"] = create_stub_adapter_registry()

        logger.info("daemon_started", version="0.1.0", models=len(model_registry))
        yield
        _daemon_state["shutting_down"] = True
        logger.info("daemon_shutdown")

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/call", post_call, methods=["POST"]),
        Route("/result/{job_id}", get_result, methods=["GET"]),
        Route("/jobs", list_jobs, methods=["GET"]),
    ]

    return Starlette(debug=False, routes=routes, lifespan=lifespan)


def run_server(host: str = "127.0.0.1", port: int = 8265) -> None:
    """Run the daemon server."""
    import uvicorn

    def handle_signal(signum, frame):
        logger.info("signal_received", signal=signum)
        _daemon_state["shutting_down"] = True
        sys.exit(0)

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGHUP, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("starting_server", host=host, port=port)

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    run_server()
