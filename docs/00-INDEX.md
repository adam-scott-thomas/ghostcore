# ControlCore Architecture - FINAL DESIGN

> **STATUS: FINAL COPY - IDEALIZED ARCHITECTURE**
> This represents the complete designed architecture, not necessarily what is currently implemented.

#architecture #final-copy

## Layers

1. [[01-CLI-Layer]] - Client entry point
2. [[02-Daemon-Layer]] - HTTP server and validation
3. [[03-Registry-Layer]] - Model selection and routing
4. [[04-Execution-Layer]] - Adapter execution
5. [[05-Observability-Layer]] - Tracing and metrics

## Quick Links

- [[ControlCoreCall-Schema]]
- [[ExecutionEngine]]
- [[CircuitBreaker]]

## Z-State Model

- **Z0**: Pure/Reversible (validation, routing)
- **Z1**: Bounded/In-Memory (job registry, circuit breaker)
- **Z2**: Irreversible/External (HTTP, subprocess)
