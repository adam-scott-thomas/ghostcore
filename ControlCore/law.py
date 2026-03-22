"""Phase P0 call law enforcement."""

from __future__ import annotations

from typing import Tuple, List
from .schemas import ControlCoreCall, CallError, ErrorCode, TargetType, RedactionMode, Determinism


OVERRIDE_PHRASES_REQUIRED = {
    "INCLUDE_SENSITIVE_DATA",
    "NO_REDACTION_ACKNOWLEDGED",
    "I_UNDERSTAND_AND_ACCEPT_RISK",
}


def enforce_call_law(call: ControlCoreCall) -> Tuple[bool, List[CallError]]:
    """
    Enforce call law on a ControlCoreCall.

    Returns (is_valid, errors).
    """
    errors: List[CallError] = []

    # Law: Tools must be read-only. We only enforce type here; adapter layer must enforce read-only.
    if call.target.type == TargetType.tool:
        pass

    # Law: Redaction off requires explicit override acknowledgements
    if call.options.redaction.mode == RedactionMode.off:
        ov = call.options.redaction.override
        if not ov or not ov.enabled:
            errors.append(CallError(code=ErrorCode.permission_denied, message="Redaction off requires override enabled"))
        else:
            acks = set(ov.acknowledgements or [])
            missing = sorted(list(OVERRIDE_PHRASES_REQUIRED - acks))
            if missing:
                errors.append(
                    CallError(
                        code=ErrorCode.permission_denied,
                        message="Missing required redaction override acknowledgements",
                        details={"missing": missing},
                    )
                )

    # Law: Determinism strict requires a seed (best practice for reproducibility)
    if call.options.determinism == Determinism.strict and call.params.seed is None:
        errors.append(
            CallError(
                code=ErrorCode.validation_error,
                message="determinism strict requires params.seed",
            )
        )

    return (len(errors) == 0), errors
