"""
Client-side assist layer for normalizing ControlCore calls.

CRITICAL: This module runs CLIENT-SIDE ONLY.
The daemon MUST NEVER execute assist logic.

INVARIANT: All execution inputs are normalized before routing.
Both `ControlCore call` and `ControlCore run` CLI commands use:
  1. assist_normalize_user_input() - fix/augment raw input
  2. validate_candidates_strict() - convert to strict ControlCoreCall

This ensures consistent validation regardless of CLI entry point.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from .schemas import ControlCoreCall, NormalizationReport, IntentClass


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def assist_normalize_user_input(
    raw: Any,
    *,
    allow_variants: bool = True,
    max_variants: int = 5,
) -> Tuple[List[Dict[str, Any]], NormalizationReport]:
    """
    Assist-mode contract.
    Input can be malformed user payload. Output is 1..N candidate ControlCoreCall dicts.
    Core remains strict. This runs before core validation.
    """
    fixes: List[str] = []
    variants: List[Dict[str, Any]] = []

    original_hash = _hash_payload(raw)
    report = NormalizationReport(applied=False, original_payload_hash=original_hash)

    if isinstance(raw, str):
        fixes.append("Interpreted raw string as prompt")
        base = _minimal_call_from_prompt(raw)
        variants.append(base)
        report.applied = True

    elif isinstance(raw, dict):
        if "prompt" not in raw:
            for key in ("query", "message", "text", "input"):
                if key in raw and isinstance(raw[key], str):
                    raw["prompt"] = raw[key]
                    fixes.append(f"Mapped {key} to prompt")
                    report.applied = True
                    break

        if allow_variants and ("intent" not in raw or not raw.get("intent")):
            fixes.append("Generated intent variants due to missing intent")
            variants.extend(_variants_from_raw(raw, max_variants=max_variants))
            report.applied = True
        else:
            variants.append(raw)
            report.applied = report.applied or False

    else:
        fixes.append("Stringified unsupported payload type")
        variants.append(_minimal_call_from_prompt(str(raw)))
        report.applied = True

    report.fixes = fixes
    report.variants_generated = max(0, len(variants) - 1)
    return variants, report


def _minimal_call_from_prompt(prompt: str) -> Dict[str, Any]:
    """Create minimal call dict from just a prompt."""
    return {
        "schema_version": "1.0.0",
        "caller": {"handle": "unknown", "account_id": "00000000-0000-0000-0000-000000000000"},
        "intent": {"class": "unknown"},
        "target": {"type": "model", "alias": "default"},
        "prompt": prompt,
    }


def _variants_from_raw(raw: Dict[str, Any], max_variants: int) -> List[Dict[str, Any]]:
    """Generate intent variants from raw input."""
    prompt = raw.get("prompt") or ""
    base = dict(raw)

    guesses = [IntentClass.lookup, IntentClass.summarize, IntentClass.extract, IntentClass.compare, IntentClass.reason]
    out: List[Dict[str, Any]] = []
    for g in guesses[:max_variants]:
        v = dict(base)
        v["intent"] = {"class": g.value, "detail": f"Assist-mode guess for prompt: {prompt[:80]}"}
        out.append(v)
    return out


def validate_candidates_strict(candidates: List[Dict[str, Any]]) -> List[ControlCoreCall]:
    """
    Convert candidate dicts into strict ControlCoreCall models.
    This is where malformed payloads get rejected. Assist-mode should fix before this.
    """
    calls: List[ControlCoreCall] = []
    for c in candidates:
        calls.append(ControlCoreCall.model_validate(c))
    return calls
