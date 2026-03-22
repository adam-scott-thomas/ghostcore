"""
Redaction pipeline for sensitive content.

REDACTION STAGE: POST-MODEL ONLY

Rationale:
- Pre-model redaction would modify the prompt before the model sees it,
  potentially breaking the user's intent or causing confusion.
- Post-model redaction sanitizes the model's OUTPUT before returning
  to the user, protecting against accidental PII/credential leakage.

Enforcement:
- Redaction is applied in executor._build_success_result() AFTER
  the adapter returns content from the model.
- Controlled by call.options.redaction.mode:
  - "auto" (default): redaction is applied
  - "off": redaction is skipped (requires explicit override ack)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple
from .schemas import RedactionReport, RedactionReportItem


# Lightweight, demo-safe detectors. Extend later.
API_KEY_PATTERNS = [
    ("api_key", re.compile(r"\b(sk|rk|pk)_[A-Za-z0-9]{16,}\b")),
    ("token", re.compile(r"\b(?:token|bearer)\s+[A-Za-z0-9\-\._~\+\/]+=*\b", re.IGNORECASE)),
]
PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]


def redact_text(text: str) -> Tuple[str, RedactionReport]:
    """
    Redact sensitive content from text.

    Returns (redacted_text, report).
    """
    items: List[RedactionReportItem] = []
    redacted = text

    def apply(kind: str, pattern: re.Pattern, s: str) -> Tuple[str, int]:
        matches = list(pattern.finditer(s))
        if not matches:
            return s, 0
        return pattern.sub(f"[REDACTED:{kind}]", s), len(matches)

    for kind, pat in API_KEY_PATTERNS + PII_PATTERNS:
        redacted, count = apply(kind, pat, redacted)
        if count:
            items.append(RedactionReportItem(kind=kind, count=count))

    report = RedactionReport(
        performed=len(items) > 0,
        items=items,
        user_notice="Sensitive data was detected and redacted. Use explicit override acknowledgements to include it."
        if len(items) > 0
        else None,
        override_used=False,
    )
    return redacted, report
