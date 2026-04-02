"""
Tests for ControlCore.redaction.redact_text

API: redact_text(text: str) -> Tuple[str, RedactionReport]

RedactionReport fields: performed, items, user_notice, override_used
RedactionReportItem fields: kind, count, note (optional)
"""

from __future__ import annotations

import pytest

from ControlCore.redaction import redact_text
from ControlCore.schemas import RedactionReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kinds_found(report: RedactionReport) -> set[str]:
    return {item.kind for item in report.items}


def count_for(report: RedactionReport, kind: str) -> int:
    for item in report.items:
        if item.kind == kind:
            return item.count
    return 0


# ---------------------------------------------------------------------------
# Clean text — no redaction
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_text_passes_through_unchanged(self):
        text = "The weather today is sunny with a high of 72 degrees."
        result, report = redact_text(text)
        assert result == text

    def test_clean_text_report_not_performed(self):
        _, report = redact_text("Hello, world!")
        assert report.performed is False

    def test_clean_text_report_no_items(self):
        _, report = redact_text("No secrets here.")
        assert report.items == []

    def test_clean_text_no_user_notice(self):
        _, report = redact_text("Plain text only.")
        assert report.user_notice is None

    def test_empty_string_passes_through(self):
        result, report = redact_text("")
        assert result == ""
        assert report.performed is False


# ---------------------------------------------------------------------------
# API key patterns  (sk_*, rk_*, pk_*)
# ---------------------------------------------------------------------------

class TestApiKeyRedaction:
    # Pattern: (sk|rk|pk)_[A-Za-z0-9]{16,}
    # Sub-prefix forms like sk_live_... are NOT matched because `live_` breaks
    # the required 16-char alphanumeric run after the single underscore.
    # Valid keys look like: sk_abcdef1234567890AB (no second underscore).

    def test_redacts_sk_key(self):
        text = "Use sk_abcdef1234567890AB to authenticate."
        result, report = redact_text(text)
        assert "sk_abcdef1234567890AB" not in result
        assert "[REDACTED:api_key]" in result

    def test_redacts_rk_key(self):
        text = "Restricted key: rk_ABCDEF1234567890XY"
        result, report = redact_text(text)
        assert "rk_ABCDEF1234567890XY" not in result
        assert "[REDACTED:api_key]" in result

    def test_redacts_pk_key(self):
        text = "Public key: pk_abcdefghijklmnop"
        result, report = redact_text(text)
        assert "pk_abcdefghijklmnop" not in result
        assert "[REDACTED:api_key]" in result

    def test_api_key_report_performed(self):
        _, report = redact_text("sk_abcdef1234567890AB was found.")
        assert report.performed is True

    def test_api_key_report_kind(self):
        _, report = redact_text("sk_abcdef1234567890AB")
        assert "api_key" in kinds_found(report)

    def test_api_key_count_multiple(self):
        text = "First: sk_1234567890abcdefAA Second: rk_1234567890abcdefBB"
        _, report = redact_text(text)
        assert count_for(report, "api_key") == 2

    def test_short_key_not_redacted(self):
        # Pattern requires 16+ alphanumeric chars after the underscore
        text = "sk_tooshort"
        result, _ = redact_text(text)
        assert result == text

    def test_subprefix_key_not_matched_by_pattern(self):
        # sk_live_... has a second underscore before the 16-char run,
        # so the regex does NOT match it — this documents the limitation.
        text = "sk_live_abcdef1234567890"
        result, report = redact_text(text)
        assert report.performed is False


# ---------------------------------------------------------------------------
# Bearer / token patterns
# ---------------------------------------------------------------------------

class TestTokenRedaction:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result, report = redact_text(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[REDACTED:token]" in result

    def test_redacts_token_prefix(self):
        text = "token abcdef1234567890ABCDEF"
        result, report = redact_text(text)
        assert "[REDACTED:token]" in result

    def test_bearer_case_insensitive(self):
        text = "BEARER MySecretToken1234567890"
        result, report = redact_text(text)
        assert "[REDACTED:token]" in result

    def test_token_report_kind(self):
        _, report = redact_text("Bearer abcdef1234567890")
        assert "token" in kinds_found(report)

    def test_token_report_performed(self):
        _, report = redact_text("Bearer abcdef1234567890")
        assert report.performed is True


# ---------------------------------------------------------------------------
# Email addresses
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    def test_redacts_simple_email(self):
        text = "Contact us at support@example.com for help."
        result, report = redact_text(text)
        assert "support@example.com" not in result
        assert "[REDACTED:email]" in result

    def test_redacts_subdomain_email(self):
        text = "admin@mail.company.org"
        result, report = redact_text(text)
        assert "admin@mail.company.org" not in result
        assert "[REDACTED:email]" in result

    def test_email_report_kind(self):
        _, report = redact_text("user@domain.io")
        assert "email" in kinds_found(report)

    def test_email_report_performed(self):
        _, report = redact_text("user@domain.io")
        assert report.performed is True

    def test_email_count_multiple(self):
        text = "From: alice@example.com, To: bob@example.com"
        _, report = redact_text(text)
        assert count_for(report, "email") == 2

    def test_email_with_plus_addressing(self):
        text = "Reply to alice+filter@example.com"
        result, _ = redact_text(text)
        assert "alice+filter@example.com" not in result


# ---------------------------------------------------------------------------
# Multiple patterns in one string
# ---------------------------------------------------------------------------

class TestMultiplePatterns:
    def test_api_key_and_email_together(self):
        text = "Key: sk_abcdef1234567890AB Email: admin@example.com"
        result, report = redact_text(text)
        assert "sk_abcdef1234567890AB" not in result
        assert "admin@example.com" not in result
        assert "[REDACTED:api_key]" in result
        assert "[REDACTED:email]" in result
        assert report.performed is True
        assert "api_key" in kinds_found(report)
        assert "email" in kinds_found(report)

    def test_bearer_and_email_together(self):
        text = "Token: Bearer abc123def456ghi789 Contact: ops@company.net"
        result, report = redact_text(text)
        assert "[REDACTED:token]" in result
        assert "[REDACTED:email]" in result
        assert len(report.items) == 2

    def test_all_three_api_key_token_email(self):
        text = (
            "sk_abcdef1234567890AB "
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 "
            "user@example.com"
        )
        result, report = redact_text(text)
        kinds = kinds_found(report)
        assert "api_key" in kinds
        assert "token" in kinds
        assert "email" in kinds
        assert report.performed is True

    def test_user_notice_set_when_redaction_occurs(self):
        _, report = redact_text("admin@example.com sk_live_abcdef1234567890")
        assert report.user_notice is not None
        assert len(report.user_notice) > 0


# ---------------------------------------------------------------------------
# Report structure invariants
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_override_used_always_false(self):
        """redact_text never sets override_used; that is set by the executor."""
        _, report = redact_text("sk_live_abcdef1234567890")
        assert report.override_used is False

    def test_report_is_redaction_report_instance(self):
        _, report = redact_text("anything")
        assert isinstance(report, RedactionReport)

    def test_item_counts_are_positive(self):
        text = "email@example.com sk_live_abcdef1234567890"
        _, report = redact_text(text)
        for item in report.items:
            assert item.count >= 1
