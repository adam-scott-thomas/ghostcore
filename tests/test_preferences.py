"""
Tests for ControlCore/registry/preferences.py

Covers:
1. Task affinity boost matches intent + model
2. Caller blocklist blocks correct models
3. Caller preferred returns correct model
4. No preferences returns defaults (0 boost, not blocked, no preferred)
5. Wildcard intent affinity matches all intents
"""

from __future__ import annotations

import pytest

from ControlCore.registry.preferences import AffinityRule, Preferences


# ---------------------------------------------------------------------------
# Test 1: Affinity boost matches on correct intent + model
# ---------------------------------------------------------------------------

def test_affinity_boost_matches_intent_and_model():
    """Boost is returned when both intent and model_alias match exactly."""
    prefs = Preferences(
        affinities=[
            AffinityRule(intent="summarize", model_alias="qwen:32b", boost=15.0),
        ]
    )

    assert prefs.get_boost("qwen:32b", intent="summarize") == 15.0


# ---------------------------------------------------------------------------
# Test 2: Caller blocklist blocks the correct models
# ---------------------------------------------------------------------------

def test_caller_blocklist_blocks_correct_models():
    """is_blocked returns True for blocked models and False for others."""
    prefs = Preferences(
        caller_blocklists={
            "alice": ["gpt-4o", "claude:opus"],
        }
    )

    assert prefs.is_blocked("gpt-4o", caller="alice") is True
    assert prefs.is_blocked("claude:opus", caller="alice") is True
    # Not blocked
    assert prefs.is_blocked("qwen:32b", caller="alice") is False
    # Different caller — same model not blocked
    assert prefs.is_blocked("gpt-4o", caller="bob") is False


# ---------------------------------------------------------------------------
# Test 3: Caller preferred returns correct model
# ---------------------------------------------------------------------------

def test_caller_preferred_returns_correct_model():
    """get_preferred returns the registered preferred alias for the caller."""
    prefs = Preferences(
        caller_preferred={
            "alice": "claude:sonnet",
            "bob": "qwen:32b",
        }
    )

    assert prefs.get_preferred("alice") == "claude:sonnet"
    assert prefs.get_preferred("bob") == "qwen:32b"


# ---------------------------------------------------------------------------
# Test 4: Empty Preferences returns safe defaults
# ---------------------------------------------------------------------------

def test_no_preferences_returns_defaults():
    """With no rules configured, all methods return zero / False / None."""
    prefs = Preferences()

    assert prefs.get_boost("any-model", intent="lookup") == 0.0
    assert prefs.is_blocked("any-model", caller="anyone") is False
    assert prefs.get_preferred("anyone") is None


# ---------------------------------------------------------------------------
# Test 5: Wildcard intent affinity matches all intents
# ---------------------------------------------------------------------------

def test_wildcard_intent_affinity_matches_all_intents():
    """An AffinityRule with intent='*' boosts the model for every intent."""
    prefs = Preferences(
        affinities=[
            AffinityRule(intent="*", model_alias="qwen:32b", boost=5.0),
        ]
    )

    for intent in ("summarize", "lookup", "generate", "translate", ""):
        boost = prefs.get_boost("qwen:32b", intent=intent)
        assert boost == 5.0, f"Expected 5.0 for intent={intent!r}, got {boost}"

    # Other model is unaffected
    assert prefs.get_boost("gpt-4o", intent="summarize") == 0.0
