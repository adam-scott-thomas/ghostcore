"""Tests for ControlCore.registry.budget — BudgetTracker."""

from __future__ import annotations

import math
import time

import pytest

from ControlCore.registry.budget import BudgetConfig, BudgetExceeded, BudgetTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(daily: float = 0.0, hourly: float = 0.0) -> BudgetTracker:
    return BudgetTracker(BudgetConfig(daily_limit=daily, hourly_limit=hourly))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_record_spend_and_spent_today():
    """record_spend stores costs that are reflected in spent_today."""
    tracker = make_tracker()
    tracker.record_spend(0.10)
    tracker.record_spend(0.25)
    assert abs(tracker.spent_today() - 0.35) < 1e-9


def test_remaining_daily():
    """remaining_daily = limit - spent_today."""
    tracker = make_tracker(daily=1.00)
    tracker.record_spend(0.30)
    assert abs(tracker.remaining_daily() - 0.70) < 1e-9


def test_remaining_hourly():
    """remaining_hourly = hourly_limit - spent_this_hour."""
    tracker = make_tracker(hourly=0.50)
    tracker.record_spend(0.20)
    assert abs(tracker.remaining_hourly() - 0.30) < 1e-9


def test_check_passes_under_budget():
    """check does not raise when estimated cost keeps spend within limits."""
    tracker = make_tracker(daily=1.00, hourly=0.50)
    tracker.record_spend(0.10)
    tracker.check(0.20)  # 0.30 total — well under both limits


def test_check_raises_daily_exceeded():
    """check raises BudgetExceeded when estimated cost breaches the daily limit."""
    tracker = make_tracker(daily=1.00)
    tracker.record_spend(0.90)
    with pytest.raises(BudgetExceeded):
        tracker.check(0.20)  # 1.10 > 1.00


def test_check_raises_hourly_exceeded():
    """check raises BudgetExceeded when estimated cost breaches the hourly limit."""
    tracker = make_tracker(daily=10.00, hourly=0.50)
    tracker.record_spend(0.45)
    with pytest.raises(BudgetExceeded):
        tracker.check(0.10)  # 0.55 > 0.50


def test_daily_ratio():
    """daily_ratio returns fraction of daily limit consumed."""
    tracker = make_tracker(daily=2.00)
    tracker.record_spend(1.00)
    assert abs(tracker.daily_ratio() - 0.50) < 1e-9


def test_unlimited_never_raises():
    """With 0-limits (unlimited), check never raises regardless of spend."""
    tracker = make_tracker(daily=0.0, hourly=0.0)
    for _ in range(100):
        tracker.record_spend(999.99)
    tracker.check(999_999.99)  # must not raise
    assert tracker.remaining_daily() == math.inf
    assert tracker.remaining_hourly() == math.inf
    assert tracker.daily_ratio() == 0.0


def test_old_records_excluded():
    """Records older than 86 400 s are not counted in the current window."""
    tracker = make_tracker(daily=1.00)
    # Inject a record from 2 days ago directly into the internal list
    old_ts = time.time() - 86_401
    tracker._records.append((old_ts, 0.99))
    # Only the fresh spend below should count
    tracker.record_spend(0.05)
    assert abs(tracker.spent_today() - 0.05) < 1e-9
    # check should pass: 0.05 + 0.10 = 0.15 < 1.00
    tracker.check(0.10)
