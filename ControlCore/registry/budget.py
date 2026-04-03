"""In-memory spend tracking with time-windowed records.

No persistence — resets on restart. learning.db handles persistent history.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when an estimated cost would exceed a configured limit."""


@dataclass
class BudgetConfig:
    daily_limit: float = 0.0   # 0 = unlimited
    hourly_limit: float = 0.0  # 0 = unlimited


class BudgetTracker:
    """Track LLM spend against daily and hourly limits."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._records: list[tuple[float, float]] = []  # (timestamp, cost)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_spend(self, cost: float) -> None:
        """Append a spend record stamped at the current time."""
        self._records.append((time.time(), cost))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def spent_today(self) -> float:
        """Sum of costs recorded in the last 86 400 seconds."""
        cutoff = time.time() - 86_400
        return sum(cost for ts, cost in self._records if ts >= cutoff)

    def spent_this_hour(self) -> float:
        """Sum of costs recorded in the last 3 600 seconds."""
        cutoff = time.time() - 3_600
        return sum(cost for ts, cost in self._records if ts >= cutoff)

    def remaining_daily(self) -> float:
        """Remaining daily budget (math.inf if unlimited)."""
        if self._config.daily_limit == 0:
            return math.inf
        return self._config.daily_limit - self.spent_today()

    def remaining_hourly(self) -> float:
        """Remaining hourly budget (math.inf if unlimited)."""
        if self._config.hourly_limit == 0:
            return math.inf
        return self._config.hourly_limit - self.spent_this_hour()

    def daily_ratio(self) -> float:
        """Fraction of daily limit consumed (0.0 if unlimited)."""
        if self._config.daily_limit == 0:
            return 0.0
        return self.spent_today() / self._config.daily_limit

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def check(self, estimated_cost: float) -> None:
        """Raise BudgetExceeded if estimated_cost would breach either limit."""
        if self._config.daily_limit > 0:
            if self.spent_today() + estimated_cost > self._config.daily_limit:
                raise BudgetExceeded(
                    f"Daily limit ${self._config.daily_limit:.4f} would be exceeded "
                    f"(spent ${self.spent_today():.4f}, estimated ${estimated_cost:.4f})"
                )
        if self._config.hourly_limit > 0:
            if self.spent_this_hour() + estimated_cost > self._config.hourly_limit:
                raise BudgetExceeded(
                    f"Hourly limit ${self._config.hourly_limit:.4f} would be exceeded "
                    f"(spent ${self.spent_this_hour():.4f}, estimated ${estimated_cost:.4f})"
                )
