"""
Task affinity rules and caller-specific preferences.

Configured via TOML, registered in spine.

Classes:
    AffinityRule  — maps an intent (or "*" wildcard) + model_alias to a score boost
    Preferences   — aggregates rules and per-caller settings; used by the router
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AffinityRule:
    """
    Boost a model's routing score when intent matches.

    Attributes:
        intent:       Intent string this rule applies to. Use ``"*"`` to match
                      every intent.
        model_alias:  Alias of the model to boost (e.g. ``"qwen:32b"``).
        boost:        Score added to the model's routing weight (default 10.0).
    """

    intent: str
    model_alias: str
    boost: float = 10.0


class Preferences:
    """
    Aggregated task-affinity rules and per-caller settings.

    Args:
        affinities:        List of AffinityRule objects.
        caller_blocklists: Mapping of caller handle → set of blocked model aliases.
        caller_preferred:  Mapping of caller handle → preferred model alias.
    """

    def __init__(
        self,
        affinities: List[AffinityRule] | None = None,
        caller_blocklists: Dict[str, List[str]] | None = None,
        caller_preferred: Dict[str, str] | None = None,
    ) -> None:
        self._affinities: List[AffinityRule] = affinities or []
        self._caller_blocklists: Dict[str, set] = {
            caller: set(models)
            for caller, models in (caller_blocklists or {}).items()
        }
        self._caller_preferred: Dict[str, str] = caller_preferred or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_boost(self, model_alias: str, *, intent: str = "") -> float:
        """
        Return the sum of all affinity boosts that apply to *model_alias*.

        A rule matches when:
        - ``rule.model_alias == model_alias``, AND
        - ``rule.intent == intent`` OR ``rule.intent == "*"``

        Args:
            model_alias: The model to score.
            intent:      The current call's intent string (may be empty).

        Returns:
            Accumulated boost (0.0 if no rules match).
        """
        total = 0.0
        for rule in self._affinities:
            if rule.model_alias != model_alias:
                continue
            if rule.intent == "*" or rule.intent == intent:
                total += rule.boost
        return total

    def is_blocked(self, model_alias: str, *, caller: str = "") -> bool:
        """
        Return True if *model_alias* is on *caller*'s blocklist.

        Args:
            model_alias: The model to check.
            caller:      The caller handle to look up.

        Returns:
            True if blocked, False otherwise (including unknown callers).
        """
        blocklist = self._caller_blocklists.get(caller, set())
        return model_alias in blocklist

    def get_preferred(self, caller: str) -> Optional[str]:
        """
        Return the preferred model alias for *caller*, or None if unset.

        Args:
            caller: The caller handle to look up.

        Returns:
            Model alias string, or None.
        """
        return self._caller_preferred.get(caller)
