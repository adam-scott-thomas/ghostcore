"""
P2-04: Routing Order & Fallback Policy

Orders eligible model aliases without executing anything.
Deterministic, explainable, pure function.

NO model calls. NO registry mutation. NO inferred defaults.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ControlCore.schemas import ControlCoreCall
from ControlCore.registry.schema import ModelEntry, TrustTier, CapabilityTag


class RoutingFactor(str, Enum):
    """Factors that influence routing order."""
    TRUST_TIER = "trust_tier"
    CAPABILITY_MATCH = "capability_match"
    LATENCY_HINT = "latency_hint"
    COST_HINT = "cost_hint"
    CONTEXT_HEADROOM = "context_headroom"
    REFUSAL_RATE = "refusal_rate"
    PROVIDER_PREFERENCE = "provider_preference"
    OBSERVED_LATENCY = "observed_latency"
    BUDGET_PRESSURE = "budget_pressure"
    TASK_AFFINITY = "task_affinity"
    LOAD_BALANCE = "load_balance"


@dataclass
class RoutingReason:
    """Explanation for why a model is placed at a specific position."""
    factor: RoutingFactor
    description: str
    score_contribution: float  # How much this factor contributed to final score
    raw_value: Optional[Any] = None  # The underlying value used


@dataclass
class RankedModel:
    """A model with its routing score and reasons."""
    alias: str
    rank: int  # 1-indexed position
    score: float  # Composite score (higher = better)
    reasons: List[RoutingReason] = field(default_factory=list)
    model: Optional[ModelEntry] = None  # Reference to full entry if available

    def explain(self) -> str:
        """Human-readable explanation of ranking."""
        lines = [f"#{self.rank} {self.alias} (score: {self.score:.2f})"]
        for reason in self.reasons:
            sign = "+" if reason.score_contribution >= 0 else ""
            lines.append(f"  {sign}{reason.score_contribution:.2f} {reason.factor.value}: {reason.description}")
        return "\n".join(lines)


@dataclass
class RoutingResult:
    """Result of routing policy application."""
    ordered: List[RankedModel]  # Models in priority order
    policy_version: str = "1.0.0"
    call_id: Optional[str] = None

    @property
    def ordered_aliases(self) -> List[str]:
        """Get ordered list of aliases."""
        return [m.alias for m in self.ordered]

    @property
    def top(self) -> Optional[RankedModel]:
        """Get top-ranked model."""
        return self.ordered[0] if self.ordered else None

    def explain_all(self) -> str:
        """Human-readable explanation of full ordering."""
        if not self.ordered:
            return "No models in routing result."
        lines = [f"Routing Result (policy v{self.policy_version}):"]
        for ranked in self.ordered:
            lines.append(ranked.explain())
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_version": self.policy_version,
            "call_id": self.call_id,
            "ordered": [
                {
                    "alias": m.alias,
                    "rank": m.rank,
                    "score": m.score,
                    "reasons": [
                        {
                            "factor": r.factor.value,
                            "description": r.description,
                            "score_contribution": r.score_contribution,
                        }
                        for r in m.reasons
                    ],
                }
                for m in self.ordered
            ],
        }


@dataclass
class RoutingWeights:
    """
    Weights for routing factors.

    These control how much each factor influences the final score.
    All weights should be non-negative. Higher = more influence.
    """
    trust_tier: float = 30.0        # Trust is important
    capability_match: float = 20.0  # Capability fit matters
    latency_hint: float = 10.0      # Prefer faster models
    cost_hint: float = 5.0          # Cost is a tiebreaker
    context_headroom: float = 5.0   # More headroom is better
    refusal_rate: float = 15.0      # Avoid models that refuse often
    observed_latency: float = 12.0  # Observed latency from learning store
    budget_pressure: float = 8.0    # Budget-aware cost penalisation
    task_affinity: float = 25.0     # Preference boost for matching tasks
    load_balance_jitter: float = 0.0  # Random jitter for load balancing


@dataclass
class RefusalHistory:
    """Historical refusal rates for models (input data, not computed)."""
    rates: Dict[str, float] = field(default_factory=dict)  # alias -> refusal rate (0.0 - 1.0)

    def get_rate(self, alias: str) -> float:
        """Get refusal rate for a model (0.0 if unknown)."""
        return self.rates.get(alias, 0.0)


def _score_trust_tier(model: ModelEntry, weight: float) -> Tuple[float, RoutingReason]:
    """Score based on trust tier."""
    tier_scores = {
        TrustTier.trusted: 1.0,
        TrustTier.standard: 0.5,
        TrustTier.untrusted: 0.0,
    }
    raw_score = tier_scores.get(model.trust_tier, 0.5)
    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.TRUST_TIER,
        description=f"Trust tier '{model.trust_tier.value}' ({raw_score:.1f})",
        score_contribution=contribution,
        raw_value=model.trust_tier.value,
    )


def _score_capability_match(
    model: ModelEntry,
    call: ControlCoreCall,
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on how well capabilities match the intent."""
    # Map intents to relevant capabilities
    intent_capability_map = {
        "lookup": [CapabilityTag.factual, CapabilityTag.extract],
        "summarize": [CapabilityTag.summarize],
        "extract": [CapabilityTag.extract],
        "compare": [CapabilityTag.compare, CapabilityTag.reason],
        "draft": [CapabilityTag.draft, CapabilityTag.creative],
        "classify": [CapabilityTag.classify],
        "reason": [CapabilityTag.reason],
        "critique": [CapabilityTag.critique, CapabilityTag.judge],
        "translate": [CapabilityTag.translate],
        "unknown": [],
    }

    intent = call.intent.cls.value
    relevant_caps = intent_capability_map.get(intent, [])

    if not relevant_caps or not model.capability_tags:
        # No specific match criteria - neutral score
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.CAPABILITY_MATCH,
            description="No specific capability requirements",
            score_contribution=weight * 0.5,
            raw_value=None,
        )

    # Count matching capabilities
    matches = sum(1 for cap in relevant_caps if cap in model.capability_tags)
    match_ratio = matches / len(relevant_caps) if relevant_caps else 0.5
    contribution = match_ratio * weight

    matched_names = [cap.value for cap in relevant_caps if cap in model.capability_tags]

    return contribution, RoutingReason(
        factor=RoutingFactor.CAPABILITY_MATCH,
        description=f"Matched {matches}/{len(relevant_caps)} relevant capabilities: {matched_names or 'none'}",
        score_contribution=contribution,
        raw_value=match_ratio,
    )


def _score_latency_hint(model: ModelEntry, weight: float) -> Tuple[float, RoutingReason]:
    """Score based on latency hints (soft timeout as proxy)."""
    # Lower soft timeout = faster = better
    # Normalize: 10s is good (1.0), 60s+ is bad (0.0)
    soft_ms = model.timeouts.soft_ms
    if soft_ms <= 10000:
        raw_score = 1.0
    elif soft_ms >= 60000:
        raw_score = 0.0
    else:
        raw_score = 1.0 - (soft_ms - 10000) / 50000

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.LATENCY_HINT,
        description=f"Soft timeout {soft_ms}ms (latency score: {raw_score:.2f})",
        score_contribution=contribution,
        raw_value=soft_ms,
    )


def _score_cost_hint(model: ModelEntry, weight: float) -> Tuple[float, RoutingReason]:
    """Score based on cost hints (lower = better)."""
    if not model.cost_hints:
        # No cost info - neutral score
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.COST_HINT,
            description="No cost information available",
            score_contribution=weight * 0.5,
            raw_value=None,
        )

    # Use input cost as primary metric
    input_cost = model.cost_hints.input_per_1k_tokens or 0.0

    # Normalize: $0 is best (1.0), $0.02+ is expensive (0.0)
    if input_cost <= 0.001:
        raw_score = 1.0
    elif input_cost >= 0.02:
        raw_score = 0.0
    else:
        raw_score = 1.0 - (input_cost - 0.001) / 0.019

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.COST_HINT,
        description=f"Input cost ${input_cost:.4f}/1k tokens (cost score: {raw_score:.2f})",
        score_contribution=contribution,
        raw_value=input_cost,
    )


def _score_context_headroom(
    model: ModelEntry,
    call: ControlCoreCall,
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on context window headroom."""
    # Estimate prompt tokens
    prompt_tokens = len(call.prompt) // 4 + 1
    for ctx in call.context:
        prompt_tokens += len(ctx.content) // 4 + 1

    headroom = model.context_window - prompt_tokens
    headroom_ratio = headroom / model.context_window if model.context_window > 0 else 0

    # More headroom = better (but diminishing returns)
    if headroom_ratio >= 0.8:
        raw_score = 1.0
    elif headroom_ratio <= 0.2:
        raw_score = 0.2
    else:
        raw_score = 0.2 + (headroom_ratio - 0.2) * (0.8 / 0.6)

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.CONTEXT_HEADROOM,
        description=f"{headroom_ratio*100:.0f}% context headroom ({headroom}/{model.context_window} tokens)",
        score_contribution=contribution,
        raw_value=headroom_ratio,
    )


def _score_refusal_rate(
    model: ModelEntry,
    refusal_history: Optional[RefusalHistory],
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on historical refusal rate."""
    if not refusal_history:
        return weight * 0.8, RoutingReason(
            factor=RoutingFactor.REFUSAL_RATE,
            description="No refusal history available",
            score_contribution=weight * 0.8,
            raw_value=None,
        )

    rate = refusal_history.get_rate(model.alias)

    # Lower refusal rate = better
    # 0% refusal = 1.0, 50%+ refusal = 0.0
    if rate <= 0.0:
        raw_score = 1.0
    elif rate >= 0.5:
        raw_score = 0.0
    else:
        raw_score = 1.0 - (rate * 2)

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.REFUSAL_RATE,
        description=f"Historical refusal rate: {rate*100:.1f}%",
        score_contribution=contribution,
        raw_value=rate,
    )


def _get_spine(name: str):
    """Safely retrieve a capability from spine singleton. Returns None if unavailable."""
    try:
        from spine import Core
        return Core.instance().get(name)
    except Exception:
        return None


def _score_observed_latency(
    model: ModelEntry,
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on observed latency from LearningStore via spine."""
    learning = _get_spine("learning")
    if learning is None:
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.OBSERVED_LATENCY,
            description="No learning store available (spine not booted)",
            score_contribution=weight * 0.5,
            raw_value=None,
        )

    stats = learning.stats(model.alias)
    if stats.call_count < 5:
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.OBSERVED_LATENCY,
            description=f"Insufficient data ({stats.call_count} calls, need 5)",
            score_contribution=weight * 0.5,
            raw_value=stats.call_count,
        )

    avg_ms = stats.avg_latency_ms
    # Normalize: <1s = 1.0, >10s = 0.0
    if avg_ms <= 1000:
        raw_score = 1.0
    elif avg_ms >= 10000:
        raw_score = 0.0
    else:
        raw_score = 1.0 - (avg_ms - 1000) / 9000

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.OBSERVED_LATENCY,
        description=f"Observed avg latency {avg_ms:.0f}ms (score: {raw_score:.2f})",
        score_contribution=contribution,
        raw_value=avg_ms,
    )


def _score_budget_pressure(
    model: ModelEntry,
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on budget pressure — penalise expensive models when budget is tight."""
    budget = _get_spine("budget")
    if budget is None:
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.BUDGET_PRESSURE,
            description="No budget tracker available (spine not booted)",
            score_contribution=weight * 0.5,
            raw_value=None,
        )

    ratio = budget.daily_ratio()
    if ratio <= 0.8:
        # Budget is fine — neutral score
        return weight * 0.5, RoutingReason(
            factor=RoutingFactor.BUDGET_PRESSURE,
            description=f"Budget OK ({ratio*100:.0f}% consumed)",
            score_contribution=weight * 0.5,
            raw_value=ratio,
        )

    # Budget pressure is high (>80%). Penalise expensive models.
    input_cost = 0.0
    if model.cost_hints:
        input_cost = model.cost_hints.input_per_1k_tokens or 0.0

    # More expensive models get penalised more under pressure.
    # Normalize cost: $0 = no penalty (1.0), $0.02+ = max penalty (0.0)
    if input_cost <= 0.001:
        cost_score = 1.0
    elif input_cost >= 0.02:
        cost_score = 0.0
    else:
        cost_score = 1.0 - (input_cost - 0.001) / 0.019

    # Scale penalty by how far over 80% we are (80%→ mild, 100%→ full)
    pressure_factor = min((ratio - 0.8) / 0.2, 1.0)
    # Cheaper models keep their score, expensive ones drop toward 0
    raw_score = cost_score * (1.0 - pressure_factor)

    contribution = raw_score * weight

    return contribution, RoutingReason(
        factor=RoutingFactor.BUDGET_PRESSURE,
        description=f"Budget {ratio*100:.0f}% consumed, model cost ${input_cost:.4f}/1k (score: {raw_score:.2f})",
        score_contribution=contribution,
        raw_value=ratio,
    )


def _score_task_affinity(
    model: ModelEntry,
    call: ControlCoreCall,
    weight: float,
) -> Tuple[float, RoutingReason]:
    """Score based on task affinity preferences. Boost IS the score (not multiplied by weight)."""
    prefs = _get_spine("preferences")
    if prefs is None:
        return 0.0, RoutingReason(
            factor=RoutingFactor.TASK_AFFINITY,
            description="No preferences available (spine not booted)",
            score_contribution=0.0,
            raw_value=None,
        )

    intent = call.intent.cls.value
    boost = prefs.get_boost(model.alias, intent=intent)

    return boost, RoutingReason(
        factor=RoutingFactor.TASK_AFFINITY,
        description=f"Affinity boost {boost:+.1f} for intent '{intent}'",
        score_contribution=boost,
        raw_value=boost,
    )


def _score_load_balance(jitter: float) -> Tuple[float, RoutingReason]:
    """Add random jitter for load balancing. If jitter=0, deterministic."""
    if jitter <= 0:
        return 0.0, RoutingReason(
            factor=RoutingFactor.LOAD_BALANCE,
            description="Load balance jitter disabled",
            score_contribution=0.0,
            raw_value=0.0,
        )

    value = random.uniform(-jitter, jitter)

    return value, RoutingReason(
        factor=RoutingFactor.LOAD_BALANCE,
        description=f"Random jitter {value:+.2f} (range ±{jitter:.1f})",
        score_contribution=value,
        raw_value=value,
    )


def compute_routing_order(
    call: ControlCoreCall,
    eligible_models: List[ModelEntry],
    *,
    weights: Optional[RoutingWeights] = None,
    refusal_history: Optional[RefusalHistory] = None,
) -> RoutingResult:
    """
    Compute routing order for eligible models.

    This is a PURE FUNCTION. It is DETERMINISTIC given the same inputs.

    It does NOT:
    - Call any models
    - Mutate registry state
    - Infer new defaults
    - Make network requests

    Args:
        call: The validated ControlCoreCall
        eligible_models: List of eligible ModelEntry objects
        weights: Optional custom weights (uses defaults if None)
        refusal_history: Optional historical refusal rates

    Returns:
        RoutingResult with ordered models and explanations
    """
    if weights is None:
        weights = RoutingWeights()

    scored_models: List[Tuple[float, List[RoutingReason], ModelEntry]] = []

    for model in eligible_models:
        reasons: List[RoutingReason] = []
        total_score = 0.0

        # Score each factor
        score, reason = _score_trust_tier(model, weights.trust_tier)
        total_score += score
        reasons.append(reason)

        score, reason = _score_capability_match(model, call, weights.capability_match)
        total_score += score
        reasons.append(reason)

        score, reason = _score_latency_hint(model, weights.latency_hint)
        total_score += score
        reasons.append(reason)

        score, reason = _score_cost_hint(model, weights.cost_hint)
        total_score += score
        reasons.append(reason)

        score, reason = _score_context_headroom(model, call, weights.context_headroom)
        total_score += score
        reasons.append(reason)

        score, reason = _score_refusal_rate(model, refusal_history, weights.refusal_rate)
        total_score += score
        reasons.append(reason)

        # --- Intelligent routing factors (spine-backed) ---

        score, reason = _score_observed_latency(model, weights.observed_latency)
        total_score += score
        reasons.append(reason)

        score, reason = _score_budget_pressure(model, weights.budget_pressure)
        total_score += score
        reasons.append(reason)

        score, reason = _score_task_affinity(model, call, weights.task_affinity)
        total_score += score
        reasons.append(reason)

        score, reason = _score_load_balance(weights.load_balance_jitter)
        total_score += score
        reasons.append(reason)

        scored_models.append((total_score, reasons, model))

    # Sort by score descending, then by alias for stability
    scored_models.sort(key=lambda x: (-x[0], x[2].alias))

    # Build result
    ordered = [
        RankedModel(
            alias=model.alias,
            rank=i + 1,
            score=score,
            reasons=reasons,
            model=model,
        )
        for i, (score, reasons, model) in enumerate(scored_models)
    ]

    return RoutingResult(
        ordered=ordered,
        call_id=call.call_id,
    )


def compute_routing_order_from_aliases(
    call: ControlCoreCall,
    eligible_aliases: List[str],
    registry: "ModelRegistry",
    *,
    weights: Optional[RoutingWeights] = None,
    refusal_history: Optional[RefusalHistory] = None,
) -> RoutingResult:
    """
    Convenience function: compute routing from alias list.

    Looks up ModelEntry objects from registry.
    """
    from ControlCore.registry.schema import ModelRegistry

    eligible_models = []
    for alias in eligible_aliases:
        model = registry.get(alias)
        if model:
            eligible_models.append(model)

    return compute_routing_order(
        call,
        eligible_models,
        weights=weights,
        refusal_history=refusal_history,
    )
