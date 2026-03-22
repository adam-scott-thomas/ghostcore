"""Model registry - authority over callable targets, no execution logic."""

from ControlCore.registry.schema import (
    ModelEntry,
    Provider,
    CapabilityTag,
    TrustTier,
    TimeoutDefaults,
    CostHints,
    ModelRegistry,
)
from ControlCore.registry.loader import (
    load_registry_from_file,
    load_registry_from_dict,
    validate_registry_entry,
    RegistryLoadError,
    RegistryValidationError,
    get_global_registry,
    set_global_registry,
    clear_global_registry,
)
from ControlCore.registry.dial import (
    filter_eligible_models,
    EligibilityResult,
    ExclusionReason,
    ModelExclusion,
    filter_by_capability,
    filter_by_provider,
)
from ControlCore.registry.routing import (
    compute_routing_order,
    compute_routing_order_from_aliases,
    RoutingResult,
    RankedModel,
    RoutingReason,
    RoutingFactor,
    RoutingWeights,
    RefusalHistory,
)
from ControlCore.registry.fallback import (
    FallbackPolicy,
    RephraseStrategy,
    RephraseConfig,
    ModelSwitchCondition,
    ModelSwitchConfig,
    QueueEscalationCondition,
    QueueEscalationConfig,
    RetryTiming,
    default_policy,
    aggressive_retry_policy,
    fail_fast_policy,
    queue_preferred_policy,
    cost_sensitive_policy,
)

__all__ = [
    # Schema
    "ModelEntry",
    "Provider",
    "CapabilityTag",
    "TrustTier",
    "TimeoutDefaults",
    "CostHints",
    "ModelRegistry",
    # Loader
    "load_registry_from_file",
    "load_registry_from_dict",
    "validate_registry_entry",
    "RegistryLoadError",
    "RegistryValidationError",
    "get_global_registry",
    "set_global_registry",
    "clear_global_registry",
    # Dial
    "filter_eligible_models",
    "EligibilityResult",
    "ExclusionReason",
    "ModelExclusion",
    "filter_by_capability",
    "filter_by_provider",
    # Routing
    "compute_routing_order",
    "compute_routing_order_from_aliases",
    "RoutingResult",
    "RankedModel",
    "RoutingReason",
    "RoutingFactor",
    "RoutingWeights",
    "RefusalHistory",
    # Fallback
    "FallbackPolicy",
    "RephraseStrategy",
    "RephraseConfig",
    "ModelSwitchCondition",
    "ModelSwitchConfig",
    "QueueEscalationCondition",
    "QueueEscalationConfig",
    "RetryTiming",
    "default_policy",
    "aggressive_retry_policy",
    "fail_fast_policy",
    "queue_preferred_policy",
    "cost_sensitive_policy",
]
