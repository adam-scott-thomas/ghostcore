"""
ControlCore Configuration Module

Handles loading configuration, model registry, and adapter initialization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ControlCore.registry.schema import ModelRegistry, ModelEntry, Provider
from ControlCore.registry.loader import load_registry_from_dict, set_global_registry
from ControlCore.adapters.executor import AdapterRegistry
from ControlCore.adapters.interface import ExecutionAdapter


# Default config paths
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_REGISTRY_FILE = DEFAULT_CONFIG_DIR / "registry.json"


class ControlCoreConfig:
    """
    Central configuration for ControlCore.

    Loads from environment variables and config files.
    """

    def __init__(self):
        # Server settings
        self.host: str = os.environ.get("CONTROLCORE_HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("CONTROLCORE_PORT", "8265"))

        # Ollama settings
        self.ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        # Registry path
        self.registry_path: str = os.environ.get(
            "CONTROLCORE_REGISTRY",
            str(DEFAULT_REGISTRY_FILE)
        )

        # API Keys (from environment)
        self.api_keys: Dict[str, Optional[str]] = {
            "openai": os.environ.get("OPENAI_API_KEY"),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "xai": os.environ.get("XAI_API_KEY"),
            "google": os.environ.get("GOOGLE_API_KEY"),
            "mistral": os.environ.get("MISTRAL_API_KEY"),
            "groq": os.environ.get("GROQ_API_KEY"),
            "together": os.environ.get("TOGETHER_API_KEY"),
            "deepseek": os.environ.get("DEEPSEEK_API_KEY"),
            "perplexity": os.environ.get("PERPLEXITY_API_KEY"),
            "cohere": os.environ.get("COHERE_API_KEY"),
        }

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is configured for provider."""
        return bool(self.api_keys.get(provider))

    def get_enabled_cloud_providers(self) -> List[str]:
        """Get list of providers with configured API keys."""
        return [p for p, k in self.api_keys.items() if k]


def load_model_registry(config: ControlCoreConfig) -> ModelRegistry:
    """
    Load model registry from config file.

    Returns empty registry if file not found.
    """
    registry_path = Path(config.registry_path)

    if not registry_path.exists():
        print(f"[ControlCore] Registry file not found: {registry_path}")
        return load_registry_from_dict({"version": "1.0.0", "models": []})

    try:
        with open(registry_path) as f:
            data = json.load(f)

        registry = load_registry_from_dict(data)
        print(f"[ControlCore] Loaded {len(registry)} models from registry")
        return registry

    except Exception as e:
        print(f"[ControlCore] Error loading registry: {e}")
        return load_registry_from_dict({"version": "1.0.0", "models": []})


def create_adapter_registry(config: ControlCoreConfig, model_registry: ModelRegistry) -> AdapterRegistry:
    """
    Create adapter registry with all configured adapters.

    Automatically registers:
    - Ollama adapter for local models
    - Cloud adapters for providers with API keys
    """
    from ControlCore.adapters.ollama import create_ollama_adapter
    from ControlCore.adapters.cloud import (
        create_openai_adapter,
        create_anthropic_adapter,
        create_xai_adapter,
        create_google_adapter,
        create_groq_adapter,
        create_together_adapter,
        create_mistral_adapter,
        create_deepseek_adapter,
        create_perplexity_adapter,
    )

    adapter_registry = AdapterRegistry()

    # Build model mappings from registry
    local_models = {}

    for model in model_registry.models:
        if model.provider == Provider.local:
            # Map alias to Ollama model name
            local_models[model.alias] = model.provider_model_id or f"{model.alias}:latest"

    # Create Ollama adapter for local models
    if local_models:
        ollama = create_ollama_adapter(
            base_url=config.ollama_url,
            model_mapping=local_models,
        )
        adapter_registry.register(ollama)
        print(f"[ControlCore] Registered Ollama adapter with {len(local_models)} models")

    # Create cloud adapters for providers with API keys
    cloud_adapters = {
        "openai": create_openai_adapter,
        "anthropic": create_anthropic_adapter,
        "xai": create_xai_adapter,
        "google": create_google_adapter,
        "groq": create_groq_adapter,
        "together": create_together_adapter,
        "mistral": create_mistral_adapter,
        "deepseek": create_deepseek_adapter,
        "perplexity": create_perplexity_adapter,
    }

    for provider, factory in cloud_adapters.items():
        if config.has_api_key(provider):
            adapter = factory()
            adapter_registry.register(adapter)
            print(f"[ControlCore] Registered {provider} adapter")

    # Set default adapter (Ollama if available, else first cloud adapter)
    adapters = adapter_registry.list_adapters()
    if adapters:
        # Prefer Ollama for default
        if "ollama" in adapters:
            ollama_adapter = adapter_registry.get_adapter_for_model(list(local_models.keys())[0] if local_models else "")
            if ollama_adapter:
                adapter_registry.set_default(ollama_adapter)

    return adapter_registry


def initialize_controlcore() -> tuple[ControlCoreConfig, ModelRegistry, AdapterRegistry]:
    """
    Initialize ControlCore with all configuration.

    Returns:
        Tuple of (config, model_registry, adapter_registry)
    """
    config = ControlCoreConfig()
    model_registry = load_model_registry(config)
    adapter_registry = create_adapter_registry(config, model_registry)

    # Set global registry
    set_global_registry(model_registry)

    return config, model_registry, adapter_registry


def print_config_status(config: ControlCoreConfig, model_registry: ModelRegistry) -> None:
    """Print configuration status for debugging."""
    print("\n" + "=" * 60)
    print("ControlCore Configuration Status")
    print("=" * 60)

    print(f"\nServer: {config.host}:{config.port}")
    print(f"Ollama: {config.ollama_url}")
    print(f"Registry: {config.registry_path}")

    # Count models by provider
    local_count = sum(1 for m in model_registry.models if m.provider == Provider.local)
    cloud_count = sum(1 for m in model_registry.models if m.provider != Provider.local)

    print(f"\nModels: {len(model_registry)} total ({local_count} local, {cloud_count} cloud)")

    # Show API key status
    print("\nAPI Keys:")
    for provider, key in config.api_keys.items():
        status = "configured" if key else "missing"
        print(f"  {provider}: {status}")

    # Show enabled local models
    print("\nLocal Models (Ollama):")
    for model in model_registry.models:
        if model.provider == Provider.local and model.enabled:
            print(f"  {model.alias} -> {model.provider_model_id}")

    print("=" * 60 + "\n")


# Environment variable template
ENV_TEMPLATE = """
# ControlCore Environment Variables
# Copy this to .env and fill in your API keys

# Server Configuration
CONTROLCORE_HOST=127.0.0.1
CONTROLCORE_PORT=8265

# Ollama Configuration
OLLAMA_URL=http://localhost:11434

# Registry Path (optional - uses default if not set)
# CONTROLCORE_REGISTRY=/path/to/registry.json

# API Keys - Add your keys here
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
COHERE_API_KEY=...
""".strip()


def write_env_template(path: Optional[str] = None) -> str:
    """Write .env template file."""
    if path is None:
        path = str(DEFAULT_CONFIG_DIR / ".env.example")

    with open(path, "w") as f:
        f.write(ENV_TEMPLATE)

    return path
