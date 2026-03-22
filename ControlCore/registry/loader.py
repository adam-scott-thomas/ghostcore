"""
P2-02: Registry Loader + Validation

Loads registry entries from local files (YAML or JSON).
Validates strictly. Rejects duplicates and invalid entries.
Exposes read-only in-memory registry.

Constraints:
- No network calls
- No auto-discovery
- No defaults beyond schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError as PydanticValidationError

from ControlCore.registry.schema import ModelRegistry, ModelEntry


class RegistryValidationError(Exception):
    """Validation error with structured details."""

    def __init__(self, message: str, errors: List[Dict[str, Any]]):
        super().__init__(message)
        self.message = message
        self.errors = errors

    def __str__(self) -> str:
        error_lines = [self.message]
        for err in self.errors[:10]:  # Limit displayed errors
            error_lines.append(f"  - {err.get('loc', 'unknown')}: {err.get('msg', 'unknown error')}")
        if len(self.errors) > 10:
            error_lines.append(f"  ... and {len(self.errors) - 10} more errors")
        return "\n".join(error_lines)


class RegistryLoadError(Exception):
    """Error loading registry file."""

    def __init__(self, message: str, path: Optional[Path] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.cause = cause

    def __str__(self) -> str:
        result = self.message
        if self.path:
            result += f" (file: {self.path})"
        if self.cause:
            result += f" (cause: {self.cause})"
        return result


def _try_load_yaml(content: str) -> Dict[str, Any]:
    """Try to load content as YAML."""
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        raise RegistryLoadError(
            "YAML support requires PyYAML. Install with: pip install pyyaml"
        )
    except yaml.YAMLError as e:
        raise RegistryLoadError(f"Invalid YAML: {e}")


def _try_load_json(content: str) -> Dict[str, Any]:
    """Try to load content as JSON."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RegistryLoadError(f"Invalid JSON: {e}")


def load_registry_from_dict(data: Dict[str, Any]) -> ModelRegistry:
    """
    Load and validate a registry from a dictionary.

    Args:
        data: Registry data as a dictionary

    Returns:
        Validated ModelRegistry

    Raises:
        RegistryValidationError: If validation fails
    """
    try:
        registry = ModelRegistry.model_validate(data)
        return registry
    except PydanticValidationError as e:
        errors = [
            {
                "loc": ".".join(str(loc) for loc in err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        raise RegistryValidationError(
            f"Registry validation failed with {len(errors)} error(s)",
            errors=errors,
        )
    except ValueError as e:
        # Catch model_validator errors (like duplicates)
        raise RegistryValidationError(
            str(e),
            errors=[{"loc": "registry", "msg": str(e), "type": "value_error"}],
        )


def load_registry_from_file(path: Union[str, Path]) -> ModelRegistry:
    """
    Load and validate a registry from a file.

    Supports JSON (.json) and YAML (.yaml, .yml) formats.

    Args:
        path: Path to the registry file

    Returns:
        Validated ModelRegistry

    Raises:
        RegistryLoadError: If file cannot be read or parsed
        RegistryValidationError: If validation fails
    """
    path = Path(path)

    if not path.exists():
        raise RegistryLoadError(f"Registry file not found", path=path)

    if not path.is_file():
        raise RegistryLoadError(f"Path is not a file", path=path)

    try:
        content = path.read_text(encoding="utf-8")
    except IOError as e:
        raise RegistryLoadError(f"Failed to read file", path=path, cause=e)

    if not content.strip():
        raise RegistryLoadError(f"Registry file is empty", path=path)

    # Determine format by extension
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = _try_load_json(content)
    elif suffix in (".yaml", ".yml"):
        data = _try_load_yaml(content)
    else:
        # Try JSON first, then YAML
        try:
            data = _try_load_json(content)
        except RegistryLoadError:
            try:
                data = _try_load_yaml(content)
            except RegistryLoadError:
                raise RegistryLoadError(
                    f"Could not parse file as JSON or YAML",
                    path=path,
                )

    if not isinstance(data, dict):
        raise RegistryLoadError(
            f"Registry must be a dictionary/object, got {type(data).__name__}",
            path=path,
        )

    try:
        return load_registry_from_dict(data)
    except RegistryValidationError as e:
        # Re-raise with path info
        raise RegistryValidationError(
            f"{e.message} in {path}",
            errors=e.errors,
        )


def validate_registry_entry(data: Dict[str, Any]) -> ModelEntry:
    """
    Validate a single registry entry.

    Useful for validating entries before adding to registry.

    Args:
        data: Entry data as dictionary

    Returns:
        Validated ModelEntry

    Raises:
        RegistryValidationError: If validation fails
    """
    try:
        return ModelEntry.model_validate(data)
    except PydanticValidationError as e:
        errors = [
            {
                "loc": ".".join(str(loc) for loc in err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        raise RegistryValidationError(
            f"Entry validation failed",
            errors=errors,
        )


# Singleton registry holder (read-only after load)
_global_registry: Optional[ModelRegistry] = None


def get_global_registry() -> Optional[ModelRegistry]:
    """Get the global registry if loaded."""
    return _global_registry


def set_global_registry(registry: ModelRegistry) -> None:
    """Set the global registry."""
    global _global_registry
    _global_registry = registry


def clear_global_registry() -> None:
    """Clear the global registry (for testing)."""
    global _global_registry
    _global_registry = None
