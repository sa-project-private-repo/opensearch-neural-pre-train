"""
Configuration loader utilities.

Supports loading from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

from src.train.config.base import BaseConfig
from src.train.config.v33 import V33Config


T = TypeVar("T", bound=BaseConfig)


def load_config(
    config_path: Optional[str] = None,
    config_type: Type[T] = V33Config,
    overrides: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        config_type: Configuration class to instantiate
        overrides: Dictionary of values to override

    Returns:
        Configured instance of config_type

    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_dict: Dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Apply explicit overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Create config instance
    config = config_type(**config_dict)
    if hasattr(config, "validate"):
        config.validate()

    return config


def save_config(config: BaseConfig, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration instance to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = _config_to_dict(config)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert config dataclass to dictionary recursively."""
    if hasattr(config, "__dataclass_fields__"):
        return {
            key: _config_to_dict(getattr(config, key))
            for key in config.__dataclass_fields__
        }
    elif isinstance(config, list):
        return [_config_to_dict(item) for item in config]
    elif isinstance(config, dict):
        return {key: _config_to_dict(value) for key, value in config.items()}
    else:
        return config


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config.

    Environment variables are prefixed with TRAIN_ and use double underscores
    for nested keys. Example: TRAIN_MODEL__NAME=bert-base
    """
    prefix = "TRAIN_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Parse key path (e.g., TRAIN_MODEL__NAME -> ["model", "name"])
        config_key = key[len(prefix):].lower()
        key_path = config_key.split("__")

        # Navigate to nested dict and set value
        current = config_dict
        for part in key_path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Try to parse value as appropriate type
        final_key = key_path[-1]
        current[final_key] = _parse_env_value(value)

    return config_dict


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
    # Try boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
