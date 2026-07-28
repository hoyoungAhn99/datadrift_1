from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and attach its absolute source path."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {source}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(source)
    return config


def deep_update(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a recursive mapping update without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_required(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"missing configuration key: {dotted_key}")
        value = value[key]
    return value

