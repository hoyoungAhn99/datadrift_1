from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return data


def namespace_from_config(
    path: str | Path,
    defaults: dict[str, Any],
    required: tuple[str, ...] = (),
    choices: dict[str, set[Any]] | None = None,
) -> Namespace:
    cfg = defaults.copy()
    loaded = load_config(path)
    unknown = sorted(set(loaded) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {', '.join(unknown)}")

    cfg.update(loaded)
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required config key(s) in {path}: {', '.join(missing)}")

    for key, allowed in (choices or {}).items():
        value = cfg.get(key)
        if value not in allowed:
            allowed_str = ", ".join(str(x) for x in sorted(allowed))
            raise ValueError(f"Invalid value for {key}: {value}. Expected one of: {allowed_str}")

    cfg["config"] = str(path)
    return Namespace(**cfg)
