from __future__ import annotations


VALID_METHODS = {"replay_ce", "global_hap", "flat_lrhap", "sacil_v0"}


def method_uses_geometry(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name != "replay_ce"

