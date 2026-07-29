from __future__ import annotations


VALID_METHODS = {
    "replay_ce",
    "global_hap",
    "flat_lrhap",
    "sacil_v0",
    "afc",
    "afc_global_hap",
    "afc_flat_lrhap",
    "afc_sacil_v0",
}

AFC_METHODS = {
    "afc",
    "afc_global_hap",
    "afc_flat_lrhap",
    "afc_sacil_v0",
}

NON_GEOMETRY_METHODS = {"replay_ce", "afc"}


def method_uses_geometry(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name not in NON_GEOMETRY_METHODS


def method_uses_afc(method_name: str) -> bool:
    if method_name not in VALID_METHODS:
        raise ValueError(f"unknown method: {method_name}")
    return method_name in AFC_METHODS
