"""
aetherscript.py – AetherScript YAML deployment-spec parser & validator.

AetherScript is the declarative configuration language for AetherForge
deployments.  A spec file (or YAML string) is loaded and validated against
a known schema, then returned as a plain Python dict ready for use by
CTGGNRuntime.

Example spec (from the architecture document):

    aetherling:
      name: "Prometheus-Core"
      topology:
        base: "FlowerOfLife-37"
        core_fluidity: 0.85
      perimeter_io:
        nodes: 18
        geometry: "GATr-Platonic"
        erasure_threshold: 0.4
      swarm:
        max_forks: 10000
        sync_protocol: "EchoRipple-gRPC"
      security:
        soul_token_enclave: true
        ledger: "Enterprise-Private"
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "aetherling": {
        "name": "aetherling",
        "topology": {
            "base": "FlowerOfLife-37",
            "core_fluidity": 0.5,
        },
        "perimeter_io": {
            "nodes": 18,
            "geometry": "GATr-Platonic",
            "erasure_threshold": 0.4,
        },
        "swarm": {
            "max_forks": 1000,
            "sync_protocol": "EchoRipple-gRPC",
        },
        "security": {
            "soul_token_enclave": False,
            "ledger": "Enterprise-Private",
        },
    }
}

SUPPORTED_TOPOLOGIES = {"FlowerOfLife-37"}
SUPPORTED_GEOMETRIES = {"GATr-Platonic", "GATr-Flat"}
SUPPORTED_PROTOCOLS = {"EchoRipple-gRPC", "EchoRipple-local"}
SUPPORTED_LEDGERS = {"Enterprise-Private", "On-Chain"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_aetherscript(source: str) -> Dict[str, Any]:
    """
    Load an AetherScript spec.

    *source* may be:
    - a file path ending in ``.yaml`` or ``.yml``, or
    - a raw YAML string.

    Returns the parsed spec as a nested dict.
    """
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = source

    spec = yaml.safe_load(raw)
    if not isinstance(spec, dict):
        raise ValueError("AetherScript spec must be a YAML mapping at the top level.")
    return spec


def validate_aetherscript(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise an AetherScript spec dict.

    Fills in missing keys with defaults and raises ``ValueError`` for
    invalid values.

    Returns the validated + normalised spec.
    """
    if "aetherling" not in spec:
        raise ValueError("AetherScript spec must contain an 'aetherling' key.")

    aether = spec["aetherling"]
    if not isinstance(aether, dict):
        raise ValueError("'aetherling' must be a YAML mapping.")

    # --- Deep-merge with defaults ---
    merged = _deep_merge(_DEFAULTS["aetherling"], aether)

    # --- topology ---
    topo = merged.get("topology", {})
    base = topo.get("base", "FlowerOfLife-37")
    if base not in SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f"topology.base={base!r} is not supported. "
            f"Choose from: {sorted(SUPPORTED_TOPOLOGIES)}"
        )
    fluidity = float(topo.get("core_fluidity", 0.5))
    if not 0.0 <= fluidity <= 1.0:
        raise ValueError(
            f"topology.core_fluidity must be in [0, 1], got {fluidity}."
        )
    topo["core_fluidity"] = fluidity

    # --- perimeter_io ---
    perim = merged.get("perimeter_io", {})
    n_nodes = int(perim.get("nodes", 18))
    if n_nodes != 18:
        raise ValueError(
            f"perimeter_io.nodes must be 18 for FlowerOfLife-37, got {n_nodes}."
        )
    geometry = perim.get("geometry", "GATr-Platonic")
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(
            f"perimeter_io.geometry={geometry!r} is not supported. "
            f"Choose from: {sorted(SUPPORTED_GEOMETRIES)}"
        )
    erasure = float(perim.get("erasure_threshold", 0.4))
    if not 0.0 <= erasure <= 1.0:
        raise ValueError(
            f"perimeter_io.erasure_threshold must be in [0, 1], got {erasure}."
        )

    # --- swarm ---
    swarm = merged.get("swarm", {})
    max_forks = int(swarm.get("max_forks", 1000))
    if max_forks < 1:
        raise ValueError(f"swarm.max_forks must be ≥ 1, got {max_forks}.")
    protocol = swarm.get("sync_protocol", "EchoRipple-gRPC")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"swarm.sync_protocol={protocol!r} is not supported. "
            f"Choose from: {sorted(SUPPORTED_PROTOCOLS)}"
        )

    # --- security ---
    sec = merged.get("security", {})
    ledger = sec.get("ledger", "Enterprise-Private")
    if ledger not in SUPPORTED_LEDGERS:
        raise ValueError(
            f"security.ledger={ledger!r} is not supported. "
            f"Choose from: {sorted(SUPPORTED_LEDGERS)}"
        )

    return {"aetherling": merged}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*, returning a new dict.
    *override* values take precedence.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
