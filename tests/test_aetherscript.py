"""Tests for the AetherScript YAML parser & validator."""

import pytest
from aetherforge.aetherscript import load_aetherscript, validate_aetherscript

VALID_SPEC = """
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

MINIMAL_SPEC = """
aetherling:
  name: "MinimalAgent"
"""


class TestLoadAetherscript:
    def test_load_valid_yaml_string(self):
        spec = load_aetherscript(VALID_SPEC)
        assert "aetherling" in spec

    def test_load_non_mapping_raises(self):
        with pytest.raises(ValueError, match="YAML mapping"):
            load_aetherscript("- item1\n- item2\n")

    def test_load_minimal_spec(self):
        spec = load_aetherscript(MINIMAL_SPEC)
        assert spec["aetherling"]["name"] == "MinimalAgent"


class TestValidateAetherscript:
    def test_valid_spec_passes(self):
        spec = load_aetherscript(VALID_SPEC)
        validated = validate_aetherscript(spec)
        assert "aetherling" in validated

    def test_minimal_spec_gets_defaults(self):
        spec = load_aetherscript(MINIMAL_SPEC)
        validated = validate_aetherscript(spec)
        a = validated["aetherling"]
        assert a["topology"]["base"] == "FlowerOfLife-37"
        assert a["perimeter_io"]["nodes"] == 18
        assert a["swarm"]["max_forks"] == 1000

    def test_missing_aetherling_key_raises(self):
        with pytest.raises(ValueError, match="aetherling"):
            validate_aetherscript({"other": "value"})

    def test_invalid_topology_base_raises(self):
        spec = {"aetherling": {"topology": {"base": "Unknown-99"}}}
        with pytest.raises(ValueError, match="topology.base"):
            validate_aetherscript(spec)

    def test_core_fluidity_out_of_range_raises(self):
        spec = {"aetherling": {"topology": {"core_fluidity": 1.5}}}
        with pytest.raises(ValueError, match="core_fluidity"):
            validate_aetherscript(spec)

    def test_wrong_perimeter_nodes_raises(self):
        spec = {"aetherling": {"perimeter_io": {"nodes": 10}}}
        with pytest.raises(ValueError, match="perimeter_io.nodes"):
            validate_aetherscript(spec)

    def test_invalid_geometry_raises(self):
        spec = {"aetherling": {"perimeter_io": {"geometry": "Octahedron"}}}
        with pytest.raises(ValueError, match="geometry"):
            validate_aetherscript(spec)

    def test_erasure_threshold_out_of_range_raises(self):
        spec = {"aetherling": {"perimeter_io": {"erasure_threshold": -0.1}}}
        with pytest.raises(ValueError, match="erasure_threshold"):
            validate_aetherscript(spec)

    def test_max_forks_zero_raises(self):
        spec = {"aetherling": {"swarm": {"max_forks": 0}}}
        with pytest.raises(ValueError, match="max_forks"):
            validate_aetherscript(spec)

    def test_invalid_protocol_raises(self):
        spec = {"aetherling": {"swarm": {"sync_protocol": "HTTP-Poll"}}}
        with pytest.raises(ValueError, match="sync_protocol"):
            validate_aetherscript(spec)

    def test_invalid_ledger_raises(self):
        spec = {"aetherling": {"security": {"ledger": "MySQL"}}}
        with pytest.raises(ValueError, match="ledger"):
            validate_aetherscript(spec)

    def test_name_preserved(self):
        spec = load_aetherscript(VALID_SPEC)
        validated = validate_aetherscript(spec)
        assert validated["aetherling"]["name"] == "Prometheus-Core"

    def test_deep_merge_overrides_defaults(self):
        spec = {"aetherling": {"swarm": {"max_forks": 500}}}
        validated = validate_aetherscript(spec)
        assert validated["aetherling"]["swarm"]["max_forks"] == 500
        # Other swarm defaults should still be present
        assert "sync_protocol" in validated["aetherling"]["swarm"]
