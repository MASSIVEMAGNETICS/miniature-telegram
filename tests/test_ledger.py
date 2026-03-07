"""Tests for the Conscience Ledger."""

import json
import os
import tempfile
import pytest
from aetherforge.ledger import LedgerEntry, ConscienceLedger


class TestLedgerEntry:
    def test_hash_is_hex_64_chars(self):
        entry = LedgerEntry(0, "TEST", {"x": 1}, "0" * 64)
        assert len(entry.hash) == 64
        assert all(c in "0123456789abcdef" for c in entry.hash)

    def test_hash_changes_with_payload(self):
        e1 = LedgerEntry(0, "TEST", {"x": 1}, "0" * 64)
        e2 = LedgerEntry(0, "TEST", {"x": 2}, "0" * 64)
        assert e1.hash != e2.hash

    def test_to_dict_keys(self):
        entry = LedgerEntry(0, "TEST", {}, "0" * 64)
        d = entry.to_dict()
        assert "index" in d
        assert "timestamp" in d
        assert "event_type" in d
        assert "payload" in d
        assert "prev_hash" in d
        assert "hash" in d


class TestConscienceLedger:
    def setup_method(self):
        self.ledger = ConscienceLedger(agent_name="test-agent")

    def test_genesis_entry_exists(self):
        assert len(self.ledger) == 1
        assert self.ledger._entries[0].event_type == "GENESIS"

    def test_integrity_after_genesis(self):
        assert self.ledger.verify()

    def test_log_weight_update(self):
        self.ledger.log_weight_update(3, {"step": 1, "edges_strengthened": 2})
        assert len(self.ledger) == 2
        assert self.ledger._entries[1].event_type == "WEIGHT_UPDATE"

    def test_log_synapse_pruned(self):
        self.ledger.log_synapse_pruned(0, 5, reason="destructive interference")
        e = self.ledger._entries[-1]
        assert e.event_type == "SYNAPSE_PRUNED"
        assert e.payload["source_node"] == 0
        assert e.payload["target_node"] == 5

    def test_log_echo_ripple_step(self):
        stats = {"step": 1, "edges_strengthened": 3, "synapses_pruned": 1}
        self.ledger.log_echo_ripple_step(stats)
        assert self.ledger._entries[-1].event_type == "ECHO_RIPPLE_STEP"

    def test_log_rotor_mutation(self):
        self.ledger.log_rotor_mutation(7, delta_norm=0.05)
        e = self.ledger._entries[-1]
        assert e.event_type == "ROTOR_MUTATION"
        assert e.payload["perimeter_node"] == 7

    def test_log_agent_event(self):
        self.ledger.log_agent_event("SPAWN", {"name": "test"})
        e = self.ledger._entries[-1]
        assert e.event_type == "AGENT_EVENT"
        assert e.payload["event"] == "SPAWN"

    def test_verify_after_multiple_entries(self):
        for i in range(5):
            self.ledger.log_weight_update(i, {"step": i})
        assert self.ledger.verify()

    def test_verify_detects_tampering(self):
        self.ledger.log_weight_update(0, {"step": 0})
        # Tamper with a field
        self.ledger._entries[-1].payload["step"] = 999
        assert not self.ledger.verify()

    def test_chain_integrity_prev_hash(self):
        self.ledger.log_weight_update(0, {})
        e1 = self.ledger._entries[0]
        e2 = self.ledger._entries[1]
        assert e2.prev_hash == e1.hash

    def test_replay_returns_all_entries(self):
        self.ledger.log_weight_update(0, {})
        replay = self.ledger.replay()
        assert len(replay) == 2
        assert all(isinstance(r, dict) for r in replay)

    def test_save_and_load_roundtrip(self):
        self.ledger.log_weight_update(1, {"x": 42})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            self.ledger.save(path)
            loaded = ConscienceLedger.load(path, agent_name="test-agent")
            assert len(loaded) == len(self.ledger)
            assert loaded.verify()
        finally:
            os.unlink(path)

    def test_loaded_ledger_entries_match(self):
        self.ledger.log_echo_ripple_step({"step": 5})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            self.ledger.save(path)
            loaded = ConscienceLedger.load(path)
            for orig, restored in zip(self.ledger._entries, loaded._entries):
                assert orig.hash == restored.hash
        finally:
            os.unlink(path)
