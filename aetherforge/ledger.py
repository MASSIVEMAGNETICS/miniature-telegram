"""
ledger.py – Conscience Ledger: a hash-chained, append-only audit log.

Every geometric mutation (weight update, synapse pruning, rotor rotation,
or agent lifecycle event) is written as an entry to the ledger.  Each
entry records a SHA-256 hash of its content chained to the previous
entry's hash, making the history tamper-evident.

Entries can be serialised to JSON Lines (.jsonl) or replayed for
regulatory audit.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class LedgerEntry:
    """A single immutable record in the Conscience Ledger."""

    __slots__ = ("index", "timestamp", "event_type", "payload", "prev_hash", "hash")

    def __init__(
        self,
        index: int,
        event_type: str,
        payload: Dict[str, Any],
        prev_hash: str,
    ) -> None:
        self.index = index
        self.timestamp = time.time()
        self.event_type = event_type
        self.payload = payload
        self.prev_hash = prev_hash
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "event_type": self.event_type,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LedgerEntry(index={self.index}, event={self.event_type!r}, "
            f"hash={self.hash[:12]}...)"
        )


class ConscienceLedger:
    """
    Append-only, hash-chained audit log for AetherForge agents.

    Parameters
    ----------
    agent_name : human-readable name of the Aetherling being audited
    """

    GENESIS_HASH = "0" * 64  # conventional zero hash for block 0

    def __init__(self, agent_name: str = "aetherling") -> None:
        self.agent_name = agent_name
        self._entries: List[LedgerEntry] = []
        # Record genesis block
        self._append("GENESIS", {"agent": agent_name, "note": "ledger initialised"})

    # ------------------------------------------------------------------

    def _append(self, event_type: str, payload: Dict[str, Any]) -> LedgerEntry:
        prev_hash = (
            self._entries[-1].hash if self._entries else self.GENESIS_HASH
        )
        entry = LedgerEntry(
            index=len(self._entries),
            event_type=event_type,
            payload=payload,
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Public logging API

    def log_weight_update(self, node_index: int, summary: dict) -> LedgerEntry:
        """Record a weight update from the Echo Ripple protocol."""
        return self._append(
            "WEIGHT_UPDATE",
            {"node": node_index, **{k: v for k, v in summary.items()}},
        )

    def log_synapse_pruned(self, source: int, target: int, reason: str = "") -> LedgerEntry:
        """Record a synapse pruning event."""
        return self._append(
            "SYNAPSE_PRUNED",
            {"source_node": source, "target_node": target, "reason": reason},
        )

    def log_echo_ripple_step(self, stats: dict) -> LedgerEntry:
        """Record an Echo Ripple learning step summary."""
        return self._append("ECHO_RIPPLE_STEP", stats)

    def log_rotor_mutation(self, node_index: int, delta_norm: float) -> LedgerEntry:
        """Record a geometric rotation update on a perimeter node."""
        return self._append(
            "ROTOR_MUTATION",
            {"perimeter_node": node_index, "delta_norm": delta_norm},
        )

    def log_agent_event(self, event: str, detail: dict) -> LedgerEntry:
        """Record a lifecycle event (spawn, fork, reincarnate, swarm, etc.)."""
        return self._append("AGENT_EVENT", {"event": event, **detail})

    # ------------------------------------------------------------------
    # Integrity & replay

    def verify(self) -> bool:
        """
        Verify the chain integrity: every entry's stored hash must equal
        the hash recomputed from its content, and prev_hash must match the
        previous entry's hash.
        """
        for i, entry in enumerate(self._entries):
            recomputed = entry._compute_hash()
            if recomputed != entry.hash:
                return False
            expected_prev = (
                self.GENESIS_HASH if i == 0 else self._entries[i - 1].hash
            )
            if entry.prev_hash != expected_prev:
                return False
        return True

    def replay(self) -> List[Dict[str, Any]]:
        """Return all entries as a list of dicts (ordered by index)."""
        return [e.to_dict() for e in self._entries]

    def save(self, path: str) -> None:
        """Serialise the ledger to a JSON Lines file."""
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str, agent_name: str = "aetherling") -> "ConscienceLedger":
        """
        Load a ledger from a JSON Lines file.

        Note: the loaded ledger is read-only (verify before trusting).
        """
        ledger = cls.__new__(cls)
        ledger.agent_name = agent_name
        ledger._entries = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line.strip())
                entry = LedgerEntry.__new__(LedgerEntry)
                entry.index = d["index"]
                entry.timestamp = d["timestamp"]
                entry.event_type = d["event_type"]
                entry.payload = d["payload"]
                entry.prev_hash = d["prev_hash"]
                entry.hash = d["hash"]
                ledger._entries.append(entry)
        return ledger

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ConscienceLedger(agent={self.agent_name!r}, "
            f"entries={len(self)}, valid={self.verify()})"
        )
