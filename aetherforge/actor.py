"""
actor.py – Aetherling Actor Model.

Each Aetherling is an isolated, autonomous agent that wraps a CTGGNRuntime.
The actor model ensures agents are loosely coupled: they communicate only
via message passing (here implemented as asyncio queues).

Lifecycle events
----------------
SPAWN      : A new Aetherling is created.
STEP       : The agent processes one input step.
FORK       : The agent clones itself (shallow copy of weights), creating a
             child Aetherling for swarm-mode deployment.
REINCARNATE: The agent resets its state (simulates recovery from node failure).
TERMINATE  : The agent shuts down and logs a final entry.

Swarm Mode
----------
When swarm_mode is enabled via `toggle_swarm`, the Aetherling will fork up
to `max_forks` child agents.  Each child runs independently and its results
can be aggregated by the parent.

TEE / Soul Token stub
---------------------
The `soul_token` attribute stores an opaque bytes object representing the
agent's identity key.  In a real deployment this would be protected by
hardware-level encryption (Intel SGX, AMD SEV, etc.).  Here it is a
SHA-256 hash of the agent's name + creation timestamp, serving as a unique
identifier and integrity anchor.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import CTGGNRuntime
    from .ledger import ConscienceLedger


def _derive_soul_token(name: str, created_at: float) -> bytes:
    """Derive a deterministic soul token from agent identity."""
    material = f"{name}:{created_at:.6f}".encode("utf-8")
    return hashlib.sha256(material).digest()


class Aetherling:
    """
    An autonomous actor wrapping a CTGGNRuntime.

    Parameters
    ----------
    runtime    : CTGGNRuntime instance powering this agent
    name       : human-readable agent name
    max_forks  : maximum number of child agents in swarm mode
    """

    def __init__(
        self,
        runtime: "CTGGNRuntime",
        name: str = "aetherling",
        max_forks: int = 10000,
    ) -> None:
        self.runtime = runtime
        self.name = name
        self.max_forks = max_forks

        self._created_at: float = time.time()
        self.soul_token: bytes = _derive_soul_token(name, self._created_at)

        self._swarm_mode: bool = False
        self._forks: List["Aetherling"] = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._running: bool = False

        # Log spawn event
        self.runtime.ledger.log_agent_event(
            "SPAWN",
            {
                "name": self.name,
                "soul_token": self.soul_token.hex(),
                "created_at": self._created_at,
            },
        )

    # ------------------------------------------------------------------
    # Core step

    def step(self, inputs=None) -> Any:
        """Process one input through the underlying runtime."""
        result = self.runtime.step(inputs)
        self.runtime.ledger.log_agent_event(
            "STEP", {"name": self.name, "step": self.runtime.step_count}
        )
        return result

    # ------------------------------------------------------------------
    # Swarm API

    def toggle_swarm(self, enabled: bool) -> None:
        """Enable or disable swarm mode."""
        self._swarm_mode = enabled
        self.runtime.ledger.log_agent_event(
            "SWARM_TOGGLE", {"name": self.name, "enabled": enabled}
        )

    def fork(self) -> "Aetherling":
        """
        Fork this Aetherling into a child agent.

        The child receives a deep copy of the parent's weights and a fresh
        ledger so its history is independent.  The parent ledger records
        the fork event.

        Raises RuntimeError if max_forks would be exceeded.
        """
        if len(self._forks) >= self.max_forks:
            raise RuntimeError(
                f"max_forks={self.max_forks} reached; cannot fork further"
            )

        child_runtime = self.runtime.clone()
        child_name = f"{self.name}:fork-{len(self._forks)}"
        child = Aetherling(
            runtime=child_runtime,
            name=child_name,
            max_forks=self.max_forks,
        )
        self._forks.append(child)
        self.runtime.ledger.log_agent_event(
            "FORK",
            {
                "parent": self.name,
                "child": child_name,
                "child_soul_token": child.soul_token.hex(),
                "total_forks": len(self._forks),
            },
        )
        return child

    # ------------------------------------------------------------------
    # Lifecycle

    def reincarnate(self) -> None:
        """
        Reset the agent's continuous-time state (simulates recovery from
        a compromised or failed edge node).  Weights are preserved; only
        the transient ODE state is cleared.
        """
        self.runtime.reset()
        self.runtime.ledger.log_agent_event(
            "REINCARNATE",
            {"name": self.name, "soul_token": self.soul_token.hex()},
        )

    def terminate(self) -> None:
        """Gracefully shut down the agent."""
        self._running = False
        self._swarm_mode = False
        self.runtime.ledger.log_agent_event(
            "TERMINATE",
            {
                "name": self.name,
                "steps_completed": self.runtime.step_count,
                "soul_token": self.soul_token.hex(),
            },
        )

    # ------------------------------------------------------------------
    # Async message loop (thin stub for gRPC / network integration)

    async def run(self) -> None:
        """
        Async event loop: receive messages from the inbox queue and
        process them.  In production this would be backed by gRPC
        streams for low-latency swarm communication.
        """
        self._running = True
        self.runtime.ledger.log_agent_event(
            "RUN_START", {"name": self.name}
        )
        while self._running:
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=0.1)
                result = self.step(msg)
                self._inbox.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as exc:  # pragma: no cover
                self.runtime.ledger.log_agent_event(
                    "ERROR", {"name": self.name, "error": str(exc)}
                )

    async def send(self, message: Any) -> None:
        """Send a message to this agent's inbox."""
        await self._inbox.put(message)

    # ------------------------------------------------------------------

    @property
    def fork_count(self) -> int:
        return len(self._forks)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Aetherling(name={self.name!r}, "
            f"swarm={self._swarm_mode}, forks={self.fork_count}, "
            f"steps={self.runtime.step_count})"
        )
