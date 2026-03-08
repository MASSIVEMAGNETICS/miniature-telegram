"""
runtime.py – CTGGNRuntime: top-level Continuous-Time Geometric Graph Network.

The runtime orchestrates the full forward pass:

    1. Receive external input (shape: [n_core, state_dim] or flat)
    2. Drive the LiquidCore ODE for one timestep
    3. Push boundary states through the PerimeterRing (GATr nodes)
    4. Inject the echo back into the core as additional input
    5. Run the EchoRippleProtocol to update weights online
    6. Log the step summary to the ConscienceLedger
    7. Return the mean-pooled output state
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

import numpy as np

from .topology import FlowerOfLife
from .liquid_core import LiquidCore
from .perimeter import PerimeterRing
from .echo_ripple import EchoRippleProtocol
from .ledger import ConscienceLedger


class CTGGNRuntime:
    """
    Continuous-Time Geometric Graph Network runtime.

    Parameters
    ----------
    state_dim         : hidden-state dimensionality for each node
    dt                : ODE integration timestep
    learning_rate     : Echo Ripple learning rate
    prune_threshold   : Edge pruning threshold (see EchoRippleProtocol)
    erasure_threshold : DDL erasure aggressiveness (see PerimeterRing)
    weight_decay      : L2 regularisation applied each step
    node_spacing      : Spacing for Flower-of-Life layout
    agent_name        : Name written to the ledger
    seed              : Global RNG seed
    """

    def __init__(
        self,
        state_dim: int = 16,
        dt: float = 0.1,
        learning_rate: float = 0.01,
        prune_threshold: float = 0.1,
        erasure_threshold: float = 0.4,
        weight_decay: float = 0.0,
        node_spacing: float = 1.0,
        agent_name: str = "aetherling",
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.agent_name = agent_name

        # Build subsystems
        self.topology = FlowerOfLife(node_spacing=node_spacing)
        self.core = LiquidCore(
            topology=self.topology,
            state_dim=state_dim,
            dt=dt,
            seed=seed,
        )
        self.perimeter = PerimeterRing(
            topology=self.topology,
            state_dim=state_dim,
            erasure_threshold=erasure_threshold,
            seed=None if seed is None else seed + 1000,
        )
        self.echo_ripple = EchoRippleProtocol(
            topology=self.topology,
            learning_rate=learning_rate,
            prune_threshold=prune_threshold,
            weight_decay=weight_decay,
        )
        self.ledger = ConscienceLedger(agent_name=agent_name)

        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Main forward step

    def step(
        self,
        inputs: Union[np.ndarray, float, list, None] = None,
    ) -> np.ndarray:
        """
        Advance the CT-GGN by one timestep.

        Parameters
        ----------
        inputs : array-like of shape [n_core, state_dim], [state_dim],
                 a scalar, or None (zero input).

        Returns
        -------
        output : ndarray [state_dim] – mean-pooled core state.
        """
        n_core = self.topology.NUM_CORE
        ext = self._prepare_input(inputs, n_core)

        # --- 1. Drive liquid core ---
        core_states = self.core.step(ext)  # [n_core, state_dim]

        # --- 2. Perimeter echo ---
        n_boundary = self.topology.NUM_PERIMETER
        boundary = core_states[-n_boundary:]  # [18, state_dim]
        echo = self.perimeter.forward(boundary)  # [18, state_dim]

        # --- 3. Inject echo back into core (second ODE step) ---
        echo_input = np.zeros((n_core, self.state_dim), dtype=np.float64)
        echo_input[-n_boundary:] = -echo  # Wave reflection: the perimeter acts as a
        # reflective boundary.  When the forward wave hits the perimeter nodes,
        # they invert the phase (multiply by −1) before propagating the echo
        # back into the core.  This sign flip drives destructive interference
        # for stale synapses and constructive interference for active ones.
        self.core.step(echo_input)

        # --- 4. Echo Ripple learning ---
        stats = self.echo_ripple.step(self.core, self.perimeter)

        # --- 5. Audit log ---
        self.ledger.log_echo_ripple_step(stats)

        self.step_count += 1

        # --- 6. Output: mean-pool all core states ---
        return self.core.states.mean(axis=0)

    # ------------------------------------------------------------------

    def _prepare_input(
        self, inputs: Any, n_core: int
    ) -> np.ndarray:
        """Normalise diverse input shapes to [n_core, state_dim]."""
        if inputs is None:
            return np.zeros((n_core, self.state_dim), dtype=np.float64)
        arr = np.asarray(inputs, dtype=np.float64)
        if arr.shape == (n_core, self.state_dim):
            return arr
        if arr.shape == (self.state_dim,):
            # Broadcast single vector to all nodes
            return np.broadcast_to(arr, (n_core, self.state_dim)).copy()
        if arr.ndim == 0:
            return np.full((n_core, self.state_dim), float(arr))
        # Attempt flat reshape
        flat = arr.ravel()
        total = n_core * self.state_dim
        if flat.size >= total:
            return flat[:total].reshape(n_core, self.state_dim)
        # Pad with zeros
        padded = np.zeros(total, dtype=np.float64)
        padded[: flat.size] = flat
        return padded.reshape(n_core, self.state_dim)

    # ------------------------------------------------------------------
    # Lifecycle helpers

    def reset(self) -> None:
        """Reset the transient ODE state of all core nodes."""
        self.core.reset()
        self.ledger.log_agent_event("RESET", {"agent": self.agent_name})

    def clone(self) -> "CTGGNRuntime":
        """
        Deep-copy the runtime (weights + topology) with a fresh ledger.
        Used by Aetherling.fork() for swarm-mode child agents.
        """
        child = CTGGNRuntime.__new__(CTGGNRuntime)
        child.state_dim = self.state_dim
        child.agent_name = self.agent_name + "-clone"
        child.topology = self.topology  # topology is read-only; safe to share
        child.core = copy.deepcopy(self.core)
        child.perimeter = copy.deepcopy(self.perimeter)
        child.echo_ripple = EchoRippleProtocol(
            topology=self.topology,
            learning_rate=self.echo_ripple.lr,
            prune_threshold=self.echo_ripple.prune_threshold,
            weight_decay=self.echo_ripple.weight_decay,
        )
        child.ledger = ConscienceLedger(agent_name=child.agent_name)
        child.step_count = 0
        return child

    # ------------------------------------------------------------------

    @classmethod
    def from_aetherscript(cls, spec: Dict) -> "CTGGNRuntime":
        """
        Instantiate a CTGGNRuntime from a validated AetherScript spec dict.

        Parameters
        ----------
        spec : output of ``validate_aetherscript``
        """
        a = spec["aetherling"]
        topo = a.get("topology", {})
        perim = a.get("perimeter_io", {})
        swarm = a.get("swarm", {})

        # Map core_fluidity → dt: high fluidity → small dt (fast updates)
        fluidity = float(topo.get("core_fluidity", 0.5))
        dt = 0.2 * (1.0 - fluidity) + 0.01  # range 0.01 … 0.21

        return cls(
            agent_name=a.get("name", "aetherling"),
            erasure_threshold=float(perim.get("erasure_threshold", 0.4)),
            dt=dt,
        )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CTGGNRuntime(agent={self.agent_name!r}, "
            f"state_dim={self.state_dim}, steps={self.step_count})"
        )
