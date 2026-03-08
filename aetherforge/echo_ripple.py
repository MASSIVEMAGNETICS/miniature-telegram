"""
echo_ripple.py – Echo Ripple Protocol (wave-interference online learning).

Mechanism
---------
1. **Forward wave**: the CT-GGN processes an input, producing activations
   at every node.  These activations are the "forward wave".
2. **Echo wave**: the output is reflected off the perimeter nodes back
   through the graph.  The reflected signal is the "echo".
3. **Interference**: at each edge (i, j) the algorithm correlates the
   forward activation of node *i* with the echo activation of node *j*.

       Δw_ij = η · corr(forward_i, echo_j)

   * Constructive (corr > +prune_threshold) → weight increases.
   * Destructive  (corr < −prune_threshold) → weight is set to 0 (pruned).
   * Weak / neutral               → no change.

This is a biologically-motivated, Hebbian-style credit-assignment rule
that replaces standard gradient descent for online, always-on learning.
It is analogous to Contrastive Hebbian Learning but operates on the
graph structure directly.

Weight matrices updated
-----------------------
* LiquidNode.W_rec  (recurrent weights between core nodes)
* LiquidNode.W_in   (input weights for driven nodes)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .liquid_core import LiquidCore
    from .perimeter import PerimeterRing
    from .topology import FlowerOfLife


def _cosine_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (range −1 … +1)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EchoRippleProtocol:
    """
    Online learning via wave interference.

    Parameters
    ----------
    topology         : FlowerOfLife graph (used for edge enumeration)
    learning_rate    : η – scales weight increments (default 0.01)
    prune_threshold  : magnitude of correlation below which no update is made;
                       weights with correlation < −prune_threshold are pruned.
    weight_decay     : small L2 regularisation applied at every step (default 0)
    max_weight_norm  : clip W_rec columns to this L2 norm (prevents explosion)
    """

    def __init__(
        self,
        topology: "FlowerOfLife",
        learning_rate: float = 0.01,
        prune_threshold: float = 0.1,
        weight_decay: float = 0.0,
        max_weight_norm: float = 5.0,
    ) -> None:
        self.topology = topology
        self.lr = learning_rate
        self.prune_threshold = prune_threshold
        self.weight_decay = weight_decay
        self.max_weight_norm = max_weight_norm

        # Track cumulative pruning events for auditing
        self.prune_count: int = 0
        self.update_count: int = 0

    # ------------------------------------------------------------------

    def _generate_echo(
        self,
        core: "LiquidCore",
        perimeter: "PerimeterRing",
    ) -> np.ndarray:
        """
        Generate the echo wave.

        The 12 outermost core nodes (ring-2, indices 7-18) are directly
        adjacent to the perimeter.  Their activations are pushed through
        the perimeter ring and inverted, producing the reflected echo.

        Returns echo [n_core, state_dim].
        """
        n_core = len(core.nodes)
        state_dim = core.state_dim
        fwd = core.activations  # [n_core, state_dim]

        # The last NUM_PERIMETER nodes in the core that border the perimeter
        # are the ring-2 nodes (indices n_core − NUM_PERIMETER : n_core in
        # the boundary, but the topology only has NUM_CORE=19 < NUM_PERIMETER=18
        # so we use the last 18 of the 19 core nodes as the boundary layer).
        n_boundary = perimeter.topology.NUM_PERIMETER
        boundary_states = fwd[-n_boundary:]  # [18, state_dim]

        perimeter_echo = perimeter.forward(boundary_states)  # [18, state_dim]

        # Invert the echo (reflection inverts phase)
        perimeter_echo = -perimeter_echo

        # Build full echo array: non-boundary nodes get zero echo
        echo = np.zeros((n_core, state_dim), dtype=np.float64)
        echo[-n_boundary:] = perimeter_echo
        return echo

    # ------------------------------------------------------------------

    def step(
        self,
        core: "LiquidCore",
        perimeter: "PerimeterRing",
    ) -> dict:
        """
        Perform one Echo Ripple learning step.

        Uses the cached forward activations stored in each LiquidNode and
        the echo signal generated via the PerimeterRing to update the
        recurrent weights of the LiquidCore.

        Returns a summary dict with statistics about the update.
        """
        fwd = core.activations       # [n_core, state_dim]
        echo = self._generate_echo(core, perimeter)  # [n_core, state_dim]

        n_core = len(core.nodes)
        total_updated = 0
        total_pruned = 0

        # Iterate over every directed edge (i → j) in the core subgraph
        for i in range(n_core):
            for j in core.topology.core_indices:
                if core.topology.adjacency[i, j] == 0:
                    continue
                # Correlation between forward activation at i and echo at j
                corr = _cosine_correlation(fwd[i], echo[j])

                if corr > self.prune_threshold:
                    # Constructive interference: strengthen synapse.
                    # Update the i-th column of node j's W_rec to reinforce
                    # the connection i→j proportional to their correlation.
                    core.nodes[j].W_rec[:, i % core.nodes[j].W_rec.shape[1]] += (
                        self.lr * corr * fwd[i][: core.nodes[j].W_rec.shape[1]]
                    )
                    total_updated += 1

                elif corr < -self.prune_threshold:
                    # Destructive interference: prune synapse
                    col = i % core.nodes[j].W_rec.shape[1]
                    core.nodes[j].W_rec[:, col] = 0.0
                    total_pruned += 1
                    self.prune_count += 1

        # Weight decay + norm clipping
        for node in core.nodes:
            if self.weight_decay > 0:
                node.W_rec *= (1.0 - self.weight_decay)
            # Clip each column to max_weight_norm
            norms = np.linalg.norm(node.W_rec, axis=0, keepdims=True)
            norms = np.where(norms < self.max_weight_norm, 1.0, norms / self.max_weight_norm)
            node.W_rec /= norms

        self.update_count += 1
        return {
            "step": self.update_count,
            "edges_strengthened": total_updated,
            "synapses_pruned": total_pruned,
            "total_pruned_cumulative": self.prune_count,
        }
