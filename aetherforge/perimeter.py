"""
perimeter.py – Geometric Algebra perimeter nodes for the outer-18 ring.

Each PerimeterNode operates in Clifford algebra Cl(3,0), whose basis is:

    Grade 0 (scalar)     : 1       – 1 component
    Grade 1 (vector)     : e1,e2,e3 – 3 components
    Grade 2 (bivector)   : e12,e13,e23 – 3 components
    Grade 3 (pseudoscalar): e123   – 1 component
    Total                          : 8 components

The geometric product is the foundational operation; inner and outer
(wedge) products are derived from it.

Deep Delta Learning (DDL) + Manifold-Constrained Hyper-Connections (mHC)
------------------------------------------------------------------------
Each perimeter node computes a "delta signal" that measures how much the
incoming forward wave has changed since the last cycle.  When this delta
exceeds `erasure_threshold`, the node applies a learned erasure mask that
zeroes (prunes) components in the multivector state before writing back to
the core – preventing context saturation as described in the spec.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Cl(3,0) geometric algebra helpers
# ---------------------------------------------------------------------------

# Basis index convention:
# 0: 1 (scalar)
# 1: e1
# 2: e2
# 3: e3
# 4: e12
# 5: e13
# 6: e23
# 7: e123

# Pre-computed Cayley table for the geometric product in Cl(3,0).
# Generated programmatically at module load time.

_BASIS_BLADES = [
    (),       # 0: scalar (1)
    (1,),     # 1: e1
    (2,),     # 2: e2
    (3,),     # 3: e3
    (1, 2),   # 4: e12
    (1, 3),   # 5: e13
    (2, 3),   # 6: e23
    (1, 2, 3),# 7: e123
]
_BLADE_TO_IDX = {b: i for i, b in enumerate(_BASIS_BLADES)}


def _blade_product(a: tuple, b: tuple):
    """
    Compute the geometric product of two basis blades in Cl(3,0).

    Returns (sign: int, result_blade: tuple).
    e_i * e_i = +1 in Cl(3,0).
    """
    sequence = list(a) + list(b)
    sign = 1
    # Insertion sort while counting swaps (each swap flips sign)
    for i in range(1, len(sequence)):
        key = sequence[i]
        j = i - 1
        while j >= 0 and sequence[j] > key:
            sequence[j + 1] = sequence[j]
            sign *= -1
            j -= 1
        sequence[j + 1] = key
    # Cancel identical adjacent pairs (e_k * e_k = +1 in Cl(3,0))
    result: list = []
    k = 0
    while k < len(sequence):
        if k + 1 < len(sequence) and sequence[k] == sequence[k + 1]:
            k += 2  # e_i^2 = +1 in Cl(3,0): cancels with no extra sign change
        else:
            result.append(sequence[k])
            k += 1
    return sign, tuple(result)


def _build_cayley_tables():
    n = len(_BASIS_BLADES)
    sign_table = np.zeros((n, n), dtype=np.float64)
    idx_table = np.zeros((n, n), dtype=np.int32)
    for i, a in enumerate(_BASIS_BLADES):
        for j, b in enumerate(_BASIS_BLADES):
            s, r = _blade_product(a, b)
            sign_table[i, j] = float(s)
            idx_table[i, j] = _BLADE_TO_IDX[r]
    return sign_table, idx_table


_GP_SIGN, _GP_IDX = _build_cayley_tables()


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Geometric product of two Cl(3,0) multivectors *a* and *b* (length-8).

    Returns a new multivector of length 8.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.zeros(8, dtype=np.float64)
    for i in range(8):
        for j in range(8):
            result[_GP_IDX[i, j]] += _GP_SIGN[i, j] * a[i] * b[j]
    return result


def _grade_mask(grade: int) -> np.ndarray:
    """Binary mask selecting components of the given grade."""
    masks = {
        0: [1, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 1, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 1, 1, 1, 0],
        3: [0, 0, 0, 0, 0, 0, 0, 1],
    }
    return np.array(masks[grade], dtype=np.float64)


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Left contraction (grade-lowering inner product)."""
    gp = geometric_product(a, b)
    # Keep only the grade-|grade_b - grade_a| component (simplified)
    return gp * _grade_mask(0)  # scalar part for the MVP


def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Wedge (outer) product – grade-raising.

    The wedge product a∧b retains only those geometric-product terms where
    the result blade has grade equal to grade(a) + grade(b), i.e. the basis
    vectors of a and b are disjoint.  We approximate this by keeping only
    terms where the result index is at least the sum of the input indices
    (a proxy for non-overlapping basis vectors).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.zeros(8, dtype=np.float64)
    for i in range(8):
        for j in range(8):
            # Two basis elements wedge to 0 if they share a basis vector.
            # For Cl(3,0) this is equivalent to the anti-symmetric part.
            if _GP_IDX[i, j] >= i + j:  # rough grade-sum check
                result[_GP_IDX[i, j]] += _GP_SIGN[i, j] * a[i] * b[j]
    return result


# ---------------------------------------------------------------------------
# PerimeterNode
# ---------------------------------------------------------------------------

class PerimeterNode:
    """
    A single GATr-inspired perimeter node operating in Cl(3,0).

    The node maintains a multivector state mv (8 components) and a
    Deep Delta Learning (DDL) running mean.  When the incoming signal
    deviates from the running mean by more than `erasure_threshold`, the
    node applies an erasure mask before forwarding the signal back to the
    liquid core.

    Parameters
    ----------
    state_dim        : dimensionality used for the linear projection in/out
    erasure_threshold: DDL threshold (0–1); higher → more aggressive pruning
    seed             : RNG seed for weight initialisation
    """

    MV_DIM = 8  # Cl(3,0) multivector components

    def __init__(
        self,
        state_dim: int = 16,
        erasure_threshold: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.erasure_threshold = erasure_threshold

        # Project from state_dim → multivector and back
        scale = 1.0 / np.sqrt(state_dim)
        self.W_to_mv: np.ndarray = rng.normal(0, scale, (self.MV_DIM, state_dim))
        self.W_from_mv: np.ndarray = rng.normal(0, scale, (state_dim, self.MV_DIM))

        # Learnable rotor (rotation in GA) – initialised as identity scalar
        self.rotor: np.ndarray = np.zeros(self.MV_DIM, dtype=np.float64)
        self.rotor[0] = 1.0  # scalar = 1 → identity rotation

        # DDL running mean of multivector signal
        self._ddl_mean: np.ndarray = np.zeros(self.MV_DIM, dtype=np.float64)
        self._ddl_momentum: float = 0.9

        # Erasure mask (learned; initialised to all-pass)
        self.erasure_mask: np.ndarray = np.ones(self.MV_DIM, dtype=np.float64)

        # Cache for Echo Ripple protocol
        self._last_mv: np.ndarray = np.zeros(self.MV_DIM, dtype=np.float64)
        self._last_output: np.ndarray = np.zeros(state_dim, dtype=np.float64)

    # ------------------------------------------------------------------

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Project state vector → multivector (grade-1 + scalar)."""
        mv = self.W_to_mv @ x
        # Normalise to keep the multivector on the unit sphere
        norm = np.linalg.norm(mv)
        if norm > 1e-9:
            mv = mv / norm
        return mv

    def _rotate(self, mv: np.ndarray) -> np.ndarray:
        """Apply the learned rotor R to multivector: R * mv * R†."""
        # Sandwich product: result = R * mv * reverse(R)
        # reverse(R) flips signs of grades 2 and 3
        rev_rotor = self.rotor.copy()
        rev_rotor[4:7] *= -1.0  # negate bivector components
        rev_rotor[7] *= -1.0    # negate pseudoscalar
        rotated = geometric_product(self.rotor, geometric_product(mv, rev_rotor))
        return rotated

    def _ddl_erasure(self, mv: np.ndarray) -> np.ndarray:
        """
        Deep Delta Learning erasure step.

        Computes the delta between the incoming multivector and the running
        mean.  Components where the normalised delta exceeds
        `erasure_threshold` are candidates for erasure.
        """
        delta = mv - self._ddl_mean
        # Update running mean (exponential moving average)
        self._ddl_mean = (
            self._ddl_momentum * self._ddl_mean + (1.0 - self._ddl_momentum) * mv
        )
        # Compute per-component normalised magnitude of change
        norm_delta = np.abs(delta) / (np.abs(self._ddl_mean) + 1e-8)
        # Update erasure mask: components with high delta are *kept*;
        # components that are stale (low delta) relative to threshold are erased.
        self.erasure_mask = np.where(
            norm_delta >= self.erasure_threshold, 1.0, 0.0
        )
        # Apply mask to cancel stale information
        return mv * self.erasure_mask

    def _decode(self, mv: np.ndarray) -> np.ndarray:
        """Project multivector → state vector."""
        return self.W_from_mv @ mv

    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process an incoming state vector *x* through the geometric pipeline:
        encode → rotate → DDL erasure → decode.

        Returns a state vector of shape (state_dim,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mv = self._encode(x)
        mv = self._rotate(mv)
        mv = self._ddl_erasure(mv)
        self._last_mv = mv.copy()
        out = self._decode(mv)
        self._last_output = out.copy()
        return out

    def get_weights(self) -> dict:
        return {
            "W_to_mv": self.W_to_mv.copy(),
            "W_from_mv": self.W_from_mv.copy(),
            "rotor": self.rotor.copy(),
            "erasure_threshold": self.erasure_threshold,
        }

    def set_weights(self, weights: dict) -> None:
        self.W_to_mv = np.array(weights["W_to_mv"], dtype=np.float64)
        self.W_from_mv = np.array(weights["W_from_mv"], dtype=np.float64)
        self.rotor = np.array(weights["rotor"], dtype=np.float64)
        self.erasure_threshold = float(weights["erasure_threshold"])


# ---------------------------------------------------------------------------
# PerimeterRing – manages all 18 perimeter nodes
# ---------------------------------------------------------------------------

class PerimeterRing:
    """
    The outer-18 geometric boundary: a ring of PerimeterNodes.

    Parameters
    ----------
    topology          : FlowerOfLife topology
    state_dim         : hidden-state dimensionality (must match LiquidCore)
    erasure_threshold : DDL erasure aggressiveness (0–1)
    seed              : RNG seed
    """

    def __init__(
        self,
        topology,
        state_dim: int = 16,
        erasure_threshold: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        self.topology = topology
        self.state_dim = state_dim
        n_perim = topology.NUM_PERIMETER

        self.nodes: List[PerimeterNode] = [
            PerimeterNode(
                state_dim=state_dim,
                erasure_threshold=erasure_threshold,
                seed=None if seed is None else seed + i,
            )
            for i in range(n_perim)
        ]

    # ------------------------------------------------------------------

    def forward(self, core_boundary_states: np.ndarray) -> np.ndarray:
        """
        Push the outermost core-node states through the perimeter ring.

        Parameters
        ----------
        core_boundary_states : ndarray [18, state_dim] – states of the 18
            core nodes that are adjacent to the perimeter ring.

        Returns
        -------
        echo : ndarray [18, state_dim] – processed echo signals to be fed
            back into the core.
        """
        inp = np.asarray(core_boundary_states, dtype=np.float64)
        if inp.shape != (len(self.nodes), self.state_dim):
            raise ValueError(
                f"Expected shape ({len(self.nodes)}, {self.state_dim}), "
                f"got {inp.shape}"
            )
        echo = np.stack([
            node.forward(inp[i]) for i, node in enumerate(self.nodes)
        ])
        return echo

    def get_erasure_masks(self) -> np.ndarray:
        """Return stacked erasure masks [18, 8]."""
        return np.stack([n.erasure_mask for n in self.nodes])
