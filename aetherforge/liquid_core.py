"""
liquid_core.py – Liquid Neural Network (LNN) nodes for the inner-19 core.

Each LiquidNode integrates a first-order ODE of the form

    τ(u) · dx/dt = −x + f(W_rec · x + W_in · u + b)

where τ(u) is a *liquid time-constant* – a scalar gating value that depends
on the current input, allowing the node to compress its memory horizon for
fast-changing inputs or expand it for long-term reasoning.

Integration is performed with a fixed-step 4th-order Runge-Kutta (RK4)
solver implemented in pure NumPy so that no additional ODE library is
required for the MVP.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def _rk4_step(
    f,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Single RK4 integration step for dx/dt = f(x, u)."""
    k1 = f(x, u)
    k2 = f(x + dt / 2.0 * k1, u)
    k3 = f(x + dt / 2.0 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class LiquidNode:
    """
    A single Liquid Neural Network node.

    Parameters
    ----------
    state_dim  : dimensionality of the hidden state vector x
    input_dim  : dimensionality of the incoming input u
    tau_init   : initial (base) liquid time-constant (>0)
    dt         : ODE integration timestep
    seed       : optional RNG seed for weight initialisation
    """

    def __init__(
        self,
        state_dim: int = 16,
        input_dim: int = 16,
        tau_init: float = 1.0,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(state_dim)

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt

        # Learnable parameters.
        # W_rec is initialised with spectral radius ≈ 0.9 (echo-state init)
        # to keep the ODE dynamics stable for arbitrary inputs.
        W_raw = rng.normal(0, scale, (state_dim, state_dim))
        sr = float(np.max(np.abs(np.linalg.eigvals(W_raw))))
        self.W_rec: np.ndarray = W_raw * (0.9 / sr) if sr > 1e-9 else W_raw
        self.W_in: np.ndarray = rng.normal(0, scale, (state_dim, input_dim))
        self.b: np.ndarray = np.zeros(state_dim)

        # Liquid time-constant gate weights: τ(u) = softplus(v_tau · u + b_tau)
        self.v_tau: np.ndarray = rng.normal(0, scale, input_dim)
        self.b_tau: float = 0.0
        self._tau_base: float = float(tau_init)

        # Hidden state
        self.state: np.ndarray = np.zeros(state_dim)

        # Cache last forward activation for the Echo Ripple protocol
        self._last_activation: np.ndarray = np.zeros(state_dim)

    # ------------------------------------------------------------------
    # Internal ODE dynamics
    # ------------------------------------------------------------------

    def _tau(self, u: np.ndarray) -> float:
        """Liquid time-constant: τ(u) = τ_base · softplus(v_tau·u + b_tau)."""
        z = float(np.dot(self.v_tau, u)) + self.b_tau
        tau_mod = self._tau_base * float(np.log1p(np.exp(z)))
        # Clamp to a reasonable range to avoid numerical blow-up
        return float(np.clip(tau_mod, 0.01, 100.0))

    @staticmethod
    def _activation(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def _dxdt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        tau = self._tau(u)
        pre = self.W_rec @ x + self.W_in @ u + self.b
        return (-x + self._activation(pre)) / tau

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Advance the node by one ODE timestep given input *u*.

        Returns the new hidden state.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        if u.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {u.shape[0]}"
            )
        new_state = _rk4_step(self._dxdt, self.state, u, self.dt)
        # Clip to a safe range to guard against transient numerical blow-up.
        self.state = np.clip(new_state, -10.0, 10.0)
        self._last_activation = self.state.copy()
        return self.state.copy()

    def reset(self) -> None:
        """Reset hidden state to zero."""
        self.state[:] = 0.0
        self._last_activation[:] = 0.0

    def get_weights(self) -> dict:
        return {
            "W_rec": self.W_rec.copy(),
            "W_in": self.W_in.copy(),
            "b": self.b.copy(),
            "v_tau": self.v_tau.copy(),
            "b_tau": self._tau_base,
        }

    def set_weights(self, weights: dict) -> None:
        self.W_rec = np.array(weights["W_rec"], dtype=np.float64)
        self.W_in = np.array(weights["W_in"], dtype=np.float64)
        self.b = np.array(weights["b"], dtype=np.float64)
        self.v_tau = np.array(weights["v_tau"], dtype=np.float64)
        self._tau_base = float(weights["b_tau"])


# ---------------------------------------------------------------------------
# LiquidCore – manages all 19 inner nodes as a coupled system
# ---------------------------------------------------------------------------

class LiquidCore:
    """
    The inner-19 liquid core: a coupled network of LiquidNodes whose
    states are exchanged via the FlowerOfLife adjacency matrix each step.

    Parameters
    ----------
    topology    : FlowerOfLife topology (used for adjacency)
    state_dim   : hidden-state dimensionality per node
    dt          : ODE timestep
    """

    def __init__(
        self,
        topology,
        state_dim: int = 16,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.topology = topology
        self.state_dim = state_dim
        n_core = topology.NUM_CORE

        self.nodes: List[LiquidNode] = [
            LiquidNode(
                state_dim=state_dim,
                input_dim=state_dim,
                dt=dt,
                seed=None if seed is None else seed + i,
            )
            for i in range(n_core)
        ]

        # Coupling weight matrix between core nodes [n_core × n_core]
        rng = np.random.default_rng(seed)
        adj = topology.adjacency[: n_core, :n_core].astype(np.float64)
        # Normalise each row so coupled inputs sum to ≤1
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.W_couple: np.ndarray = adj / row_sums

    # ------------------------------------------------------------------

    def step(self, external_inputs: np.ndarray) -> np.ndarray:
        """
        Advance all core nodes by one ODE timestep.

        Parameters
        ----------
        external_inputs : ndarray [n_core, state_dim] – external stimuli,
            one per node (zeros for undriven nodes).

        Returns
        -------
        states : ndarray [n_core, state_dim]
        """
        n = len(self.nodes)
        ext = np.asarray(external_inputs, dtype=np.float64)
        if ext.shape != (n, self.state_dim):
            raise ValueError(
                f"external_inputs must be shape ({n}, {self.state_dim}), "
                f"got {ext.shape}"
            )

        # Collect current states for coupling
        states = np.stack([node.state for node in self.nodes])  # [n, d]

        new_states = np.empty_like(states)
        for i, node in enumerate(self.nodes):
            # Combine coupled neighbour signal + external input
            coupled = self.W_couple[i] @ states  # [d]
            u = coupled + ext[i]
            new_states[i] = node.step(u)

        return new_states

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()

    @property
    def states(self) -> np.ndarray:
        """Current hidden states as [n_core, state_dim] array."""
        return np.stack([node.state for node in self.nodes])

    @property
    def activations(self) -> np.ndarray:
        """Last cached activations as [n_core, state_dim] array."""
        return np.stack([node._last_activation for node in self.nodes])
