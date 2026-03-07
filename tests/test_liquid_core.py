"""Tests for the Liquid Core (LNN + ODE dynamics)."""

import pytest
import numpy as np
from aetherforge.topology import FlowerOfLife
from aetherforge.liquid_core import LiquidNode, LiquidCore


class TestLiquidNode:
    def setup_method(self):
        self.node = LiquidNode(state_dim=8, input_dim=8, seed=42)

    def test_initial_state_zeros(self):
        np.testing.assert_array_equal(self.node.state, np.zeros(8))

    def test_step_returns_correct_shape(self):
        u = np.ones(8)
        out = self.node.step(u)
        assert out.shape == (8,)

    def test_step_changes_state(self):
        u = np.random.default_rng(0).standard_normal(8)
        self.node.step(u)
        assert not np.allclose(self.node.state, np.zeros(8))

    def test_reset_clears_state(self):
        self.node.step(np.ones(8))
        self.node.reset()
        np.testing.assert_array_equal(self.node.state, np.zeros(8))

    def test_bad_input_dim_raises(self):
        with pytest.raises(ValueError):
            self.node.step(np.ones(5))

    def test_get_set_weights_roundtrip(self):
        w = self.node.get_weights()
        self.node.W_rec[:] = 0.0
        self.node.set_weights(w)
        np.testing.assert_array_equal(self.node.W_rec, w["W_rec"])

    def test_liquid_time_constant_positive(self):
        """tau(u) must remain positive for any input."""
        for _ in range(20):
            u = np.random.default_rng(99).standard_normal(8)
            tau = self.node._tau(u)
            assert tau > 0, f"tau={tau} is not positive"

    def test_state_bounded(self):
        """After many steps the state should not blow up (tanh keeps it bounded)."""
        u = np.ones(8) * 10.0
        for _ in range(100):
            self.node.step(u)
        assert np.all(np.abs(self.node.state) < 100)


class TestLiquidCore:
    def setup_method(self):
        self.topo = FlowerOfLife()
        self.core = LiquidCore(self.topo, state_dim=8, seed=0)

    def test_num_nodes(self):
        assert len(self.core.nodes) == 19

    def test_step_output_shape(self):
        ext = np.zeros((19, 8))
        out = self.core.step(ext)
        assert out.shape == (19, 8)

    def test_step_bad_shape_raises(self):
        with pytest.raises(ValueError):
            self.core.step(np.zeros((5, 8)))

    def test_states_property(self):
        states = self.core.states
        assert states.shape == (19, 8)

    def test_reset_clears_all_states(self):
        ext = np.random.default_rng(1).standard_normal((19, 8))
        self.core.step(ext)
        self.core.reset()
        np.testing.assert_array_equal(self.core.states, np.zeros((19, 8)))

    def test_coupling_matrix_rows_sum_to_one_or_zero(self):
        W = self.core.W_couple
        assert W.shape == (19, 19)
        row_sums = W.sum(axis=1)
        for s in row_sums:
            assert 0.0 <= s <= 1.0 + 1e-9
