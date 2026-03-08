"""Tests for the geometric algebra perimeter nodes."""

import pytest
import numpy as np
from aetherforge.topology import FlowerOfLife
from aetherforge.perimeter import (
    geometric_product,
    PerimeterNode,
    PerimeterRing,
)


class TestGeometricProduct:
    def test_scalar_identity(self):
        """e0 * anything = anything (scalar identity element)."""
        e0 = np.zeros(8)
        e0[0] = 1.0
        v = np.random.default_rng(0).standard_normal(8)
        result = geometric_product(e0, v)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_e1_e1_equals_scalar_1(self):
        """e1 * e1 = +1 in Cl(3,0)."""
        e1 = np.zeros(8)
        e1[1] = 1.0
        result = geometric_product(e1, e1)
        expected = np.zeros(8)
        expected[0] = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_e2_e2_equals_scalar_1(self):
        e2 = np.zeros(8)
        e2[2] = 1.0
        result = geometric_product(e2, e2)
        expected = np.zeros(8)
        expected[0] = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_e3_e3_equals_scalar_1(self):
        e3 = np.zeros(8)
        e3[3] = 1.0
        result = geometric_product(e3, e3)
        expected = np.zeros(8)
        expected[0] = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_output_shape(self):
        a = np.random.default_rng(7).standard_normal(8)
        b = np.random.default_rng(8).standard_normal(8)
        result = geometric_product(a, b)
        assert result.shape == (8,)

    def test_anticommutativity_e1_e2(self):
        """e1*e2 = -e2*e1 for orthogonal basis vectors."""
        e1 = np.zeros(8); e1[1] = 1.0
        e2 = np.zeros(8); e2[2] = 1.0
        r1 = geometric_product(e1, e2)
        r2 = geometric_product(e2, e1)
        np.testing.assert_allclose(r1, -r2, atol=1e-12)


class TestPerimeterNode:
    def setup_method(self):
        self.node = PerimeterNode(state_dim=8, erasure_threshold=0.4, seed=42)

    def test_forward_output_shape(self):
        x = np.ones(8)
        out = self.node.forward(x)
        assert out.shape == (8,)

    def test_forward_returns_finite(self):
        x = np.random.default_rng(3).standard_normal(8)
        out = self.node.forward(x)
        assert np.all(np.isfinite(out))

    def test_erasure_mask_binary(self):
        x = np.random.default_rng(5).standard_normal(8)
        self.node.forward(x)
        mask = self.node.erasure_mask
        assert set(mask).issubset({0.0, 1.0})

    def test_rotor_initialised_as_identity(self):
        """Rotor should start as unit scalar (identity rotation)."""
        assert self.node.rotor[0] == 1.0
        np.testing.assert_array_equal(self.node.rotor[1:], np.zeros(7))

    def test_get_set_weights_roundtrip(self):
        w = self.node.get_weights()
        self.node.rotor[:] = 0.0
        self.node.set_weights(w)
        np.testing.assert_array_equal(self.node.rotor, w["rotor"])

    def test_ddl_mean_updates(self):
        x1 = np.ones(8)
        x2 = -np.ones(8)
        self.node.forward(x1)
        mean_after_1 = self.node._ddl_mean.copy()
        self.node.forward(x2)
        mean_after_2 = self.node._ddl_mean.copy()
        # Mean should shift towards x2
        assert not np.allclose(mean_after_1, mean_after_2)


class TestPerimeterRing:
    def setup_method(self):
        self.topo = FlowerOfLife()
        self.ring = PerimeterRing(self.topo, state_dim=8, seed=0)

    def test_num_nodes(self):
        assert len(self.ring.nodes) == 18

    def test_forward_output_shape(self):
        boundary = np.random.default_rng(0).standard_normal((18, 8))
        echo = self.ring.forward(boundary)
        assert echo.shape == (18, 8)

    def test_forward_bad_shape_raises(self):
        with pytest.raises(ValueError):
            self.ring.forward(np.zeros((10, 8)))

    def test_get_erasure_masks_shape(self):
        boundary = np.ones((18, 8))
        self.ring.forward(boundary)
        masks = self.ring.get_erasure_masks()
        assert masks.shape == (18, 8)
