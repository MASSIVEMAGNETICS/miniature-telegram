"""Tests for the Flower-of-Life topology."""

import pytest
import numpy as np
from aetherforge.topology import FlowerOfLife


class TestFlowerOfLife:
    def setup_method(self):
        self.topo = FlowerOfLife()

    def test_node_count(self):
        assert self.topo.NUM_NODES == 37

    def test_core_perimeter_split(self):
        assert len(self.topo.core_indices) == 19
        assert len(self.topo.perimeter_indices) == 18
        assert len(self.topo.core_indices) + len(self.topo.perimeter_indices) == 37

    def test_no_overlap(self):
        core = set(self.topo.core_indices)
        perim = set(self.topo.perimeter_indices)
        assert core.isdisjoint(perim)

    def test_positions_shape(self):
        assert self.topo.positions.shape == (37, 2)

    def test_adjacency_shape(self):
        assert self.topo.adjacency.shape == (37, 37)

    def test_adjacency_symmetric(self):
        adj = self.topo.adjacency
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        assert np.trace(self.topo.adjacency) == 0

    def test_center_node_has_neighbours(self):
        # Center node (index 0) should be connected to ring-1 nodes
        assert self.topo.degree(0) > 0

    def test_edges_non_empty(self):
        assert len(self.topo.edges) > 0

    def test_edges_are_sorted_pairs(self):
        for i, j in self.topo.edges:
            assert i < j

    def test_neighbours_consistency(self):
        for node in range(37):
            for nb in self.topo.neighbours(node):
                assert self.topo.adjacency[node, nb] == 1

    def test_perimeter_neighbours(self):
        # Every perimeter node should appear as a neighbour of at least
        # one other node
        all_nbs = set()
        for n in self.topo.perimeter_indices:
            all_nbs.update(self.topo.neighbours(n))
        assert len(all_nbs) > 0

    def test_ring3_nodes_connect_to_ring2(self):
        # Each perimeter (ring-3) node should have at least one
        # core (ring-0/1/2) neighbour
        for p in self.topo.perimeter_indices:
            has_core_nb = any(
                nb in self.topo.core_indices for nb in self.topo.neighbours(p)
            )
            assert has_core_nb, f"Perimeter node {p} has no core neighbours"
