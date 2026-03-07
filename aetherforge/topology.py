"""
topology.py – Flower-of-Life 37-node graph topology.

Layout
------
The Flower of Life is a hexagonally-packed arrangement:

    Ring 0 (center) : 1 node   – index 0
    Ring 1          : 6 nodes  – indices 1-6
    Ring 2          : 12 nodes – indices 7-18   (inner core total = 19)
    Ring 3          : 18 nodes – indices 19-36  (perimeter)

Edges connect every node to all topologically adjacent nodes (shared
boundary in the hex grid).  Self-loops are excluded.

Coordinate system
-----------------
Nodes are placed in 2-D axial hex coordinates and then converted to
Cartesian (x, y) for distance-based edge construction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hex-grid helpers
# ---------------------------------------------------------------------------

def _axial_to_cartesian(q: int, r: int, radius: float = 1.0) -> Tuple[float, float]:
    """Convert axial hex coordinates (q, r) to Cartesian (x, y)."""
    x = radius * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = radius * (3.0 / 2.0 * r)
    return x, y


# Hex direction vectors for the ring walk (redblobgames convention):
# start at (-ring, ring) and walk each segment for `ring` steps.
_DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def _ring_coords_fixed(ring: int) -> List[Tuple[int, int]]:
    """
    Return axial (q, r) coordinates for every cell in *ring* of a hex grid.
    Ring 0 → [(0, 0)], ring k>0 → 6k cells, no duplicates with other rings.
    """
    if ring == 0:
        return [(0, 0)]
    coords: List[Tuple[int, int]] = []
    # Starting corner: (-ring, ring) ensures no overlap with ring-0 or centre
    q, r = -ring, ring
    for dq, dr in _DIRECTIONS:
        for _ in range(ring):
            coords.append((q, r))
            q += dq
            r += dr
    return coords


# ---------------------------------------------------------------------------
# FlowerOfLife
# ---------------------------------------------------------------------------

class FlowerOfLife:
    """
    37-node Flower-of-Life graph topology.

    Attributes
    ----------
    num_nodes      : int – total nodes (37)
    num_core_nodes : int – inner liquid-core nodes (19)
    num_perimeter_nodes : int – outer perimeter nodes (18)
    positions      : ndarray [37, 2] – Cartesian (x, y) positions
    adjacency      : ndarray [37, 37] – symmetric binary adjacency matrix
    edges          : list[(int,int)] – undirected edge list
    core_indices   : list[int] – node indices belonging to the inner core
    perimeter_indices : list[int] – node indices belonging to the perimeter
    """

    NUM_NODES = 37
    NUM_CORE = 19       # rings 0-2
    NUM_PERIMETER = 18  # ring 3

    def __init__(self, node_spacing: float = 1.0):
        self.node_spacing = node_spacing
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        axial: List[Tuple[int, int]] = []
        for ring in range(4):  # rings 0, 1, 2, 3
            axial.extend(_ring_coords_fixed(ring))

        assert len(axial) == self.NUM_NODES, (
            f"Expected {self.NUM_NODES} nodes, got {len(axial)}"
        )

        self.positions = np.array(
            [_axial_to_cartesian(q, r, self.node_spacing) for q, r in axial],
            dtype=np.float64,
        )

        # Core vs perimeter split
        self.core_indices: List[int] = list(range(self.NUM_CORE))
        self.perimeter_indices: List[int] = list(
            range(self.NUM_CORE, self.NUM_NODES)
        )

        # Build adjacency: two nodes are neighbours if their Euclidean
        # distance ≈ √3 × node_spacing (hex nearest-neighbour spacing).
        threshold = self.node_spacing * math.sqrt(3) * 1.05
        adj = np.zeros((self.NUM_NODES, self.NUM_NODES), dtype=np.int32)
        for i in range(self.NUM_NODES):
            for j in range(i + 1, self.NUM_NODES):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d <= threshold:
                    adj[i, j] = 1
                    adj[j, i] = 1

        self.adjacency: np.ndarray = adj
        self.edges: List[Tuple[int, int]] = [
            (int(i), int(j))
            for i in range(self.NUM_NODES)
            for j in range(i + 1, self.NUM_NODES)
            if adj[i, j]
        ]

    # ------------------------------------------------------------------
    def neighbours(self, node: int) -> List[int]:
        """Return node indices adjacent to *node*."""
        return [int(j) for j in np.where(self.adjacency[node] == 1)[0]]

    def degree(self, node: int) -> int:
        return int(self.adjacency[node].sum())

    def core_neighbours(self, node: int) -> List[int]:
        """Neighbours of *node* that belong to the inner core."""
        return [n for n in self.neighbours(node) if n in self.core_indices]

    def perimeter_neighbours(self, node: int) -> List[int]:
        """Neighbours of *node* that belong to the perimeter ring."""
        return [n for n in self.neighbours(node) if n in self.perimeter_indices]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FlowerOfLife(nodes={self.NUM_NODES}, "
            f"core={self.NUM_CORE}, perimeter={self.NUM_PERIMETER}, "
            f"edges={len(self.edges)})"
        )
