# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
from typing import Iterable
from unittest.mock import MagicMock

import cirq
import networkx as nx
import pytest
from cirq_google.named_topologies import (
    NamedTopology,
    draw_gridlike,
    LineTopology,
    DiagonalRectangleTopology,
    get_placements,
    plot_placements,
)
import cirq_google as cg


def test_diagonal_rectangle_topology():
    width = 2
    height = 3
    topo = DiagonalRectangleTopology(width, height)
    assert all(1 <= topo.graph.degree[node] <= 4 for node in topo.graph.nodes)
    assert topo.name == '2-3-diagonal-rectangle'
    assert topo.n_nodes == topo.graph.number_of_nodes()

    with pytest.raises(ValueError):
        _ = DiagonalRectangleTopology(0, 3)
    with pytest.raises(ValueError):
        _ = DiagonalRectangleTopology(3, 0)


def test_line_topology():
    n = 10
    topo = LineTopology(n)
    assert topo.n_nodes == n
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert all(1 <= topo.graph.degree[node] <= 2 for node in topo.graph.nodes)
    assert topo.name == '10-line'


@pytest.mark.parametrize('cartesian', [True, False])
def test_draw_gridlike(cartesian):
    graph = nx.grid_2d_graph(3, 3)
    ax = MagicMock()
    pos = draw_gridlike(graph, cartesian=cartesian, ax=ax)
    ax.scatter.assert_called()
    for (row, column), (x, y) in pos.items():
        assert 0 <= row < 3
        assert 0 <= column < 3


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    # cirq contrib routing --> gridqubits_to_graph_device
    def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
        return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)

    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


def test_get_placements():
    topo = DiagonalRectangleTopology(2, 1)
    syc23 = _gridqubits_to_graph_device(cg.Sycamore23.qubits)
    placements = get_placements(syc23, topo.graph)
    assert len(placements) == 12

    axes = [MagicMock() for _ in range(4)]
    plot_placements(syc23, topo.graph, placements[::3], axes=axes)
    for ax in axes:
        ax.scatter.assert_called()
