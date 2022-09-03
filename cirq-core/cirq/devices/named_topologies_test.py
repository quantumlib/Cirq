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
from unittest.mock import MagicMock

import cirq
import networkx as nx
import pytest
from cirq import (
    draw_gridlike,
    LineTopology,
    TiltedSquareLattice,
    get_placements,
    draw_placements,
    is_valid_placement,
)


@pytest.mark.parametrize('width, height', list(itertools.product([1, 2, 3, 24], repeat=2)))
def test_tilted_square_lattice(width, height):
    topo = TiltedSquareLattice(width, height)
    assert topo.graph.number_of_edges() == width * height
    assert all(1 <= topo.graph.degree[node] <= 4 for node in topo.graph.nodes)
    assert topo.name == f'tilted-square-lattice-{width}-{height}'
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert nx.is_connected(topo.graph)
    assert nx.algorithms.planarity.check_planarity(topo.graph)

    cirq.testing.assert_equivalent_repr(topo)


def test_bad_tilted_square_lattice():
    with pytest.raises(ValueError):
        _ = TiltedSquareLattice(0, 3)
    with pytest.raises(ValueError):
        _ = TiltedSquareLattice(3, 0)


def test_tilted_square_methods():
    topo = TiltedSquareLattice(5, 5)
    ax = MagicMock()
    topo.draw(ax=ax)
    ax.scatter.assert_called()

    qubits = topo.nodes_as_gridqubits()
    assert all(isinstance(q, cirq.GridQubit) for q in qubits)

    mapping = topo.nodes_to_gridqubits(offset=(3, 5))
    assert all(
        isinstance(q, cirq.GridQubit) and q >= cirq.GridQubit(0, 0) for q in mapping.values()
    )


def test_tilted_square_lattice_n_nodes():
    for width, height in itertools.product(list(range(1, 4 + 1)), repeat=2):
        topo = TiltedSquareLattice(width, height)
        assert topo.n_nodes == topo.graph.number_of_nodes()


def test_line_topology():
    n = 10
    topo = LineTopology(n)
    assert topo.n_nodes == n
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert all(1 <= topo.graph.degree[node] <= 2 for node in topo.graph.nodes)
    assert topo.name == 'line-10'

    ax = MagicMock()
    topo.draw(ax=ax)
    ax.scatter.assert_called()

    with pytest.raises(ValueError, match='greater than 1.*'):
        _ = LineTopology(1)
    assert LineTopology(2).n_nodes == 2
    assert LineTopology(2).graph.number_of_nodes() == 2

    mapping = topo.nodes_to_linequbits(offset=3)
    assert sorted(mapping.keys()) == list(range(n))
    assert all(isinstance(q, cirq.LineQubit) for q in mapping.values())
    assert all(mapping[x] == cirq.LineQubit(x + 3) for x in mapping)

    cirq.testing.assert_equivalent_repr(topo)


def test_line_topology_nodes_as_qubits():
    for n in range(2, 10, 2):
        assert LineTopology(n).nodes_as_linequbits() == cirq.LineQubit.range(n)


@pytest.mark.parametrize('tilted', [True, False])
def test_draw_gridlike(tilted):
    graph = nx.grid_2d_graph(3, 3)
    ax = MagicMock()
    pos = draw_gridlike(graph, tilted=tilted, ax=ax)
    ax.scatter.assert_called()
    for (row, column), _ in pos.items():
        assert 0 <= row < 3
        assert 0 <= column < 3


@pytest.mark.parametrize('tilted', [True, False])
def test_draw_gridlike_qubits(tilted):
    graph = nx.grid_2d_graph(3, 3)
    graph = nx.relabel_nodes(graph, {(r, c): cirq.GridQubit(r, c) for r, c in sorted(graph.nodes)})
    ax = MagicMock()
    pos = draw_gridlike(graph, tilted=tilted, ax=ax)
    ax.scatter.assert_called()
    for q, _ in pos.items():
        assert 0 <= q.row < 3
        assert 0 <= q.col < 3


def test_get_placements():
    topo = TiltedSquareLattice(4, 2)
    syc23 = TiltedSquareLattice(8, 4).graph
    placements = get_placements(syc23, topo.graph)
    assert len(placements) == 12

    axes = [MagicMock() for _ in range(4)]
    draw_placements(syc23, topo.graph, placements[::3], axes=axes)
    for ax in axes:
        ax.scatter.assert_called()


def test_is_valid_placement():
    topo = TiltedSquareLattice(4, 2)
    syc23 = TiltedSquareLattice(8, 4).graph
    placements = get_placements(syc23, topo.graph)
    for placement in placements:
        assert is_valid_placement(syc23, topo.graph, placement)

    bad_placement = topo.nodes_to_gridqubits(offset=(100, 100))
    assert not is_valid_placement(syc23, topo.graph, bad_placement)
