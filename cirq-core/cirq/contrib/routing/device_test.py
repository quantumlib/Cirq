# Copyright 2019 The Cirq Developers
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

import networkx as nx
import pytest

import cirq
import cirq.contrib.routing as ccr


@pytest.mark.parametrize('n_qubits', (2, 5, 11))
def test_get_linear_device_graph(n_qubits):
    graph = ccr.get_linear_device_graph(n_qubits)
    assert sorted(graph) == cirq.LineQubit.range(n_qubits)
    assert len(graph.edges()) == n_qubits - 1
    assert all(abs(a.x - b.x) == 1 for a, b in graph.edges())


def test_nx_qubit_layout():
    grid_qubit_graph = ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(5, 5))
    pos = ccr.nx_qubit_layout(grid_qubit_graph)
    assert len(pos) == len(grid_qubit_graph)
    for k, (x, y) in pos.items():
        assert x == k.col
        assert y == -k.row


def test_nx_qubit_layout_2():
    g = nx.from_edgelist(
        [(cirq.LineQubit(0), cirq.LineQubit(1)), (cirq.LineQubit(1), cirq.LineQubit(2))]
    )
    pos = ccr.nx_qubit_layout(g)
    for k, (x, y) in pos.items():
        assert x == k.x
        assert y == 0.5


def test_nx_qubit_layout_3():
    g = nx.from_edgelist(
        [(cirq.NamedQubit('a'), cirq.NamedQubit('b')), (cirq.NamedQubit('b'), cirq.NamedQubit('c'))]
    )
    node_to_i = {cirq.NamedQubit('a'): 0, cirq.NamedQubit('b'): 1, cirq.NamedQubit('c'): 2}

    pos = ccr.nx_qubit_layout(g)
    for k, (x, y) in pos.items():
        assert x == 0.5
        assert y == node_to_i[k] + 1
