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

import itertools
from typing import Iterable, Tuple, Dict

import networkx as nx

import cirq


def xmon_device_to_graph(device: cirq.google.XmonDevice) -> nx.Graph:
    """Gets the graph of an XmonDevice."""
    return gridqubits_to_graph_device(device.qubits)


def get_linear_device_graph(n_qubits: int) -> nx.Graph:
    """Gets the graph of a linearly connected device."""
    qubits = cirq.LineQubit.range(n_qubits)
    edges = [tuple(qubits[i:i + 2]) for i in range(n_qubits - 1)]
    return nx.Graph(edges)


def get_grid_device_graph(*args, **kwargs) -> nx.Graph:
    """Gets the graph of a grid of qubits.

    See GridQubit.rect for argument details."""
    return gridqubits_to_graph_device(cirq.GridQubit.rect(*args, **kwargs))


def gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    """Gets the graph of a set of grid qubits."""
    return nx.Graph(pair for pair in itertools.combinations(qubits, 2)
                    if _manhattan_distance(*pair) == 1)


def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
    return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)


def nx_qubit_layout(graph: nx.Graph) \
        -> Dict[cirq.Qid, Tuple[float, float]]:
    """Return a layout for a graph for nodes which are qubits.

    This can be used in place of nx.spring_layout or other networkx layouts.
    GridQubits are positioned according to their row/col. LineQubits are
    positioned in a line.

    >>> import cirq.contrib.routing as ccr
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> # Clear plot state to prevent issues with pyplot dimensionality.
    >>> plt.clf()
    >>> g = ccr.xmon_device_to_graph(cirq.google.Foxtail)
    >>> pos = ccr.nx_qubit_layout(g)
    >>> nx.draw_networkx(g, pos=pos)

    """
    pos: Dict[cirq.Qid, Tuple[float, float]] = {}

    _node_to_i_cache = None
    for node in graph.nodes:
        if isinstance(node, cirq.GridQubit):
            pos[node] = (node.col, -node.row)
        elif isinstance(node, cirq.LineQubit):
            # Offset to avoid overlap with gridqubits
            pos[node] = (node.x, 0.5)
        else:
            if _node_to_i_cache is None:
                _node_to_i_cache = {
                    n: i for i, n in enumerate(sorted(graph.nodes))
                }
            # Position in a line according to sort order
            # Offset to avoid overlap with gridqubits
            pos[node] = (0.5, _node_to_i_cache[node] + 1)
    return pos
