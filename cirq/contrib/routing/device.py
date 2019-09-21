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
from typing import Iterable, Tuple

import cirq
import networkx as nx


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
