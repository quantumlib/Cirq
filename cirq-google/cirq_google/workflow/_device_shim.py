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
from typing import Iterable, cast

import networkx as nx

import cirq


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    # cirq contrib: routing.gridqubits_to_graph_device
    def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
        return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)

    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


def _Device_dot_get_nx_graph(device: 'cirq.Device') -> nx.Graph:
    """Shim over future `cirq.Device` method to get a NetworkX graph."""
    return _gridqubits_to_graph_device(cast(Iterable[cirq.GridQubit], device.qubit_set()))
