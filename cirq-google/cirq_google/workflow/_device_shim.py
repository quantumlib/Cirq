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

import cirq
import networkx as nx


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if pair[0].is_adjacent(pair[1])
    )


def _Device_dot_get_nx_graph(device: 'cirq.Device') -> nx.Graph:
    """Shim over future `cirq.Device` method to get a NetworkX graph."""
    if device.metadata is not None:
        return device.metadata.nx_graph
    raise ValueError('Supplied device must contain metadata.')
