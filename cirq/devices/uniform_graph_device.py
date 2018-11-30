# Copyright 2018 The Cirq Developers
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

from typing import Iterable, Mapping, Optional, Union

from cirq import devices


def uniform_undirected_graph_device(
        edges: Iterable[Iterable],
        edge_label: devices.UndirectedGraphDeviceEdge=None
        ) -> devices.UndirectedGraphDevice:
    """An undirected graph device all of whose edges are the same.

    Args:
        edges: The edges.
        edge_label: The label to apply to all edges. Defaults to None.
    """

    labelled_edges = {frozenset(edge): edge_label for edge in edges}
    device_graph = devices.UndirectedHypergraph(labelled_edges)
    return devices.UndirectedGraphDevice(device_graph=device_graph)


def uniform_undirected_linear_device(
        n_qubits: int,
        edge_labels: Mapping[int,
                             Optional[devices.UndirectedGraphDeviceEdge]]
        ) -> devices.UndirectedGraphDevice:
    """A uniform , undirected graph device whose qubits are arranged
    on a line.

    Uniformity refers to the fact that all edges of the same size have the same
    label.

    Args:
        n_qubits: The number of qubits.
        edge_labels: The label to apply to all edges
    """

    if min(edge_labels) < 0:
        raise ValueError('arities must be non-negative')

    device = devices.UndirectedGraphDevice()
    for arity, label in edge_labels.items():
        edges = (range(i, i + arity) for i in range(i, n_qubits - arity))
        device += uniform_undirected_graph_device(edges, label)
    return device
