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
"""Provides test devices that can validate circuits during a routing procedure."""

from typing import Optional, TYPE_CHECKING
import networkx as nx

from cirq import devices

if TYPE_CHECKING:
    import cirq


class RoutingTestingDevice(devices.Device):
    """Testing device to be used only for testing qubit connectivity in routing procedures."""

    def __init__(self, metadata: devices.DeviceMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> Optional[devices.DeviceMetadata]:
        return self._metadata

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}.')

        if len(operation.qubits) == 2 and operation.qubits not in self._metadata.nx_graph.edges:
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')


def construct_square_device(d: int) -> RoutingTestingDevice:
    qubits = devices.GridQubit.square(d)

    nx_graph = nx.Graph()
    row_edges = [
        (devices.GridQubit(i, j), devices.GridQubit(i, j + 1))
        for i in range(d)
        for j in range(d - 1)
    ]
    col_edges = [
        (devices.GridQubit(i, j), devices.GridQubit(i + 1, j))
        for j in range(d)
        for i in range(d - 1)
    ]
    nx_graph.add_edges_from(row_edges)
    nx_graph.add_edges_from(col_edges)

    metadata = devices.DeviceMetadata(qubits, nx_graph)
    return RoutingTestingDevice(metadata)


def construct_ring_device(d: int, directed: bool = False) -> RoutingTestingDevice:
    qubits = devices.LineQubit.range(d)
    if directed:
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()
    edges = [(qubits[i % d], qubits[(i + 1) % d]) for i in range(d)]
    nx_graph.add_edges_from(edges)

    metadata = devices.DeviceMetadata(qubits, nx_graph)
    return RoutingTestingDevice(metadata)
