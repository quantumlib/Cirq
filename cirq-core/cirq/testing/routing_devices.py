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

from importlib_metadata import metadata
import networkx as nx

from cirq import devices, ops

if TYPE_CHECKING:
    import cirq


class RoutingTestingDevice(devices.Device):
    """Testing device to be used only for testing qubit connectivity in routing procedures."""

    def __init__(self, nx_graph: nx.Graph, qubit_type: str = 'NamedQubit') -> None:
        if qubit_type == 'GridQubit':
            relabeling_map = {old: devices.GridQubit(*old) for old in nx_graph}
        elif qubit_type == 'LineQubit':
            relabeling_map = {old: devices.LineQubit(old) for old in nx_graph}
        else:
            relabeling_map = {old: ops.NamedQubit(str(old)) for old in nx_graph}

        # Relabel nodes in-place.
        nx.relabel_nodes(nx_graph, relabeling_map, copy=False)

        self._metadata = devices.DeviceMetadata(relabeling_map.values(), nx_graph)

    @property
    def metadata(self) -> Optional[devices.DeviceMetadata]:
        return self._metadata

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}.')

        if len(operation.qubits) == 2 and operation.qubits not in self._metadata.nx_graph.edges:
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')


def construct_grid_device(m: int, n: int) -> RoutingTestingDevice:
    return RoutingTestingDevice(nx.grid_2d_graph(m, n), qubit_type="GridQubit")


def construct_ring_device(l: int, directed: bool = False) -> RoutingTestingDevice:
    if directed:
        # If create_using is directed, the direction is in increasing order.
        nx_graph = nx.cycle_graph(l, create_using=nx.DiGraph)
    else:
        nx_graph = nx.cycle_graph(l)

    return RoutingTestingDevice(nx_graph, qubit_type="LineQubit")
