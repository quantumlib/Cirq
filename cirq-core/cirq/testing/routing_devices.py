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

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from cirq import devices, ops

if TYPE_CHECKING:
    import cirq


class RoutingTestingDevice(devices.Device):
    """Testing device to be used for testing qubit connectivity in routing procedures."""

    def __init__(self, nx_graph: nx.Graph) -> None:
        relabeling_map = {
            old: ops.q(old) if isinstance(old, (int, str)) else ops.q(*old) for old in nx_graph
        }
        # Relabel nodes in-place.
        nx.relabel_nodes(nx_graph, relabeling_map, copy=False)

        self._metadata = devices.DeviceMetadata(relabeling_map.values(), nx_graph)

    @property
    def metadata(self) -> devices.DeviceMetadata:
        return self._metadata

    def validate_operation(self, operation: cirq.Operation) -> None:
        if not self._metadata.qubit_set.issuperset(operation.qubits):
            raise ValueError(f'Qubits not on device: {operation.qubits!r}.')

        if len(operation.qubits) > 1:
            if len(operation.qubits) == 2:
                if operation.qubits not in self._metadata.nx_graph.edges:
                    raise ValueError(
                        f'Qubit pair is not a valid edge on device: {operation.qubits!r}.'
                    )
                return

            if not isinstance(operation.gate, ops.MeasurementGate):
                raise ValueError(
                    f'Unsupported operation: {operation}. '
                    f'Routing device only supports 1 / 2 qubit operations.'
                )


def construct_grid_device(m: int, n: int) -> RoutingTestingDevice:
    return RoutingTestingDevice(nx.grid_2d_graph(m, n))


def construct_ring_device(l: int, directed: bool = False) -> RoutingTestingDevice:
    nx_graph = nx.cycle_graph(l, create_using=nx.DiGraph if directed else nx.Graph)
    return RoutingTestingDevice(nx_graph)
