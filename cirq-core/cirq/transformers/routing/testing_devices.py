# Copyright 2022 The Cirq Developers
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

from typing import Optional
import networkx as nx

import cirq


class GridTestingDevice(cirq.Device):
    def __init__(self, metadata: cirq.DeviceMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> Optional[cirq.DeviceMetadata]:
        return self._metadata

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}.')

        #TODO: update if stattement so it doesn't use qubit_pairs
        if (
            len(operation.qubits) == 2
            and frozenset(operation.qubits) not in self._metadata.qubit_pairs
        ):
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')

def construct_grid_device(d: int) -> GridTestingDevice:
    qubits = (cirq.GridQubit(i,j) for i in range(d) for j in range(d))

    nx_graph = nx.Graph()
    row_edges = [(cirq.GridQubit(i,j), cirq.GridQubit(i,j+1)) for i in range(d) for j in range(d-1)]
    col_edges = [(cirq.GridQubit(i,j), cirq.GridQubit(i+1,j)) for j in range(d) for i in range(d-1)]
    nx_graph.add_edges_from(row_edges)
    nx_graph.add_edges_from(col_edges)

    metadata = cirq.DeviceMetadata(qubits, nx_graph)
    return GridTestingDevice(metadata)