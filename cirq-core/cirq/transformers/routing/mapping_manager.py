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

from typing import Dict, TYPE_CHECKING
import networkx as nx

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


class MappingManager:
    """Class that keeps track of the mapping of logical to physical qubits and provides
    convenience methods for distance queries on the physical qubits.

    Qubit variables with the characer 'p' preppended to them are physical and qubits with  the
    character 'l' preppended to them are logical qubits.
    """

    def __init__(self, device_graph: nx.Graph, initial_mapping: Dict[ops.Qid, ops.Qid]) -> None:
        """Initializes MappingManager.

        Args:
            device_graph: connectivity graph of qubits in the hardware device.
            circuit_graph: connectivity graph of the qubits in the input circuit.
            initial_mapping: the initial mapping of logical (keys) to physical qubits (values).
        """
        self.device_graph = device_graph
        self._map = initial_mapping.copy()
        self._inverse_map = {v: k for k, v in self._map.items()}
        self._induced_subgraph = nx.induced_subgraph(self.device_graph, self._map.values())
        self._shortest_paths_matrix = dict(nx.all_pairs_shortest_path(self._induced_subgraph))

    @property
    def map(self) -> Dict[ops.Qid, ops.Qid]:
        """The mapping of logical qubits (keys) to physical qubits (values)."""
        return self._map

    @property
    def inverse_map(self) -> Dict[ops.Qid, ops.Qid]:
        """The mapping of physical qubits (keys) to logical qubits (values)."""
        return self._inverse_map

    @property
    def induced_subgraph(self) -> nx.Graph:
        """The device_graph induced on the physical qubits that are mapped to."""
        return self._induced_subgraph

    def dist_on_device(self, lq1: ops.Qid, lq2: ops.Qid) -> int:
        """Finds shortest path distance path between the corresponding physical qubits for logical
        qubits q1 and q2 on the device.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.

        Returns:
            The shortest path distance.
        """
        return len(self._shortest_paths_matrix[self._map[lq1]][self._map[lq2]]) - 1

    def can_execute(self, op: ops.Operation) -> bool:
        """Finds whether the given operation can be executed on the device.

        Args:
            op: an operation on logical qubits.

        Returns:
            Whether the given operation is executable on the device.
        """
        return protocols.num_qubits(op) < 2 or self.dist_on_device(*op.qubits) == 1

    def apply_swap(self, lq1: ops.Qid, lq2: ops.Qid) -> None:
        """Swaps two logical qubits in the map and in the inverse map.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.
        """
        self._map[lq1], self._map[lq2] = self._map[lq2], self._map[lq1]

        pq1 = self._map[lq1]
        pq2 = self._map[lq2]
        self._inverse_map[pq1], self._inverse_map[pq2] = (
            self._inverse_map[pq2],
            self._inverse_map[pq1],
        )

    def mapped_op(self, op: ops.Operation) -> ops.Operation:
        """Transforms the given operation with the qubits in self._map.

        Args:
            op: an operation on logical qubits.

        Returns:
            The same operation on corresponding physical qubits."""
        return op.transform_qubits(self._map)

    def shortest_path(self, lq1: ops.Qid, lq2: ops.Qid):
        """Find that shortest path between two logical qubits on the device given their mapping."""
        return self._shortest_paths_matrix[self._map[lq1]][self._map[lq2]]
