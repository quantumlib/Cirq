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

from typing import Dict, Sequence, TYPE_CHECKING
from cirq._compat import cached_method
import networkx as nx

from cirq import protocols

if TYPE_CHECKING:
    import cirq


class MappingManager:
    """Class that keeps track of the mapping of logical to physical qubits.

    Convenience methods over distance and mapping queries of the physical qubits are also provided.
    All such public methods of this class expect logical qubits.
    """

    def __init__(
        self, device_graph: nx.Graph, initial_mapping: Dict['cirq.Qid', 'cirq.Qid']
    ) -> None:
        """Initializes MappingManager.

        Args:
            device_graph: connectivity graph of qubits in the hardware device.
            initial_mapping: the initial mapping of logical (keys) to physical qubits (values).
        """
        self.device_graph = device_graph
        self._map = initial_mapping.copy()
        self._inverse_map = {v: k for k, v in self._map.items()}
        self._induced_subgraph = nx.induced_subgraph(self.device_graph, self._map.values())

    @property
    def map(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """The mapping of logical qubits (keys) to physical qubits (values)."""
        return self._map

    @property
    def inverse_map(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """The mapping of physical qubits (keys) to logical qubits (values)."""
        return self._inverse_map

    @property
    def induced_subgraph(self) -> nx.Graph:
        """The induced subgraph on the set of physical qubits which are part of `self.map`."""
        return self._induced_subgraph

    def dist_on_device(self, lq1: 'cirq.Qid', lq2: 'cirq.Qid') -> int:
        """Finds distance between logical qubits q1 and q2 on the device.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.

        Returns:
            The shortest path distance.
        """
        return len(self._physical_shortest_path(self._map[lq1], self._map[lq2])) - 1

    def can_execute(self, op: 'cirq.Operation') -> bool:
        """Finds whether the given operation can be executed on the device.

        Args:
            op: an operation on logical qubits.

        Returns:
            Whether the given operation is executable on the device.
        """
        return protocols.num_qubits(op) < 2 or self.dist_on_device(*op.qubits) == 1

    def apply_swap(self, lq1: 'cirq.Qid', lq2: 'cirq.Qid') -> None:
        """Swaps two logical qubits in the map and in the inverse map.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.

        Raises:
            ValueError: whenever lq1 and lq2 are no adjacent on the device.
        """
        if self.dist_on_device(lq1, lq2) > 1:
            raise ValueError(
                f"q1: {lq1} and q2: {lq2} are not adjacent on the device. Cannot swap them."
            )

        pq1, pq2 = self._map[lq1], self._map[lq2]
        self._map[lq1], self._map[lq2] = self._map[lq2], self._map[lq1]

        self._inverse_map[pq1], self._inverse_map[pq2] = (
            self._inverse_map[pq2],
            self._inverse_map[pq1],
        )

    def mapped_op(self, op: 'cirq.Operation') -> 'cirq.Operation':
        """Transforms the given operation with the qubits in self._map.

        Args:
            op: an operation on logical qubits.

        Returns:
            The same operation on corresponding physical qubits."""
        return op.transform_qubits(self._map)

    def shortest_path(self, lq1: 'cirq.Qid', lq2: 'cirq.Qid') -> Sequence['cirq.Qid']:
        """Find that shortest path between two logical qubits on the device given their mapping."""
        return self._physical_shortest_path(self._map[lq1], self._map[lq2])

    @cached_method
    def _physical_shortest_path(self, pq1: 'cirq.Qid', pq2: 'cirq.Qid') -> Sequence['cirq.Qid']:
        return nx.shortest_path(self._induced_subgraph, pq1, pq2)
