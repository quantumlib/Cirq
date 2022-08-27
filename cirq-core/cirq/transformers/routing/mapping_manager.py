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

"""Manages the mapping from logical to physical qubits during a routing procedure."""

from typing import Dict, Sequence, TYPE_CHECKING
import networkx as nx

from cirq import protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality
class MappingManager:
    """Class that manages the mapping from logical to physical qubits.

    Convenience methods over distance and mapping queries of the physical qubits are also provided.
    All such public methods of this class expect logical qubits.
    """

    def __init__(
        self, device_graph: nx.Graph, initial_mapping: Dict['cirq.Qid', 'cirq.Qid']
    ) -> None:
        """Initializes MappingManager.

        Sorts the nodes and edges in the device graph to guarantee graph equality. If undirected,
        also sorts the nodes within each edge.

        Args:
            device_graph: connectivity graph of qubits in the hardware device.
            initial_mapping: the initial mapping of logical (keys) to physical qubits (values).
        """
        # make sure edge insertion order is the same amongst equivalent graphs.
        if nx.is_directed(device_graph):
            self.device_graph = nx.DiGraph()
            self.device_graph.add_nodes_from(sorted(list(device_graph.nodes(data=True))))
            self.device_graph.add_edges_from(sorted(list(device_graph.edges)))
        else:
            self.device_graph = nx.Graph()
            self.device_graph.add_nodes_from(sorted(list(device_graph.nodes(data=True))))
            self.device_graph.add_edges_from(
                sorted(list(sorted(edge) for edge in device_graph.edges))
            )

        self._map = initial_mapping.copy()
        self._inverse_map = {v: k for k, v in self._map.items()}
        self._induced_subgraph = nx.induced_subgraph(self.device_graph, self._map.values())
        self._predecessors, self._distances = nx.floyd_warshall_predecessor_and_distance(
            self._induced_subgraph
        )

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
        """Finds distance between logical qubits 'lq1' and 'lq2' on the device.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.

        Returns:
            The shortest path distance.
        """
        return self._distances[self._map[lq1]][self._map[lq2]]

    def can_execute(self, op: 'cirq.Operation') -> bool:
        """Finds whether the given operation acts on qubits that are adjacent on the device.

        Args:
            op: an operation on logical qubits.

        Returns:
            True, if physical qubits corresponding to logical qubits `op.qubits` are adjacent on
            the device.
        """
        return protocols.num_qubits(op) < 2 or self.dist_on_device(*op.qubits) == 1

    def apply_swap(self, lq1: 'cirq.Qid', lq2: 'cirq.Qid') -> None:
        """Updates the mapping to simulate inserting a swap operation between `lq1` and `lq2`.

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
        """Find the shortest path between two logical qubits on the device given their mapping.

        Args:
            lq1: the first logical qubit.
            lq2: the second logical qubit.

        Returns:
            A sequence of logical qubits on the shortest path from lq1 to lq2.
        """
        return [
            self._inverse_map[pq]
            for pq in nx.reconstruct_path(self._map[lq1], self._map[lq2], self._predecessors)
        ]

    def _value_equality_values_(self):
        graph_equality = (
            tuple(self.device_graph.nodes),
            tuple(self.device_graph.edges),
            nx.is_directed(self.device_graph),
        )
        map_equality = tuple(sorted(self._map.items()))
        return (graph_equality, map_equality)

    def __repr__(self) -> str:
        graph_type = type(self.device_graph).__name__
        return (
            f'cirq.MappingManager('
            f'nx.{graph_type}({dict(self.device_graph.adjacency())}),'
            f' {self._map})'
        )
