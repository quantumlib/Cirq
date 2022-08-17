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

"""Concrete implementation of AbstractInitialMapper that places lines of qubits onto the device."""

from re import X
from typing import Dict, Optional, TYPE_CHECKING
import networkx as nx

from cirq import value, _compat
from cirq.transformers import routing

if TYPE_CHECKING:
    import cirq


@value.value_equality
class LineInitialMapper(routing.AbstractInitialMapper):
    """Places logical qubits in the circuit onto physical qubits on the device."""

    def __init__(self, device_graph: nx.Graph) -> None:
        """Initializes a LineInitialMapper.

        Args:
            device_graph: device graph
        """
        self.device_graph = device_graph

    def _make_circuit_graph(self, circuit: 'cirq.AbstractCircuit') -> nx.Graph:
        """Creates a (potentially incomplete) qubit connectivity graph of the circuit.

        Iterates over the moments circuit from left to right drawing edges between logical qubits
        that:
            (1) have degree < 2, and
            (2) that are involved in a 2-qubit operation in the current moment.
        At this point the graph is forest of paths and/or simple cycles. For each simple cycle, make
        it a path by removing the last edge that was added to it.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            The (potentially incomplete) qubit connectivity graph of the circuit.
        """
        circuit_graph = nx.Graph()
        edge_order = 0
        for op in circuit.all_operations():
            circuit_graph.add_nodes_from(op.qubits)
            if len(op.qubits) == 2 and all(
                circuit_graph.degree[op.qubits[i]] < 2 for i in range(2)
            ):
                circuit_graph.add_edge(*op.qubits, edge_order=edge_order)
                edge_order += 1
        found = True
        while found:
            try:
                cycle = nx.find_cycle(circuit_graph)
                edge_to_remove = max(
                    cycle, key=lambda x: circuit_graph.edges[x[0], x[1]]['edge_order']
                )
                circuit_graph.remove_edge(*edge_to_remove)
            except nx.exception.NetworkXNoCycle:
                found = False
        return circuit_graph

    def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Maps disjoint lines of logical qubits onto lines of physical qubits.

        Starting from the center physical qubit on the device, attempts to map disjoint lines of
        logical qubits given by the circuit graph onto one long line of physical qubits on the
        device, greedily maximizing each physical qubit's degree.
        If this mapping cannot be completed as one long line of qubits in the circuit graph mapped
        to qubits in the device graph, the line can be split as several line segments and then we:
            (i)   Map first line segment.
            (ii)  Find another high degree vertex in G near the center.
            (iii) Map the second line segment
            (iv)  etc.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            a dictionary that maps logical qubits in the circuit (keys) to physical qubits on the
            device (values).
        """
        return self._initial_mapping(circuit.freeze())

    @_compat.cached_method
    def _initial_mapping(self, circuit: 'cirq.FrozenCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Maps disjoint lines of logical qubits onto lines of physical qubits.

        Helper for 'initial_mapping' that takes a (hashable) frozen circuit to cache the result.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            a dictionary that maps logical qubits in the circuit (keys) to physical qubits on the
            device (values).
        """
        qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = {}
        circuit_graph = self._make_circuit_graph(circuit)
        physical_center = nx.center(self.device_graph)[0]

        def next_physical(current_physical: 'cirq.Qid') -> 'cirq.Qid':
            # use current physical if last logical line ended before mapping to it.
            if self.device_graph.nodes[current_physical]["mapped"] is False:
                return current_physical
            # else greedily map to highest degree neighbor that that is available
            sorted_neighbors = sorted(
                self.device_graph.neighbors(current_physical),
                key=lambda x: self.device_graph.degree(x),
            )
            for neighbor in reversed(sorted_neighbors):
                if self.device_graph.nodes[neighbor]["mapped"] is False:
                    return neighbor
            # if cannot map onto one long line of physical qubits, then break down into multiple
            # small lines by finding nearest available qubit to the physical center
            return self._closest_unmapped_qubit(physical_center)

        def next_logical(current_logical: 'cirq.Qid') -> Optional['cirq.Qid']:
            for neighbor in circuit_graph.neighbors(current_logical):
                if circuit_graph.nodes[neighbor]["mapped"] is False:
                    return neighbor
            return None

        for pq in self.device_graph.nodes:
            self.device_graph.nodes[pq]["mapped"] = False
        for lq in circuit_graph.nodes:
            circuit_graph.nodes[lq]["mapped"] = False

        current_physical = physical_center
        for logical_cc in nx.connected_components(circuit_graph):
            if len(logical_cc) == 1:
                continue
            # logical_cc is a set, make it a sorted list to guarantee deterministic behavior
            logical_cc = sorted(logical_cc)

            current_physical = next_physical(current_physical)
            # start by mapping a logical line from one of its endpoints.
            current_logical = next(q for q in logical_cc if circuit_graph.degree(q) == 1)

            while current_logical is not None:
                self.device_graph.nodes[current_physical]["mapped"] = True
                circuit_graph.nodes[current_logical]["mapped"] = True
                qubit_map[current_logical] = current_physical
                current_logical = next_logical(current_logical)
                if current_logical is not None:
                    current_physical = next_physical(current_physical)

        self._map_remaining_qubits(circuit, circuit_graph, qubit_map)
        return qubit_map

    def _map_remaining_qubits(
        self,
        circuit: 'cirq.AbstractCircuit',
        circuit_graph: nx.Graph,
        qubit_map: Dict['cirq.Qid', 'cirq.Qid'],
    ) -> None:
        """Maps remaining qubits that are not incident to edges in the circuit_graph.

        First maps logical qubits that interact in circuit but have missing edges in the circuit
        graph. Then maps logical qubits that don't interact with any other logical qubits in the
        circuit.

        Args:
            circuit_graph: the (potentially incomplete) qubit connectivity graph of the circuit.
            qubit_map: the mapping of logical to physical qubits done so far.
        """
        for op in circuit.all_operations():
            if len(op.qubits) == 2:
                q1, q2 = op.qubits
                if q1 not in qubit_map.keys():
                    physical = self._closest_unmapped_qubit(qubit_map[q2])
                    qubit_map[q1] = physical
                    self.device_graph.nodes[physical]["mapped"] = True
                # 'elif' because at least one must be mapped already
                elif q2 not in qubit_map.keys():
                    physical = self._closest_unmapped_qubit(qubit_map[q1])
                    qubit_map[q2] = physical
                    self.device_graph.nodes[physical]["mapped"] = True

        for isolated_qubit in (q for q in circuit_graph.nodes if q not in qubit_map):
            physical = self._closest_unmapped_qubit(qubit_map[next(iter(qubit_map))])
            qubit_map[isolated_qubit] = physical
            self.device_graph.nodes[physical]["mapped"] = True

    def _closest_unmapped_qubit(self, source: 'cirq.Qid') -> 'cirq.Qid':
        """Finds the closest available neighbor to a physical qubit 'source' on the device.

        Args:
            source: a physical qubit on the device.

        Returns:
            the closest available physical qubit to 'source'.

        Raises:
            ValueError: if there are no available qubits left on the device.
        """
        for _, successors in nx.bfs_successors(self.device_graph, source):
            for successor in successors:
                if self.device_graph.nodes[successor]["mapped"] is False:
                    return successor
        raise ValueError("No available physical qubits left on the device.")

    # Tanuj, should this be made a helper method somewhere since it is the same code being used
    # by MappingManager too (and likely subsequent initial mapping strategies)? Particularly,
    # since there is no good way of representating a graph using a hashable object, the
    # 'graph_equality' bit here can become standard or we can file an issue for it.
    def _value_equality_values_(self):
        """Two LineInitialMapper(s) are equal if they execute on the same device graph."""
        if nx.is_directed(self.device_graph):
            return (
                tuple(sorted(self.device_graph.nodes)),
                tuple(sorted(self.device_graph.edges)),
                True,
            )
        return (
            tuple(sorted(self.device_graph.nodes)),
            tuple(sorted(tuple(sorted(edge)) for edge in self.device_graph.edges)),
            False,
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        adj_dict = dict(self.device_graph.adjacency())
        graph_type = 'nx.DiGraph' if nx.is_directed(self.device_graph) else 'nx.Graph'
        return f'cirq.LineInitialMapper({graph_type}({adj_dict}))'
