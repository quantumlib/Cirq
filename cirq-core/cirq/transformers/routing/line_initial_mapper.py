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

from typing import Dict, List, Set, TYPE_CHECKING
import networkx as nx

from cirq.transformers.routing import AbstractInitialMapper
from cirq import protocols

if TYPE_CHECKING:
    import cirq


class LineInitialMapper(AbstractInitialMapper):
    """Places logical qubits in the circuit onto physical qubits on the device."""

    def __init__(self, device_graph: nx.Graph) -> None:
        """Initializes a LineInitialMapper.

        Args:
            device_graph: device graph
        """
        self.device_graph = device_graph
        self.mapped_physicals: Set['cirq.Qid'] = set()
        self.partners: Dict['cirq.Qid', 'cirq.Qid'] = {}

    def _make_circuit_graph(self, circuit: 'cirq.AbstractCircuit') -> List[List['cirq.Qid']]:
        """Creates a (potentially incomplete) qubit connectivity graph of the circuit.

        Iterates over moments in the circuit from left to right and adds edges between logical
        qubits if the logical qubit pair l1 and l2
            (1) have degree < 2,
            (2) are involved in a 2-qubit operation in the current moment, and
            (3) adding such an edge will not produce a cycle in the graph.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            The (potentially incomplete) qubit connectivity graph of the circuit, which is
                guaranteed to be a forest of line graphs.
        """
        circuit_graph: List[List['cirq.Qid']] = [[q] for q in sorted(circuit.all_qubits())]
        component_id: Dict['cirq.Qid', int] = {q[0]: i for i, q in enumerate(circuit_graph)}

        def degree_lt_two(q: 'cirq.Qid'):
            return any(circuit_graph[component_id[q]][i] == q for i in [-1, 0])

        for op in circuit.all_operations():
            if protocols.num_qubits(op) != 2:
                continue

            q0, q1 = op.qubits
            c0, c1 = component_id[q0], component_id[q1]

            # Keep track of partners for mapping isolated qubits later.
            if q0 not in self.partners:
                self.partners[q0] = q1
            if q1 not in self.partners:
                self.partners[q1] = q0

            if not (degree_lt_two(q0) and degree_lt_two(q1) and c0 != c1):
                continue

            # Make sure c0/q0 are for the largest component.
            if len(circuit_graph[c0]) < len(circuit_graph[c1]):
                c0, c1, q0, q1 = c1, c0, q1, q0

            # copy smaller component into larger one.
            if circuit_graph[c0][0] == q0:
                if circuit_graph[c1][0] == q1:
                    for q in circuit_graph[c1]:
                        circuit_graph[c0].insert(0, q)
                        component_id[q] = c0
                else:
                    for q in reversed(circuit_graph[c1]):
                        circuit_graph[c0].insert(0, q)
                        component_id[q] = c0
            else:
                if circuit_graph[c1][0] == q1:
                    for q in circuit_graph[c1]:
                        circuit_graph[c0].append(q)
                        component_id[q] = c0
                else:
                    for q in reversed(circuit_graph[c1]):
                        circuit_graph[c0].append(q)
                        component_id[q] = c0

        return sorted([circuit_graph[c] for c in set(component_id.values())], key=len, reverse=True)

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
        A line is split by mapping the next logical qubit to the nearest available physical qubit
        to the center of the device graph.

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
            # Greedily map to highest degree neighbor that that is available
            sorted_neighbors = sorted(
                self.device_graph.neighbors(current_physical),
                key=lambda x: self.device_graph.degree(x),
                reverse=True,
            )
            for neighbor in sorted_neighbors:
                if neighbor not in self.mapped_physicals:
                    return neighbor
            # If cannot map onto one long line of physical qubits, then break down into multiple
            # small lines by finding nearest available qubit to the physical center
            return self._closest_unmapped_qubit(physical_center)

        pq = physical_center
        first_isolated_idx = len(circuit_graph)
        for idx, logical_line in enumerate(circuit_graph):
            if len(logical_line) == 1:
                first_isolated_idx = idx
                break

            for lq in logical_line:
                self.mapped_physicals.add(pq)
                qubit_map[lq] = pq
                # Edge case: if mapping n qubits on an n-qubit device should not call next_physical
                # when finished mapping the last logical qubit else will raise an error
                if len(circuit.all_qubits()) != len(self.mapped_physicals):
                    pq = next_physical(pq)

        for i in range(first_isolated_idx, len(circuit_graph)):
            lq = circuit_graph[i][0]
            partner = qubit_map[self.partners[lq]] if lq in self.partners else physical_center
            pq = self._closest_unmapped_qubit(partner)
            self.mapped_physicals.add(pq)
            qubit_map[lq] = pq

        return qubit_map

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
                if successor not in self.mapped_physicals:
                    return successor
        raise ValueError("No available physical qubits left on the device.")

    def __eq__(self, other) -> bool:
        return nx.utils.graphs_equal(self.device_graph, other.device_graph)

    def __repr__(self):
        graph_type = type(self.device_graph).__name__
        return f'cirq.LineInitialMapper(nx.{graph_type}({dict(self.device_graph.adjacency())}))'
