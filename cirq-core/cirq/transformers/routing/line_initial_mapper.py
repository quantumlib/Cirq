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

"""Maps logical to physical qubits by greedily placing lines of logical qubits on the device.

This is the default placement strategy used in the CQC router.

It first creates a partial connectivity graph between logical qubits in the given circuit and then
maps these logical qubits on physical qubits on the device by starting at the center of the device
and greedily choosing the highest degree neighbor.

If some logical qubits are unampped after this first procedure then there are two cases:
    (1) These unmammep logical qubits do interact in the circuit with some other logical partner.
    In this case we map such a qubit to the nearest available physical qubit on the device to the
    one that its partner was mapped to.

    (2) These unampped logical qubits only have single qubit operations on them (i.e they do not
    interact with any other logical qubit at any point in the circuit). In this case we map them to
    the nearest available neighbor to the center of the device.
"""

from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx

from cirq.transformers.routing import initial_mapper
from cirq import protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality
class LineInitialMapper(initial_mapper.AbstractInitialMapper):
    """Places logical qubits in the circuit onto physical qubits on the device.

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

    The expected runtime of this strategy is O(m logn + n^2) where m is the # of operations in the
    given circuit and n is the number of qubits. The first term corresponds to the runtime of
    'make_circuit_graph()' and the second for 'initial_mapping()'.
    """

    def __init__(self, device_graph: nx.Graph) -> None:
        """Initializes a LineInitialMapper.

        Args:
            device_graph: device graph
        """
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
        self.center = nx.center(self.device_graph)[0]

    def _make_circuit_graph(
        self, circuit: 'cirq.AbstractCircuit'
    ) -> Tuple[List[Deque['cirq.Qid']], Dict['cirq.Qid', 'cirq.Qid']]:
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
        circuit_graph: List[Deque['cirq.Qid']] = [deque([q]) for q in sorted(circuit.all_qubits())]
        component_id: Dict['cirq.Qid', int] = {q[0]: i for i, q in enumerate(circuit_graph)}
        partners: Dict['cirq.Qid', 'cirq.Qid'] = {}

        def degree_lt_two(q: 'cirq.Qid'):
            return any(circuit_graph[component_id[q]][i] == q for i in [-1, 0])

        for op in circuit.all_operations():
            if protocols.num_qubits(op) != 2:
                continue

            q0, q1 = op.qubits
            c0, c1 = component_id[q0], component_id[q1]
            # Keep track of partners for mapping isolated qubits later.
            partners[q0] = partners[q0] if q0 in partners else q1
            partners[q1] = partners[q1] if q1 in partners else q0

            if not (degree_lt_two(q0) and degree_lt_two(q1) and c0 != c1):
                continue

            # Make sure c0/q0 are for the largest component.
            if len(circuit_graph[c0]) < len(circuit_graph[c1]):
                c0, c1, q0, q1 = c1, c0, q1, q0

            # copy smaller component into larger one.
            c1_order = (
                reversed(circuit_graph[c1])
                if circuit_graph[c1][-1] == q1
                else iter(circuit_graph[c1])
            )
            for q in c1_order:
                if circuit_graph[c0][0] == q0:
                    circuit_graph[c0].appendleft(q)
                else:
                    circuit_graph[c0].append(q)
                component_id[q] = c0

        graph = sorted(
            [circuit_graph[c] for c in set(component_id.values())], key=len, reverse=True
        )
        return graph, partners

    def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Maps disjoint lines of logical qubits onto lines of physical qubits.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            a dictionary that maps logical qubits in the circuit (keys) to physical qubits on the
            device (values).
        """
        mapped_physicals: Set['cirq.Qid'] = set()
        qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = {}
        circuit_graph, partners = self._make_circuit_graph(circuit)

        def next_physical(
            current_physical: 'cirq.Qid', partner: 'cirq.Qid', isolated: bool = False
        ) -> 'cirq.Qid':
            # Handle the first physical qubit getting mapped.
            if current_physical not in mapped_physicals:
                return current_physical
            # Greedily map to highest degree neighbor that is available
            if not isolated:
                sorted_neighbors = sorted(
                    self.device_graph.neighbors(current_physical),
                    key=lambda x: self.device_graph.degree(x),
                    reverse=True,
                )
                for neighbor in sorted_neighbors:
                    if neighbor not in mapped_physicals:
                        return neighbor
                # If cannot map onto one long line of physical qubits, then break down into multiple
                # small lines by finding nearest available qubit to the physical center
            return self._closest_unmapped_qubit(partner, mapped_physicals)

        pq = self.center
        for logical_line in circuit_graph:
            for lq in logical_line:
                is_isolated = len(logical_line) == 1
                partner = (
                    qubit_map[partners[lq]] if (lq in partners and is_isolated) else self.center
                )
                pq = next_physical(pq, partner, isolated=is_isolated)
                mapped_physicals.add(pq)
                qubit_map[lq] = pq

        return qubit_map

    def _closest_unmapped_qubit(
        self, source: 'cirq.Qid', mapped_physicals: Set['cirq.Qid']
    ) -> 'cirq.Qid':
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
                if successor not in mapped_physicals:
                    return successor
        raise ValueError("No available physical qubits left on the device.")

    def _value_equality_values_(self):
        return (
            tuple(self.device_graph.nodes),
            tuple(self.device_graph.edges),
            nx.is_directed(self.device_graph),
        )

    def __repr__(self):
        graph_type = type(self.device_graph).__name__
        return f'cirq.LineInitialMapper(nx.{graph_type}({dict(self.device_graph.adjacency())}))'
