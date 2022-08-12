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

from cirq import circuits, routing

if TYPE_CHECKING:
    import cirq


class LineInitialMapper(routing.AbstractInitialMapper):
    """Maps disjoint lines of logical qubits onto lines of physical qubits."""

    def __init__(self, device_graph: nx.Graph, circuit: circuits.AbstractCircuit):
        """Initializes a LineInitialMapper.

        Args:
            device_graph: device graph
            circuit_graph: circuit graph
        """
        self.device_graph = device_graph
        self.circuit = circuit
        self.circuit_graph = self._make_circuit_graph()
        self._map = None

    def _make_circuit_graph(self) -> nx.Graph:
        """Creates a (potentially incomplete) qubit connectivity graph of the circuit.

        Iterates over the moments circuit from left to right drawing edges between logical qubits
        that:
            (1) have degree < 2, and
            (2) that are involved in a 2-qubit operation in the current moment.
        At this point the graph is forest of paths and/or simple cycles. For each simple cycle, make
        it a path by removing the last edge that was added to it.

        Returns:
            The qubit connectivity graph of the circuit.
        """
        circuit_graph = nx.Graph()
        edge_order, node_order = 0, 0
        for op in self.circuit.all_operations():
            circuit_graph.add_nodes_from(op.qubits, node_order=node_order)
            node_order += 1
            if len(op.qubits) == 2 and all(
                circuit_graph.degree[op.qubits[i]] < 2 for i in range(2)
            ):
                circuit_graph.add_edge(*op.qubits, edge_order=edge_order)
                edge_order += 1

        # make cycles into paths by removing last edge that was added
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