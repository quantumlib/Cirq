# Copyright 2019 The Cirq Developers
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

from typing import Callable, Iterable, List, Optional

import networkx as nx

from cirq import circuits, ops
import cirq.contrib.acquaintance as cca
from cirq.contrib.routing.swap_network import SwapNetwork

BINARY_OP_PREDICATE = Callable[[ops.Operation, ops.Operation], bool]


def get_timeslices(dag: circuits.CircuitDag) -> List[nx.Graph]:
    circuit = circuits.Circuit.from_ops(
        op for op in dag.all_operations() if len(op.qubits) > 1)
    return [
        nx.Graph(op.qubits for op in moment.operations) for moment in circuit
    ]


def are_ops_consistent_with_device_graph(ops: Iterable[ops.Operation],
                                          device_graph: nx.Graph) -> bool:
    for op in ops:
        if not set(op.qubits).issubset(device_graph):
            return False
        if len(op.qubits) >= 2 and not device_graph.has_edge(*op.qubits):
            return False
    return True


def is_valid_routing(
        circuit: circuits.Circuit,
        swap_network: SwapNetwork,
        *,
        equals: Optional[BINARY_OP_PREDICATE] = None,
        can_reorder: BINARY_OP_PREDICATE = circuits.circuit_dag._disjoint_qubits
) -> bool:
    """Determines whether a swap network is consistent with a given circuit.

    Args:
        circuit: The circuit.
        swap_network: The swap network whose validity is to be checked.
        equals: The function to determine equality of operations. Defaults to
            `operator.eq`.
        can_reorder: A predicate that determines if two operations may be
            reordered.
    """
    circuit_dag = circuits.CircuitDag.from_circuit(circuit,
                                                   can_reorder=can_reorder)
    logical_operations = swap_network.get_logical_operations()
    return cca.is_topologically_sorted(circuit_dag, logical_operations, equals)
