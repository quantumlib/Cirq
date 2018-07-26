# Copyright 2018 The ops Developers
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

from typing import Union, cast

from cirq import ops, circuits

from cirq.contrib.paulistring.pauli_string_phasor import PauliStringPhasor
from cirq.contrib.paulistring.pauli_string_dag import (
    pauli_string_dag_from_circuit)


def move_non_clifford_into_clifford(circuit_left: Union[circuits.Circuit,
                                                        circuits.CircuitDag],
                                    circuit_right: circuits.Circuit
                                    ) -> circuits.Circuit:
    if isinstance(circuit_left, circuits.CircuitDag):
        string_dag = cast(circuits.CircuitDag, circuit_left)
    else:
        string_dag = pauli_string_dag_from_circuit(
                        cast(circuits.Circuit, circuit_left))
    output_ops = list(circuit_right.all_operations())

    rightmost_nodes = (set(string_dag.nodes)
                       - set(before for before, _ in string_dag.edges))

    while rightmost_nodes:
        # Pick the Pauli string that can be moved furthest through the Clifford
        # circuit
        shortest_string_len = float('inf')
        best_index = 0
        best_string_op = None
        best_node = None

        for right_node in rightmost_nodes:
            string_op = right_node.val
            # Try moving the Pauli string through, stop at measurements
            if len(string_op.pauli_string) < shortest_string_len:
                shortest_string_len = len(string_op.pauli_string)
                best_index = 0
                best_string_op = string_op
                best_node = right_node
            for i, out_op in enumerate(output_ops):
                if not set(out_op.qubits) & set(string_op.qubits):
                    # Skip if operations don't share qubits
                    continue
                if (isinstance(out_op, PauliStringPhasor) or
                    not (isinstance(out_op, ops.GateOperation) and
                         isinstance(out_op.gate, (ops.CliffordGate,
                                                  ops.PauliInteractionGate)))):
                    # This is as far through as this Pauli string can move
                    break
                string_op = string_op.pass_operations_over([out_op],
                                                           after_to_before=True)
                if (len(string_op.pauli_string) < shortest_string_len
                    or (len(string_op.pauli_string) == shortest_string_len
                        and i+1 > best_index)):
                    shortest_string_len = len(string_op.pauli_string)
                    best_index = i + 1
                    best_string_op = string_op
                    best_node = right_node

        assert best_node is not None

        # Place the best one into the output circuit
        output_ops.insert(best_index, cast(ops.Operation, best_string_op))
        # Remove the best one from the dag and update rightmost_nodes
        rightmost_nodes.remove(best_node)
        rightmost_nodes.update(
            pred_node
            for pred_node in string_dag.predecessors(best_node)
            if len(string_dag.succ[pred_node]) <= 1)
        string_dag.remove_node(best_node)

    assert not string_dag.nodes

    return circuits.Circuit.from_ops(
            output_ops,
            strategy=circuits.InsertStrategy.EARLIEST,
            device=circuit_right.device)
