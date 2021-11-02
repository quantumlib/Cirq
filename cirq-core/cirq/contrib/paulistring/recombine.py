# Copyright 2018 The Cirq Developers
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

from typing import Any, Callable, Iterable, Sequence, Tuple, Union, cast, List

from cirq import circuits, ops, protocols

from cirq.contrib.paulistring.pauli_string_dag import (
    pauli_string_reorder_pred,
    pauli_string_dag_from_circuit,
)


def _sorted_best_string_placements(
    possible_nodes: Iterable[Any],
    output_ops: Sequence[ops.Operation],
    key: Callable[[Any], ops.PauliStringPhasor] = lambda node: node.val,
) -> List[Tuple[ops.PauliStringPhasor, int, circuits.Unique[ops.PauliStringPhasor]]]:

    sort_key = lambda placement: (-len(placement[0].pauli_string), placement[1])

    node_maxes = []
    for possible_node in possible_nodes:
        string_op = key(possible_node)
        # Try moving the Pauli string through, stop at measurements
        node_max = (string_op, 0, possible_node)

        for i, out_op in enumerate(output_ops):
            if not set(out_op.qubits) & set(string_op.qubits):
                # Skip if operations don't share qubits
                continue
            if isinstance(out_op, ops.PauliStringPhasor) and protocols.commutes(
                out_op.pauli_string, string_op.pauli_string
            ):
                # Pass through another Pauli string if they commute
                continue
            if not (
                isinstance(out_op, ops.GateOperation)
                and isinstance(
                    out_op.gate,
                    (ops.SingleQubitCliffordGate, ops.PauliInteractionGate, ops.CZPowGate),
                )
            ):
                # This is as far through as this Pauli string can move
                break
            string_op = string_op.pass_operations_over([out_op], after_to_before=True)
            curr = (string_op, i + 1, possible_node)
            if sort_key(curr) > sort_key(node_max):
                node_max = curr

        node_maxes.append(node_max)

    return sorted(node_maxes, key=sort_key, reverse=True)


def move_pauli_strings_into_circuit(
    circuit_left: Union[circuits.Circuit, circuits.CircuitDag], circuit_right: circuits.Circuit
) -> circuits.Circuit:
    if isinstance(circuit_left, circuits.CircuitDag):
        string_dag = circuits.CircuitDag(pauli_string_reorder_pred, circuit_left)
    else:
        string_dag = pauli_string_dag_from_circuit(cast(circuits.Circuit, circuit_left))
    output_ops = list(circuit_right.all_operations())

    rightmost_nodes = set(string_dag.nodes()) - set(before for before, _ in string_dag.edges())

    while rightmost_nodes:
        # Sort the pauli string placements based on paulistring length and
        # furthest possible distance in circuit_right
        placements = _sorted_best_string_placements(rightmost_nodes, output_ops)
        last_index = len(output_ops)

        # Pick the Pauli string that can be moved furthest through
        # the Clifford circuit
        for best_string_op, best_index, best_node in placements:

            assert (
                best_index <= last_index
            ), "Unexpected insertion index order, {} >= {}, len: {}".format(
                best_index, last_index, len(output_ops)
            )

            last_index = best_index
            output_ops.insert(best_index, best_string_op)
            # Remove the best one from the dag and update rightmost_nodes
            rightmost_nodes.remove(best_node)
            rightmost_nodes.update(
                pred_node
                for pred_node in string_dag.predecessors(best_node)
                if len(string_dag.succ[pred_node]) <= 1
            )
            string_dag.remove_node(best_node)

    assert not string_dag.nodes(), 'There was a cycle in the CircuitDag'

    return circuits.Circuit(
        output_ops, strategy=circuits.InsertStrategy.EARLIEST, device=circuit_right.device
    )
