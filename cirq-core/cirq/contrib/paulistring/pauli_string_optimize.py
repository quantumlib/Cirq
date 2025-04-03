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

import networkx

from cirq import circuits, linalg
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import pauli_string_dag_from_circuit
from cirq.contrib.paulistring.recombine import move_pauli_strings_into_circuit
from cirq.contrib.paulistring.separate import convert_and_separate_circuit
from cirq.ops import PauliStringGateOperation


def pauli_string_optimized_circuit(
    circuit: circuits.Circuit, move_cliffords: bool = True, atol: float = 1e-8
) -> circuits.Circuit:
    cl, cr = convert_and_separate_circuit(circuit, leave_cliffords=not move_cliffords, atol=atol)
    string_dag = pauli_string_dag_from_circuit(cl)

    # Merge and remove Pauli string phasors
    while True:
        before_len = len(string_dag.nodes())
        merge_equal_strings(string_dag)
        remove_negligible_strings(string_dag)
        if len(string_dag.nodes()) >= before_len:
            break

    c_all = move_pauli_strings_into_circuit(string_dag, cr)

    assert_no_multi_qubit_pauli_strings(c_all)

    return c_all


def assert_no_multi_qubit_pauli_strings(circuit: circuits.Circuit) -> None:
    for op in circuit.all_operations():
        if isinstance(op, PauliStringGateOperation):
            assert (
                len(op.pauli_string) == 1
            ), 'Multi qubit Pauli string left over'  # pragma: no cover


def merge_equal_strings(string_dag: circuitdag.CircuitDag) -> None:
    for node in tuple(string_dag.nodes()):
        if node not in string_dag.nodes():
            # Node was removed
            continue
        commuting_nodes = (
            set(string_dag.nodes())
            - set(networkx.dag.ancestors(string_dag, node))
            - set(networkx.dag.descendants(string_dag, node))
            - set([node])
        )
        for other_node in commuting_nodes:
            if node.val.pauli_string.equal_up_to_coefficient(other_node.val.pauli_string):
                string_dag.remove_node(other_node)
                node.val = node.val.merged_with(other_node.val)


def remove_negligible_strings(string_dag: circuitdag.CircuitDag, atol=1e-8) -> None:
    for node in tuple(string_dag.nodes()):
        if linalg.all_near_zero_mod(node.val.exponent_relative, 2, atol=atol):
            string_dag.remove_node(node)
