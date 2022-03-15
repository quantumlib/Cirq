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

from typing import Callable

from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.pauli_string_optimize import pauli_string_optimized_circuit
from cirq.contrib.paulistring.clifford_optimize import clifford_optimized_circuit


class _CZTargetGateSet(transformers.CZTargetGateset):
    """Private implementation of `cirq.CZTargetGateset` used for optimized_circuit method below.

    The implementation extends `cirq.CZTargetGateset` by modifying decomposed operations using
    `post_clean_up` before putting them back in the circuit.
    """

    def __init__(
        self,
        post_clean_up: Callable[[ops.OP_TREE], ops.OP_TREE] = lambda op_tree: op_tree,
    ):
        super().__init__()
        self.post_clean_up = post_clean_up

    def _decompose_two_qubit_operation(self, op: ops.Operation, _) -> ops.OP_TREE:
        ret = super()._decompose_two_qubit_operation(op, _)
        return ret if ret is NotImplemented else self.post_clean_up(ret)


def optimized_circuit(
    circuit: circuits.Circuit, atol: float = 1e-8, repeat: int = 10, merge_interactions: bool = True
) -> circuits.Circuit:
    circuit = circuits.Circuit(circuit)  # Make a copy
    gateset = _CZTargetGateSet(post_clean_up=_optimized_ops)
    for _ in range(repeat):
        start_len = len(circuit)
        start_cz_count = _cz_count(circuit)
        if merge_interactions:
            circuit = transformers.optimize_for_target_gateset(circuit, gateset=gateset)
        circuit2 = pauli_string_optimized_circuit(circuit, move_cliffords=False, atol=atol)
        circuit3 = clifford_optimized_circuit(circuit2, atol=atol)
        if len(circuit3) == start_len and _cz_count(circuit3) == start_cz_count:
            return circuit3
        circuit = circuit3
    return circuit


def _optimized_ops(ops: ops.OP_TREE, atol: float = 1e-8, repeat: int = 10) -> ops.OP_TREE:
    c = circuits.Circuit(ops)
    c_opt = optimized_circuit(c, atol, repeat, merge_interactions=False)
    return [*c_opt.all_operations()]


def _cz_count(circuit):
    return sum(isinstance(op.gate, ops.CZPowGate) for moment in circuit for op in moment)
