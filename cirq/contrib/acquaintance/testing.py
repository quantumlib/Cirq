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

from typing import cast, Sequence, TYPE_CHECKING

from cirq import devices, ops, protocols
from cirq.contrib.acquaintance.permutation import (
    PermutationGate, update_mapping)

if TYPE_CHECKING:
    import cirq


def assert_permutation_decomposition_equivalence(
        gate: PermutationGate,
        n_qubits: int) -> None:
    qubits = devices.LineQubit.range(n_qubits)
    operations = protocols.decompose_once_with_qubits(gate, qubits)
    operations = list(
        cast(Sequence['cirq.Operation'], ops.flatten_op_tree(operations)))
    mapping = {cast(ops.Qid, q): i for i, q in enumerate(qubits)}
    update_mapping(mapping, operations)
    expected_mapping = {qubits[j]: i
            for i, j in gate.permutation().items()}
    assert mapping == expected_mapping, (
        "{!r}.permutation({}) doesn't match decomposition.\n"
        '\n'
        'Actual mapping:\n'
        '{}\n'
        '\n'
        'Expected mapping:\n'
        '{}\n'.format(gate, n_qubits,
            [mapping[q] for q in qubits],
            [expected_mapping[q] for q in qubits])
    )
