# Copyright 2023 The Cirq Developers
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


from typing import Any
import cirq
import numpy as np


def assert_unitary_is_consistent(val: Any, ignoring_global_phase: bool = False):
    if not cirq.has_unitary(val):
        return
    if isinstance(val, cirq.Operation):
        qubits = val.qubits
        decomposition = cirq.decompose_once(val, default=None)
    else:
        qubits = tuple(cirq.LineQid.for_gate(val))
        decomposition = cirq.decompose_once_with_qubits(val, qubits, default=None)
    print(f'{decomposition=}')
    if decomposition is None or decomposition is NotImplemented:
        return

    # Ensure that `u` is a unitary.
    u = cirq.unitary(val)
    assert not (u is None or u is NotImplemented)
    assert cirq.is_unitary(u)

    c = cirq.Circuit(decomposition)
    if len(c.all_qubits().difference(qubits)) == 0:
        return

    clean_qubits = tuple(q for q in c.all_qubits() if isinstance(q, cirq.ops.CleanQubit))
    borrowable_qubits = tuple(q for q in c.all_qubits() if isinstance(q, cirq.ops.BorrowableQubit))
    qubit_order = clean_qubits + borrowable_qubits + qubits

    full_unitary = cirq.apply_unitaries(
        decomposition,
        qubits=qubit_order,
        args=cirq.ApplyUnitaryArgs.for_unitary(qid_shape=cirq.qid_shape(qubit_order)),
        default=None,
    )
    if full_unitary is None:
        raise ValueError(f'apply_unitaries failed on the decomposition of {val}')
    vol = np.prod(cirq.qid_shape(qubit_order), dtype=np.int64)
    full_unitary = full_unitary.reshape((vol, vol))

    vol = np.prod(cirq.qid_shape(borrowable_qubits + qubits), dtype=np.int64)

    # This matrix should be a unitary.
    # More over it has to have the shape I \otimes u.
    actual = full_unitary[:vol, :vol]
    expected = np.kron(np.eye(2 ** len(borrowable_qubits), dtype=np.complex128), u)

    if ignoring_global_phase:
        cirq.testing.assert_allclose_up_to_global_phase(actual, expected, atol=1e-8)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-8)
