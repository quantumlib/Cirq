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

import cirq

import pytest
import numpy as np


class InconsistentGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def _decompose_with_context_(self, qubits, *, context):
        (q,) = context.qubit_manager.qalloc(1)
        yield cirq.X(q)
        yield cirq.CNOT(q, qubits[0])


class FailsOnDecompostion(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def _has_unitary_(self) -> bool:
        return True

    def _decompose_with_context_(self, qubits, *, context):
        (q,) = context.qubit_manager.qalloc(1)
        yield cirq.X(q)
        yield cirq.measure(qubits[0])


class CleanCorrectButBorrowableIncorrectGate(cirq.Gate):
    """Ancilla type determines if the decomposition is correct or not."""

    def __init__(self, use_clean_ancilla: bool) -> None:
        self.ancillas_are_clean = use_clean_ancilla

    def _num_qubits_(self):
        return 2

    def _decompose_with_context_(self, qubits, *, context):
        if self.ancillas_are_clean:
            anc = context.qubit_manager.qalloc(1)
        else:
            anc = context.qubit_manager.qborrow(1)
        yield cirq.CCNOT(*qubits, *anc)
        yield cirq.Z(*anc)
        yield cirq.CCNOT(*qubits, *anc)
        context.qubit_manager.qfree(anc)


@pytest.mark.parametrize('ignore_phase', [False, True])
@pytest.mark.parametrize(
    'g,is_consistent',
    [
        (cirq.testing.PhaseUsingCleanAncilla(theta=0.1, ancilla_bitsize=3), True),
        (cirq.testing.PhaseUsingDirtyAncilla(phase_state=1, ancilla_bitsize=4), True),
        (InconsistentGate(), False),
        (CleanCorrectButBorrowableIncorrectGate(use_clean_ancilla=True), True),
        (CleanCorrectButBorrowableIncorrectGate(use_clean_ancilla=False), False),
    ],
)
def test_assert_unitary_is_consistent(g, ignore_phase, is_consistent):
    if is_consistent:
        cirq.testing.assert_unitary_is_consistent(g, ignore_phase)
        cirq.testing.assert_unitary_is_consistent(g.on(*cirq.LineQid.for_gate(g)), ignore_phase)
    else:
        with pytest.raises(AssertionError):
            cirq.testing.assert_unitary_is_consistent(g, ignore_phase)
        with pytest.raises(AssertionError):
            cirq.testing.assert_unitary_is_consistent(g.on(*cirq.LineQid.for_gate(g)), ignore_phase)


def test_failed_decomposition():
    with pytest.raises(ValueError):
        cirq.testing.assert_unitary_is_consistent(FailsOnDecompostion())

    _ = cirq.testing.assert_unitary_is_consistent(cirq.Circuit())
