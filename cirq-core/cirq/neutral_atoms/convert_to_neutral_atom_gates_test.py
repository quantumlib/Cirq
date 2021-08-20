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

import pytest
import numpy as np

import cirq
from cirq import ops


def test_coverage():
    q = cirq.LineQubit.range(3)
    g = cirq.ThreeQubitGate()

    class FakeOperation(ops.Operation):
        def __init__(self, gate, qubits):
            self._gate = gate
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            return FakeOperation(self._gate, new_qubits)

    op = FakeOperation(g, q).with_qubits(*q)
    circuit_ops = [cirq.Y(q[0]), cirq.ParallelGate(cirq.X, 3).on(*q)]
    c = cirq.Circuit(circuit_ops)
    cirq.neutral_atoms.ConvertToNeutralAtomGates().optimize_circuit(c)
    assert c == cirq.Circuit(circuit_ops)
    assert cirq.neutral_atoms.ConvertToNeutralAtomGates().convert(cirq.X.on(q[0])) == [
        cirq.X.on(q[0])
    ]
    with pytest.raises(TypeError, match="Don't know how to work with"):
        cirq.neutral_atoms.ConvertToNeutralAtomGates().convert(op)
    assert not cirq.neutral_atoms.is_native_neutral_atom_op(op)


def test_avoids_decompose_fallback_when_matrix_available_single_qubit():
    class OtherX(cirq.SingleQubitGate):
        def _unitary_(self) -> np.ndarray:
            return np.array([[0, 1], [1, 0]])

    class OtherOtherX(cirq.SingleQubitGate):
        def _decompose_(self, qubits):
            return OtherX().on(*qubits)

    q = cirq.GridQubit(0, 0)
    c = cirq.Circuit(OtherX().on(q), OtherOtherX().on(q))
    cirq.neutral_atoms.ConvertToNeutralAtomGates().optimize_circuit(c)
    cirq.testing.assert_has_diagram(c, '(0, 0): ───PhX(1)───PhX(1)───')


def test_avoids_decompose_fallback_when_matrix_available_two_qubit():
    class OtherCZ(cirq.TwoQubitGate):
        def _unitary_(self) -> np.ndarray:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    class OtherOtherCZ(cirq.TwoQubitGate):
        def _decompose_(self, qubits):
            return OtherCZ().on(*qubits)

    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    c = cirq.Circuit(OtherCZ().on(q00, q01), OtherOtherCZ().on(q00, q01))
    cirq.neutral_atoms.ConvertToNeutralAtomGates().optimize_circuit(c)
    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0): ───@───@───
           │   │
(0, 1): ───@───@───
""",
    )
