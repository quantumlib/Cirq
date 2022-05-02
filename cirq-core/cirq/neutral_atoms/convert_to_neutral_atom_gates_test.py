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

import numpy as np
import pytest

import cirq


Q = cirq.LineQubit.range(3)


@pytest.mark.parametrize(
    'expected',
    (
        cirq.Circuit(cirq.X.on(Q[0])),
        cirq.Circuit(cirq.Y.on(Q[0])),
        cirq.Circuit(cirq.ParallelGate(cirq.X, 3).on(*Q)),
        cirq.Circuit(cirq.CNOT.on(Q[0], Q[1])),
    ),
)
def test_gates_preserved(expected: cirq.Circuit):
    actual = cirq.optimize_for_target_gateset(
        expected, gateset=cirq.neutral_atoms.NeutralAtomGateset()
    )
    assert actual == expected


def test_coverage():
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v0.16', count=5
    ):
        q = cirq.LineQubit.range(3)
        g = cirq.testing.ThreeQubitGate()

        class FakeOperation(cirq.Operation):
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
        assert not cirq.neutral_atoms.is_native_neutral_atom_gate(g)


def test_avoids_decompose_fallback_when_matrix_available_single_qubit():
    class OtherX(cirq.testing.SingleQubitGate):
        def _unitary_(self) -> np.ndarray:
            return np.array([[0, 1], [1, 0]])

    class OtherOtherX(cirq.testing.SingleQubitGate):
        def _decompose_(self, qubits):
            return OtherX().on(*qubits)

    q = cirq.GridQubit(0, 0)
    c = cirq.Circuit(OtherX().on(q), OtherOtherX().on(q))
    converted = cirq.optimize_for_target_gateset(c, gateset=cirq.neutral_atoms.NeutralAtomGateset())
    cirq.testing.assert_has_diagram(converted, '(0, 0): ───PhX(1)───PhX(1)───')
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v0.16', count=2
    ):
        cirq.neutral_atoms.ConvertToNeutralAtomGates().optimize_circuit(c)
        cirq.testing.assert_has_diagram(c, '(0, 0): ───PhX(1)───PhX(1)───')


def test_avoids_decompose_fallback_when_matrix_available_two_qubit():
    class OtherCZ(cirq.testing.TwoQubitGate):
        def _unitary_(self) -> np.ndarray:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    class OtherOtherCZ(cirq.testing.TwoQubitGate):
        def _decompose_(self, qubits):
            return OtherCZ().on(*qubits)

    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    c = cirq.Circuit(OtherCZ().on(q00, q01), OtherOtherCZ().on(q00, q01))
    expected_diagram = """
(0, 0): ───@───@───
           │   │
(0, 1): ───@───@───
"""
    converted = cirq.optimize_for_target_gateset(c, gateset=cirq.neutral_atoms.NeutralAtomGateset())
    cirq.testing.assert_has_diagram(converted, expected_diagram)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v0.16', count=2
    ):
        cirq.neutral_atoms.ConvertToNeutralAtomGates().optimize_circuit(c)
        cirq.testing.assert_has_diagram(c, expected_diagram)
