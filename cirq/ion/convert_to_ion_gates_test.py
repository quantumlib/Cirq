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


class OtherX(cirq.SingleQubitGate):
    def _unitary_(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])


class NoUnitary(cirq.SingleQubitGate):
    pass


class OtherCNOT(cirq.TwoQubitGate):
    def _unitary_(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])


def test_convert_to_ion_gates():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    op = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit()

    with pytest.raises(TypeError):
        cirq.ion.ConvertToIonGates().convert_one(circuit)

    with pytest.raises(TypeError):
        cirq.ion.ConvertToIonGates().convert_one(NoUnitary().on(q0))

    no_unitary_op = NoUnitary().on(q0)
    assert cirq.ion.ConvertToIonGates(ignore_failures=True).convert_one(
        no_unitary_op) == [no_unitary_op]

    rx = cirq.ion.ConvertToIonGates().convert_one(OtherX().on(q0))
    rop = cirq.ion.ConvertToIonGates().convert_one(op)
    rcnot = cirq.ion.ConvertToIonGates().convert_one(OtherCNOT().on(q0, q1))
    assert rx == [(cirq.X**1.0).on(cirq.GridQubit(0, 0))]
    assert rop == [cirq.Ry(np.pi/2).on(op.qubits[0]),
                   cirq.ion.MS(np.pi/4).on(op.qubits[0], op.qubits[1]),
                   cirq.ops.Rx(-1*np.pi/2).on(op.qubits[0]),
                   cirq.ops.Rx(-1*np.pi/2).on(op.qubits[1]),
                   cirq.ops.Ry(-1*np.pi/2).on(op.qubits[0])]
    assert rcnot == [cirq.PhasedXPowGate(phase_exponent=-0.75, exponent=0.5).
                         on(cirq.GridQubit(0, 0)),
                     (cirq.X**-0.25).on(cirq.GridQubit(0, 1)),
                     cirq.T.on(cirq.GridQubit(0, 0)),
                     cirq.MS(-0.5*np.pi/2).on(cirq.GridQubit(0, 0),
                                              cirq.GridQubit(0, 1)),
                     (cirq.Y**0.5).on(cirq.GridQubit(0, 0)),
                     (cirq.X**-0.25).on(cirq.GridQubit(0, 1)),
                     (cirq.Z**-0.75).on(cirq.GridQubit(0, 0))]


def _operations_to_matrix(operations, qubits):
    return cirq.Circuit.from_ops(operations).to_unitary_matrix(
        qubit_order=cirq.QubitOrder.explicit(qubits),
        qubits_that_should_be_present=qubits)


def assert_ops_implement_unitary(q0, q1, operations, intended_effect,
                                 atol=0.01):
    actual_effect = _operations_to_matrix(operations, (q0, q1))
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect,
                                            atol=atol)


def test_convert_to_ion_circuit():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    clifford_circuit_1 = cirq.Circuit()
    clifford_circuit_1.append([cirq.X(q0), cirq.H(q1),
                               cirq.MS(np.pi/4).on(q0, q1)])
    ion_circuit_1 = cirq.ion.ConvertToIonGates().\
        convert_circuit(clifford_circuit_1)
    clifford_circuit_2 = cirq.Circuit()
    clifford_circuit_2.append([cirq.X(q0), cirq.CNOT(q1, q0), cirq.MS(
        np.pi/4).on(q0, q1)])
    ion_circuit_2 = cirq.ion.ConvertToIonGates().\
        convert_circuit(clifford_circuit_2)

    cirq.testing.assert_has_diagram(ion_circuit_1,
                                    """
(0, 0): ───X────────────MS(0.25π)───
                        │
(0, 1): ───Y^-0.5───Z───MS(0.25π)───
    """,
                                    use_unicode_characters=True)

    cirq.testing.assert_has_diagram(ion_circuit_2,
                                    """
(0, 0): ───X───────MS(0.25π)───X^-0.5─────────────────────MS(0.25π)───
                   │                                      │
(0, 1): ───Y^0.5───MS(0.25π)───PhasedX(-0.5)^0.5───S^-1───MS(0.25π)───
        """,
                                    use_unicode_characters=True)

    assert_ops_implement_unitary(q0, q1, ion_circuit_1,
                                 cirq.unitary(clifford_circuit_1))
    assert_ops_implement_unitary(q0, q1, ion_circuit_2,
                                 cirq.unitary(clifford_circuit_2))
