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
    assert rx == [
        cirq.PhasedXPowGate(phase_exponent=1).on(cirq.GridQubit(0, 0))
    ]
    assert rop == [
        cirq.ry(np.pi / 2).on(op.qubits[0]),
        cirq.ms(np.pi / 4).on(op.qubits[0], op.qubits[1]),
        cirq.rx(-1 * np.pi / 2).on(op.qubits[0]),
        cirq.rx(-1 * np.pi / 2).on(op.qubits[1]),
        cirq.ry(-1 * np.pi / 2).on(op.qubits[0])
    ]
    assert rcnot == [
        cirq.PhasedXPowGate(phase_exponent=-0.75,
                            exponent=0.5).on(cirq.GridQubit(0, 0)),
        cirq.PhasedXPowGate(phase_exponent=1,
                            exponent=0.25).on(cirq.GridQubit(0, 1)),
        cirq.T.on(cirq.GridQubit(0, 0)),
        cirq.ms(-0.5 * np.pi / 2).on(cirq.GridQubit(0, 0), cirq.GridQubit(0,
                                                                          1)),
        (cirq.Y**0.5).on(cirq.GridQubit(0, 0)),
        cirq.PhasedXPowGate(phase_exponent=1,
                            exponent=0.25).on(cirq.GridQubit(0, 1)),
        (cirq.Z**-0.75).on(cirq.GridQubit(0, 0))
    ]


def test_convert_to_ion_circuit():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    us = cirq.Duration(nanos=1000)
    ion_device = cirq.IonDevice(us, us, us, [q0, q1])

    clifford_circuit_1 = cirq.Circuit()
    clifford_circuit_1.append(
        [cirq.X(q0), cirq.H(q1),
         cirq.ms(np.pi / 4).on(q0, q1)])
    ion_circuit_1 = cirq.ion.ConvertToIonGates().convert_circuit(
        clifford_circuit_1)

    ion_device.validate_circuit(ion_circuit_1)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        clifford_circuit_1, ion_circuit_1, atol=1e-6)
    clifford_circuit_2 = cirq.Circuit()
    clifford_circuit_2.append(
        [cirq.X(q0),
         cirq.CNOT(q1, q0),
         cirq.ms(np.pi / 4).on(q0, q1)])
    ion_circuit_2 = cirq.ion.ConvertToIonGates().convert_circuit(
        clifford_circuit_2)
    ion_device.validate_circuit(ion_circuit_2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        clifford_circuit_2, ion_circuit_2, atol=1e-6)
