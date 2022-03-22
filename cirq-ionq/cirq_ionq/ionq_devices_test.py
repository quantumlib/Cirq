# Copyright 2021 The Cirq Developers
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

import cirq
import cirq_ionq as ionq


VALID_GATES = (
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.X**0.5,
    cirq.Y**0.5,
    cirq.Z**0.5,
    cirq.rx(0.1),
    cirq.ry(0.1),
    cirq.rz(0.1),
    cirq.H,
    cirq.HPowGate(exponent=1, global_shift=-0.5),
    cirq.T,
    cirq.S,
    cirq.CNOT,
    cirq.CXPowGate(exponent=1, global_shift=-0.5),
    cirq.XX,
    cirq.YY,
    cirq.ZZ,
    cirq.XX**0.5,
    cirq.YY**0.5,
    cirq.ZZ**0.5,
    cirq.SWAP,
    cirq.SwapPowGate(exponent=1, global_shift=-0.5),
    cirq.MeasurementGate(num_qubits=1, key='a'),
    cirq.MeasurementGate(num_qubits=2, key='b'),
    cirq.MeasurementGate(num_qubits=10, key='c'),
)


INVALID_GATES = (cirq.CNOT**0.5, cirq.SWAP**0.5, cirq.CCX, cirq.CCZ, cirq.CZ)


@pytest.mark.parametrize('gate', VALID_GATES)
def test_decompose_leaves_supported_alone(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    operation = gate(*qubits)
    assert ionq.decompose_to_device(operation) == operation


VALID_DECOMPOSED_GATES = cirq.Gateset(cirq.XPowGate, cirq.ZPowGate, cirq.CNOT, cirq.MeasurementGate)


def test_decompose_single_qubit_matrix_gateset():
    q = cirq.LineQubit(0)
    device = ionq.IonQCompilationTargetGateset()
    for _ in range(100):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2))
        circuit = cirq.Circuit(gate(q))
        decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=device)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )


def test_decompose_single_qubit_matrix_gate():
    q = cirq.LineQubit(0)
    for _ in range(100):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2))
        circuit = cirq.Circuit(gate(q))
        decomposed_circuit = cirq.Circuit(*ionq.decompose_to_device(gate(q)))
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


def test_decompose_two_qubit_matrix_gateset():
    q0, q1 = cirq.LineQubit.range(2)
    device = ionq.IonQCompilationTargetGateset()
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=device)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


def test_decompose_two_qubit_matrix_gate():
    q0, q1 = cirq.LineQubit.range(2)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.Circuit(*ionq.decompose_to_device(gate(q0, q1)))
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


def test_decompose_two_qubit_matrix_gate():
    q0, q1 = cirq.LineQubit.range(2)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.Circuit(*ionq.decompose_to_device(gate(q0, q1)))
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


def test_decompose_unsupported_gate():
    q0, q1, q2 = cirq.LineQubit.range(3)
    op = cirq.CCZ(q0, q1, q2)
    with pytest.raises(ValueError, match='not supported'):
        _ = ionq.decompose_to_device(op)
