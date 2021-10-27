# Copyright 2021 The Cirq Developers
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
    cirq.X ** 0.5,
    cirq.Y ** 0.5,
    cirq.Z ** 0.5,
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
    cirq.XX ** 0.5,
    cirq.YY ** 0.5,
    cirq.ZZ ** 0.5,
    cirq.SWAP,
    cirq.SwapPowGate(exponent=1, global_shift=-0.5),
    cirq.MeasurementGate(num_qubits=1, key='a'),
    cirq.MeasurementGate(num_qubits=2, key='b'),
    cirq.MeasurementGate(num_qubits=10, key='c'),
)


@pytest.mark.parametrize('gate', VALID_GATES)
def test_validate_operation_valid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    device.validate_operation(operation)


INVALID_GATES = (
    cirq.CNOT ** 0.5,
    cirq.SWAP ** 0.5,
    cirq.CCX,
    cirq.CCZ,
    cirq.CZ,
)


@pytest.mark.parametrize('gate', INVALID_GATES)
def test_validate_operation_invalid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    with pytest.raises(ValueError, match='unsupported gate'):
        device.validate_operation(operation)


def test_validate_operation_no_gate():
    device = ionq.IonQAPIDevice(qubits=[])
    with pytest.raises(ValueError, match='no gates'):
        device.validate_operation(cirq.GlobalPhaseOperation(1j))


def test_validate_operation_qubit_not_on_device():
    device = ionq.IonQAPIDevice(qubits=[cirq.LineQubit(0)])
    with pytest.raises(ValueError, match='not on the device'):
        device.validate_operation(cirq.H(cirq.LineQubit(1)))


def test_validate_moment_valid():
    moment = cirq.Moment()
    q = 0
    all_qubits = []
    for gate in VALID_GATES:
        qubits = cirq.LineQubit.range(q, q + gate.num_qubits())
        all_qubits.extend(qubits)
        moment += [gate(*qubits)]
        q += gate.num_qubits()
    device = ionq.IonQAPIDevice(len(all_qubits))
    device.validate_moment(moment)


@pytest.mark.parametrize('gate', INVALID_GATES)
def test_validate_moment_invalid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    moment = cirq.Moment([gate(*qubits)])
    device = ionq.IonQAPIDevice(qubits=qubits)
    with pytest.raises(ValueError, match='unsupported gate'):
        device.validate_moment(moment)


def test_validate_circuit_valid():
    qubits = cirq.LineQubit.range(10)
    device = ionq.IonQAPIDevice(qubits)
    for _ in range(100):
        circuit = cirq.testing.random_circuit(
            qubits=qubits,
            n_moments=3,
            op_density=0.5,
            gate_domain={gate: gate.num_qubits() for gate in VALID_GATES},
        )
        device.validate_circuit(circuit)


@pytest.mark.parametrize('gate', VALID_GATES)
def test_decompose_leaves_supported_alone(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    assert device.decompose_operation(operation) == operation


def test_decompose_single_qubit_matrix_gate():
    q = cirq.LineQubit(0)
    device = ionq.IonQAPIDevice(qubits=[q])
    for _ in range(100):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2))
        circuit = cirq.Circuit(gate(q))
        decomposed_circuit = cirq.Circuit(*device.decompose_operation(gate(q)))
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )


def test_decompose_two_qubit_matrix_gate():
    q0, q1 = cirq.LineQubit.range(2)
    device = ionq.IonQAPIDevice(qubits=[q0, q1])
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.Circuit(*device.decompose_operation(gate(q0, q1)))
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )


def test_decompose_unsupported_gate():
    q0, q1, q2 = cirq.LineQubit.range(3)
    device = ionq.IonQAPIDevice(qubits=[q0, q1, q2])
    op = cirq.CCZ(q0, q1, q2)
    with pytest.raises(ValueError, match='not supported'):
        _ = device.decompose_operation(op)
