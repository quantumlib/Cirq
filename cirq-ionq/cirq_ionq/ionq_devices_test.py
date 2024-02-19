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

import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES


@pytest.mark.parametrize('gate', VALID_GATES)
def test_validate_operation_valid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    device.validate_operation(operation)


INVALID_GATES = (cirq.CNOT**0.5, cirq.SWAP**0.5, cirq.CCX, cirq.CCZ, cirq.CZ)


@pytest.mark.parametrize('gate', INVALID_GATES)
def test_validate_operation_invalid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    with pytest.raises(ValueError, match='unsupported gate'):
        device.validate_operation(operation)


def test_metadata():
    device = ionq.IonQAPIDevice(qubits=[cirq.LineQubit(0)])
    assert device.metadata.qubit_set == {cirq.LineQubit(0)}


def test_validate_operation_no_gate():
    device = ionq.IonQAPIDevice(qubits=[])
    with pytest.raises(ValueError, match='no gates'):
        device.validate_operation(cirq.CircuitOperation(cirq.FrozenCircuit()))


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
