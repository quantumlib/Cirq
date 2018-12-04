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


def test_pauli_string_expectation_value():
    qubits = cirq.LineQubit.range(4)
    qubit_index_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit.from_ops(
            cirq.X(qubits[1]),
            cirq.H(qubits[2]),
            cirq.X(qubits[3]),
            cirq.H(qubits[3]),
    )
    state = circuit.apply_unitary_effect_to_state(qubit_order=qubits)

    z0z1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[1]: cirq.Pauli.Z})
            )
    z0z2 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[2]: cirq.Pauli.Z})
            )
    z0z3 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[3]: cirq.Pauli.Z})
            )
    z0x1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[1]: cirq.Pauli.X})
            )
    z1x2 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[1]: cirq.Pauli.Z,
                              qubits[2]: cirq.Pauli.X})
            )
    x0z1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.X,
                              qubits[1]: cirq.Pauli.Z})
            )
    x3 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[3]: cirq.Pauli.X})
            )

    np.testing.assert_allclose(z0z1.value(state, qubit_index_map), -1)
    np.testing.assert_allclose(z0z2.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z0z3.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z0x1.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z1x2.value(state, qubit_index_map), -1)
    np.testing.assert_allclose(x0z1.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(x3.value(state, qubit_index_map), -1)


@pytest.mark.parametrize('paulis', [
    (cirq.Pauli.Z, cirq.Pauli.Z),
    (cirq.Pauli.Z, cirq.Pauli.X),
    (cirq.Pauli.X, cirq.Pauli.X),
    (cirq.Pauli.X, cirq.Pauli.Y),
])
def test_approx_pauli_string_expectation_measurement_basis_change(paulis):
    qubits = cirq.LineQubit.range(2)
    display = cirq.ApproxPauliStringExpectation(
        cirq.PauliString({qubits[0]: paulis[0],
                          qubits[1]: paulis[1]}),
        repetitions=1
    )
    matrix = np.kron(cirq.unitary(paulis[0]), cirq.unitary(paulis[1]))

    circuit = cirq.Circuit.from_ops(display.measurement_basis_change())
    unitary = circuit.to_unitary_matrix(qubit_order=qubits)

    ZZ = np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(
        np.dot(unitary, np.dot(matrix, unitary.T.conj())),
        ZZ
    )


@pytest.mark.parametrize('measurements, value', [
    (np.array([[0, 0, 0],
               [0, 0, 0]]),
     1),
    (np.array([[0, 0, 0],
               [0, 0, 1]]),
     0),
    (np.array([[0, 1, 0],
               [1, 0, 0]]),
     -1),
    (np.array([[0, 1, 0],
               [1, 1, 1]]),
     -1),
])
def test_approx_pauli_string_expectation_value(measurements, value):
    display = cirq.ApproxPauliStringExpectation(
        cirq.PauliString({}),
        repetitions=1
    )
    assert display.value(measurements) == value


def test_properties():
    qubits = cirq.LineQubit.range(9)
    qubit_pauli_map = {q: cirq.Pauli.XYZ[q.x % 3] for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, negated=True)

    pauli_string_expectation = cirq.PauliStringExpectation(
        pauli_string, key='a')
    assert pauli_string_expectation.qubits == tuple(qubits)
    assert pauli_string_expectation.key == 'a'

    approx_pauli_string_expectation = cirq.ApproxPauliStringExpectation(
        pauli_string, repetitions=5, key='a')
    assert approx_pauli_string_expectation.qubits == tuple(qubits)
    assert approx_pauli_string_expectation.repetitions == 5
    assert approx_pauli_string_expectation.key == 'a'


def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.XYZ[q.x % 3] for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, negated=True)

    assert (
        cirq.PauliStringExpectation(pauli_string).with_qubits(*new_qubits)
        == cirq.PauliStringExpectation(pauli_string.with_qubits(*new_qubits))
    )
    assert (
        cirq.ApproxPauliStringExpectation(
            pauli_string, repetitions=1).with_qubits(*new_qubits)
        == cirq.ApproxPauliStringExpectation(
            pauli_string.with_qubits(*new_qubits), repetitions=1))
