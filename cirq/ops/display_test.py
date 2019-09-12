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


class Zero(cirq.DensityMatrixDisplay):

    def __init__(self, qubits, key):
        self._qubits = tuple(qubits)
        self._key = key

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return Zero(new_qubits, self._key) # coverage: ignore

    @property
    def key(self):
        return self._key

    def value_derived_from_density_matrix(self, state, qubit_map):
        return 0


def test_density_matrix_display_on_wavefunction():
    zero_display = Zero([cirq.LineQubit(0)], key='zero')
    circuit = cirq.Circuit.from_ops(zero_display)
    simulator = cirq.Simulator()
    result = simulator.compute_displays(circuit)
    assert result.display_values['zero'] == 0


@pytest.mark.parametrize('paulis', [
    (cirq.Z, cirq.Z),
    (cirq.Z, cirq.X),
    (cirq.X, cirq.X),
    (cirq.X, cirq.Y),
])
def test_approx_pauli_string_expectation_measurement_basis_change(paulis):
    qubits = cirq.LineQubit.range(2)
    qubit_map = {qubits[0]: paulis[0], qubits[1]: paulis[1]}
    display = cirq.approx_pauli_string_expectation(cirq.PauliString(qubit_map),
                                                   num_samples=1)
    matrix = np.kron(cirq.unitary(paulis[0]), cirq.unitary(paulis[1]))

    circuit = cirq.Circuit.from_ops(display.measurement_basis_change())
    unitary = circuit.unitary(qubit_order=qubits)

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
    display = cirq.approx_pauli_string_expectation(cirq.PauliString({}),
                                                   num_samples=1)
    assert display.value_derived_from_samples(measurements) == value


@pytest.mark.parametrize('measurements, value, coefficient', [
    (np.array([[0, 0, 0], [0, 0, 0]]), 1, 0.123),
    (np.array([[0, 0, 0], [0, 0, 1]]), 0, 999),
    (np.array([[0, 1, 0], [1, 0, 0]]), -1, -1),
    (np.array([[0, 1, 0], [1, 1, 1]]), -1, 1),
])
def test_approx_pauli_string_expectation_value_with_coef(
        measurements, value, coefficient):
    display = cirq.approx_pauli_string_expectation(cirq.PauliString(
        {}, coefficient=coefficient),
                                                   num_samples=1)
    assert display.value_derived_from_samples(
        measurements) == value * coefficient


def test_properties():
    qubits = cirq.LineQubit.range(9)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    approx_pauli_string_expectation = cirq.approx_pauli_string_expectation(
        pauli_string, num_samples=5, key='a')
    assert approx_pauli_string_expectation.qubits == tuple(qubits)
    assert approx_pauli_string_expectation.num_samples == 5
    assert approx_pauli_string_expectation.key == 'a'


def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    assert (cirq.approx_pauli_string_expectation(
        pauli_string, num_samples=1).with_qubits(*new_qubits) ==
            cirq.approx_pauli_string_expectation(
                pauli_string.with_qubits(*new_qubits), num_samples=1))


def test_approx_pauli_string_expectation_helper():
    qubits = cirq.LineQubit.range(9)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    assert (cirq.approx_pauli_string_expectation(
        pauli_string, num_samples=5,
        key='a') == cirq.approx_pauli_string_expectation(pauli_string,
                                                         num_samples=5,
                                                         key='a'))
