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


def test_pauli_string_expectation_value_pure_state():
    qubits = cirq.LineQubit.range(4)
    qubit_index_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit.from_ops(
            cirq.X(qubits[1]),
            cirq.H(qubits[2]),
            cirq.X(qubits[3]),
            cirq.H(qubits[3]),
    )
    wavefunction = circuit.final_wavefunction(qubit_order=qubits)
    density_matrix = np.outer(wavefunction, np.conj(wavefunction))

    z0z1 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[1]: cirq.Z})
            )
    z0z2 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[2]: cirq.Z})
            )
    z0z3 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[3]: cirq.Z})
            )
    z0x1 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[1]: cirq.X})
            )
    z1x2 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[1]: cirq.Z,
                              qubits[2]: cirq.X})
            )
    x0z1 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.X,
                              qubits[1]: cirq.Z})
            )
    x3 = cirq.pauli_string_expectation(
            cirq.PauliString({qubits[3]: cirq.X})
            )

    np.testing.assert_allclose(
        z0z1.value_derived_from_wavefunction(wavefunction, qubit_index_map), -1)
    np.testing.assert_allclose(
        z0z2.value_derived_from_wavefunction(wavefunction, qubit_index_map), 0)
    np.testing.assert_allclose(
        z0z3.value_derived_from_wavefunction(wavefunction, qubit_index_map), 0)
    np.testing.assert_allclose(
        z0x1.value_derived_from_wavefunction(wavefunction, qubit_index_map), 0)
    np.testing.assert_allclose(
        z1x2.value_derived_from_wavefunction(wavefunction, qubit_index_map), -1)
    np.testing.assert_allclose(
        x0z1.value_derived_from_wavefunction(wavefunction, qubit_index_map), 0)
    np.testing.assert_allclose(
        x3.value_derived_from_wavefunction(wavefunction, qubit_index_map), -1)

    np.testing.assert_allclose(
        z0z1.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), -1)
    np.testing.assert_allclose(
        z0z2.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), 0)
    np.testing.assert_allclose(
        z0z3.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), 0)
    np.testing.assert_allclose(
        z0x1.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), 0)
    np.testing.assert_allclose(
        z1x2.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), -1)
    np.testing.assert_allclose(
        x0z1.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), 0)
    np.testing.assert_allclose(
        x3.value_derived_from_density_matrix(
            density_matrix, qubit_index_map), -1)


def test_pauli_string_expectation_value_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    qubit_index_map = {q: i for i, q in enumerate(qs)}

    circuit = cirq.Circuit.from_ops(
        cirq.X(qs[1]),
        cirq.H(qs[2]),
        cirq.X(qs[3]),
        cirq.H(qs[3]),
    )
    wavefunction = circuit.apply_unitary_effect_to_state(qubit_order=qs)
    density_matrix = np.outer(wavefunction, np.conj(wavefunction))

    z0z1 = cirq.pauli_string_expectation(cirq.Z(qs[0]) * cirq.Z(qs[1]) * .123)
    z0z2 = cirq.pauli_string_expectation(cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1)
    z1x2 = cirq.pauli_string_expectation(-cirq.Z(qs[1]) * cirq.X(qs[2]))

    np.testing.assert_allclose(
        z0z1.value_derived_from_wavefunction(wavefunction, qubit_index_map),
        -0.123)
    np.testing.assert_allclose(
        z0z2.value_derived_from_wavefunction(wavefunction, qubit_index_map), 0)
    np.testing.assert_allclose(
        z1x2.value_derived_from_wavefunction(wavefunction, qubit_index_map), 1)

    np.testing.assert_allclose(
        z0z1.value_derived_from_density_matrix(density_matrix, qubit_index_map),
        -0.123)
    np.testing.assert_allclose(
        z0z2.value_derived_from_density_matrix(density_matrix, qubit_index_map),
        0)
    np.testing.assert_allclose(
        z1x2.value_derived_from_density_matrix(density_matrix, qubit_index_map),
        1)


def test_pauli_string_expectation_value_mixed_state_linearity():
    n_qubits = 10

    wavefunction1 = cirq.testing.random_superposition(2**n_qubits)
    wavefunction2 = cirq.testing.random_superposition(2**n_qubits)

    rho1 = np.outer(wavefunction1, np.conj(wavefunction1))
    rho2 = np.outer(wavefunction2, np.conj(wavefunction2))
    density_matrix = rho1 + rho2

    qubits = cirq.LineQubit.range(n_qubits)
    qubit_index_map = {q: i for i, q in enumerate(qubits)}
    paulis = [cirq.X, cirq.Y, cirq.Z]
    pauli_string_expectation = cirq.pauli_string_expectation(
            cirq.PauliString({q: np.random.choice(paulis) for q in qubits}))

    a = pauli_string_expectation.value_derived_from_wavefunction(
            wavefunction1, qubit_index_map)
    b = pauli_string_expectation.value_derived_from_wavefunction(
            wavefunction2, qubit_index_map)
    c = pauli_string_expectation.value_derived_from_density_matrix(
            density_matrix, qubit_index_map)

    np.testing.assert_allclose(a + b, c)


@pytest.mark.parametrize('paulis', [
    (cirq.Z, cirq.Z),
    (cirq.Z, cirq.X),
    (cirq.X, cirq.X),
    (cirq.X, cirq.Y),
])
def test_approx_pauli_string_expectation_measurement_basis_change(paulis):
    qubits = cirq.LineQubit.range(2)
    display = cirq.pauli_string_expectation(
        cirq.PauliString({qubits[0]: paulis[0],
                          qubits[1]: paulis[1]}),
        num_samples=1
    )
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
    display = cirq.pauli_string_expectation(
        cirq.PauliString({}),
        num_samples=1
    )
    assert display.value_derived_from_samples(measurements) == value


@pytest.mark.parametrize('measurements, value, coefficient', [
    (np.array([[0, 0, 0], [0, 0, 0]]), 1, 0.123),
    (np.array([[0, 0, 0], [0, 0, 1]]), 0, 999),
    (np.array([[0, 1, 0], [1, 0, 0]]), -1, -1),
    (np.array([[0, 1, 0], [1, 1, 1]]), -1, 1),
])
def test_approx_pauli_string_expectation_value_with_coef(
        measurements, value, coefficient):
    display = cirq.pauli_string_expectation(cirq.PauliString(
        {}, coefficient=coefficient),
                                            num_samples=1)
    assert display.value_derived_from_samples(
        measurements) == value * coefficient


def test_properties():
    qubits = cirq.LineQubit.range(9)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    pauli_string_expectation = cirq.pauli_string_expectation(
        pauli_string, key='a')
    assert pauli_string_expectation.qubits == tuple(qubits)
    assert pauli_string_expectation.key == 'a'

    approx_pauli_string_expectation = cirq.pauli_string_expectation(
        pauli_string, num_samples=5, key='a')
    assert approx_pauli_string_expectation.qubits == tuple(qubits)
    assert approx_pauli_string_expectation.num_samples == 5
    assert approx_pauli_string_expectation.key == 'a'


def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    assert (
        cirq.pauli_string_expectation(pauli_string).with_qubits(*new_qubits)
        == cirq.pauli_string_expectation(pauli_string.with_qubits(*new_qubits))
    )
    assert (
        cirq.pauli_string_expectation(
            pauli_string, num_samples=1).with_qubits(*new_qubits)
        == cirq.pauli_string_expectation(
            pauli_string.with_qubits(*new_qubits), num_samples=1))


def test_pauli_string_expectation_helper():
    qubits = cirq.LineQubit.range(9)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)

    assert (cirq.pauli_string_expectation(pauli_string, key='a')
            == cirq.pauli_string_expectation(pauli_string, key='a'))
    assert (cirq.pauli_string_expectation(pauli_string, 5, key='a')
            == cirq.pauli_string_expectation(pauli_string, 5, key='a'))
