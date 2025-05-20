# Copyright 2019 The Cirq Developers
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

from __future__ import annotations

import numpy as np
import pytest

import cirq
import cirq.testing


def assert_dirac_notation_numpy(vec, expected, decimals=2):
    assert cirq.dirac_notation(np.array(vec), decimals=decimals) == expected


def assert_dirac_notation_python(vec, expected, decimals=2):
    assert cirq.dirac_notation(vec, decimals=decimals) == expected


def assert_valid_density_matrix(matrix, num_qubits=None, qid_shape=None):
    if qid_shape is None and num_qubits is None:
        num_qubits = 1
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            matrix, num_qubits=num_qubits, qid_shape=qid_shape, dtype=matrix.dtype
        ),
        matrix,
    )


def test_quantum_state():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))
    density_matrix_1 = np.outer(state_vector_1, np.conj(state_vector_1))

    state = cirq.QuantumState(state_vector_1)
    assert state.data is state_vector_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state.state_vector(), state_vector_1)
    np.testing.assert_array_equal(state.state_tensor(), state_tensor_1)
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), state_vector_1)

    state = cirq.QuantumState(state_tensor_1, qid_shape=(2, 2))
    assert state.data is state_tensor_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state.state_vector(), state_vector_1)
    np.testing.assert_array_equal(state.state_tensor(), state_tensor_1)
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), state_vector_1)

    state = cirq.QuantumState(density_matrix_1, qid_shape=(2, 2))
    assert state.data is density_matrix_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    assert state.state_vector() is None
    assert state.state_tensor() is None
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), density_matrix_1)


def test_quantum_state_quantum_state():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    quantum_state = cirq.QuantumState(state_vector_1)

    state = cirq.quantum_state(quantum_state)
    assert state is quantum_state
    assert state.data is quantum_state.data
    assert state.dtype == np.complex128

    state = cirq.quantum_state(quantum_state, copy=True)
    assert state is not quantum_state
    assert state.data is not quantum_state.data
    assert state.dtype == np.complex128

    state = cirq.quantum_state(quantum_state, dtype=np.complex64)
    assert state is not quantum_state
    assert state.data is not quantum_state.data
    assert state.dtype == np.complex64

    with pytest.raises(ValueError, match='qid shape'):
        state = cirq.quantum_state(quantum_state, qid_shape=(4,))


def test_quantum_state_computational_basis_state():
    state = cirq.quantum_state(7, qid_shape=(3, 4))
    np.testing.assert_allclose(state.data, cirq.one_hot(index=7, shape=(12,), dtype=np.complex64))
    assert state.qid_shape == (3, 4)
    assert state.dtype == np.complex64

    state = cirq.quantum_state((0, 1, 2, 3), qid_shape=(1, 2, 3, 4), dtype=np.complex128)
    np.testing.assert_allclose(
        state.data, cirq.one_hot(index=(0, 1, 2, 3), shape=(1, 2, 3, 4), dtype=np.complex64)
    )
    assert state.qid_shape == (1, 2, 3, 4)
    assert state.dtype == np.complex128

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state(7)

    with pytest.raises(ValueError, match='out of range'):
        _ = cirq.quantum_state(7, qid_shape=(2, 2))

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state((0, 1, 2, 3))

    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.quantum_state((0, 1, 2, 3), qid_shape=(2, 2, 2, 2))

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state((0, 0, 1, 1), qid_shape=(1, 1, 2, 2))


def test_quantum_state_state_vector_state_tensor():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))

    state = cirq.quantum_state(state_vector_1, dtype=np.complex64)
    np.testing.assert_array_equal(state.data, state_vector_1)
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex64

    state = cirq.quantum_state(state_tensor_1, qid_shape=(2, 2))
    assert state.data is state_tensor_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state(state_tensor_1)

    with pytest.raises(ValueError, match='not compatible'):
        _ = cirq.quantum_state(state_tensor_1, qid_shape=(2, 3))


def test_quantum_state_density_matrix():
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4

    state = cirq.quantum_state(density_matrix_1, qid_shape=(4,), copy=True)
    assert state.data is not density_matrix_1
    np.testing.assert_array_equal(state.data, density_matrix_1)
    assert state.qid_shape == (4,)
    assert state.dtype == np.complex64

    with pytest.raises(ValueError, match='not compatible'):
        _ = cirq.quantum_state(density_matrix_1, qid_shape=(8,))


def test_quantum_state_product_state():
    q0, q1, q2 = cirq.LineQubit.range(3)
    product_state_1 = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ONE(q2)

    state = cirq.quantum_state(product_state_1)
    np.testing.assert_allclose(state.data, product_state_1.state_vector())
    assert state.qid_shape == (2, 2, 2)
    assert state.dtype == np.complex64

    with pytest.raises(ValueError, match='qid shape'):
        _ = cirq.quantum_state(product_state_1, qid_shape=(2, 2))


def test_density_matrix():
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex64)

    state = cirq.density_matrix(density_matrix_1)
    assert state.data is density_matrix_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex64

    with pytest.raises(ValueError, match='square'):
        _ = cirq.density_matrix(state_vector_1)


def test_infer_qid_shape():
    computational_basis_state_1 = [0, 0, 0, 1]
    computational_basis_state_2 = [0, 1, 2, 3]
    computational_basis_state_3 = [0, 1, 2, 4]
    computational_basis_state_4 = 9
    computational_basis_state_5 = [0, 1, 2, 4, 5]
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex64)
    state_vector_2 = cirq.one_hot(shape=(24,), dtype=np.complex64)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))
    state_tensor_2 = np.reshape(state_vector_2, (1, 2, 3, 4))
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4
    density_matrix_2 = np.eye(24, dtype=np.complex64) / 24
    q0, q1 = cirq.LineQubit.range(2)
    product_state_1 = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)

    assert cirq.qis.infer_qid_shape(
        computational_basis_state_1,
        state_vector_1,
        state_tensor_1,
        density_matrix_1,
        product_state_1,
    ) == (2, 2)

    assert cirq.qis.infer_qid_shape(
        product_state_1,
        density_matrix_1,
        state_tensor_1,
        state_vector_1,
        computational_basis_state_1,
    ) == (2, 2)

    assert cirq.qis.infer_qid_shape(
        computational_basis_state_1,
        computational_basis_state_2,
        computational_basis_state_4,
        state_tensor_2,
    ) == (1, 2, 3, 4)

    assert cirq.qis.infer_qid_shape(
        state_vector_2, density_matrix_2, computational_basis_state_4
    ) == (24,)

    assert cirq.qis.infer_qid_shape(state_tensor_2, density_matrix_2) == (1, 2, 3, 4)

    assert cirq.qis.infer_qid_shape(computational_basis_state_4) == (10,)
    assert cirq.qis.infer_qid_shape(15, 7, 22, 4) == (23,)

    with pytest.raises(ValueError, match='No states were specified'):
        _ = cirq.qis.infer_qid_shape()

    with pytest.raises(ValueError, match='Failed'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1, computational_basis_state_5)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(state_tensor_1)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(density_matrix_1)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1, computational_basis_state_2)

    with pytest.raises(ValueError, match='Failed'):
        _ = cirq.qis.infer_qid_shape(state_vector_1, computational_basis_state_4)

    with pytest.raises(ValueError, match='Failed to infer'):
        _ = cirq.qis.infer_qid_shape(state_vector_1, state_vector_2)

    with pytest.raises(ValueError, match='Failed to infer'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_3, state_tensor_2)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_zero_state(global_phase):
    zero_state = global_phase * np.array([1, 0])

    bloch = cirq.bloch_vector_from_state_vector(zero_state, 0)
    desired_simple = np.array([0, 0, 1])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_one_state(global_phase):
    one_state = global_phase * np.array([0, 1])

    bloch = cirq.bloch_vector_from_state_vector(one_state, 0)
    desired_simple = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_plus_state(global_phase):
    sqrt = np.sqrt(0.5)
    plus_state = global_phase * np.array([sqrt, sqrt])

    bloch = cirq.bloch_vector_from_state_vector(plus_state, 0)
    desired_simple = np.array([1, 0, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_minus_state(global_phase):
    sqrt = np.sqrt(0.5)
    minus_state = np.array([-1.0j * sqrt, 1.0j * sqrt])
    bloch = cirq.bloch_vector_from_state_vector(minus_state, 0)

    desired_simple = np.array([-1, 0, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_iplus_state(global_phase):
    sqrt = np.sqrt(0.5)
    iplus_state = global_phase * np.array([sqrt, 1j * sqrt])

    bloch = cirq.bloch_vector_from_state_vector(iplus_state, 0)
    desired_simple = np.array([0, 1, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_iminus_state(global_phase):
    sqrt = np.sqrt(0.5)
    iminus_state = global_phase * np.array([sqrt, -1j * sqrt])

    bloch = cirq.bloch_vector_from_state_vector(iminus_state, 0)
    desired_simple = np.array([0, -1, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_simple_th_zero():
    sqrt = np.sqrt(0.5)
    # State TH|0>.
    th_state = np.array([sqrt, 0.5 + 0.5j])
    bloch = cirq.bloch_vector_from_state_vector(th_state, 0)

    desired_simple = np.array([sqrt, sqrt, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_equal_sqrt3():
    sqrt3 = 1 / np.sqrt(3)
    test_state = np.array([0.888074, 0.325058 + 0.325058j])
    bloch = cirq.bloch_vector_from_state_vector(test_state, 0)

    desired_simple = np.array([sqrt3, sqrt3, sqrt3])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_multi_pure():
    plus_plus_state = np.array([0.5, 0.5, 0.5, 0.5])

    bloch_0 = cirq.bloch_vector_from_state_vector(plus_plus_state, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(plus_plus_state, 1)
    desired_simple = np.array([1, 0, 0])

    np.testing.assert_array_almost_equal(bloch_1, desired_simple)
    np.testing.assert_array_almost_equal(bloch_0, desired_simple)


def test_bloch_vector_multi_mixed():
    sqrt = np.sqrt(0.5)
    # Bell state 1/sqrt(2)(|00>+|11>)
    phi_plus = np.array([sqrt, 0.0, 0.0, sqrt])

    bloch_0 = cirq.bloch_vector_from_state_vector(phi_plus, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(phi_plus, 1)
    zero = np.zeros(3)

    np.testing.assert_array_almost_equal(bloch_0, zero)
    np.testing.assert_array_almost_equal(bloch_1, zero)

    rcnot_state = np.array([0.90612745, -0.07465783j, -0.37533028j, 0.18023996])
    bloch_mixed_0 = cirq.bloch_vector_from_state_vector(rcnot_state, 0)
    bloch_mixed_1 = cirq.bloch_vector_from_state_vector(rcnot_state, 1)

    true_mixed_0 = np.array([0.0, -0.6532815, 0.6532815])
    true_mixed_1 = np.array([0.0, 0.0, 0.9238795])

    np.testing.assert_array_almost_equal(true_mixed_0, bloch_mixed_0)
    np.testing.assert_array_almost_equal(true_mixed_1, bloch_mixed_1)


def test_bloch_vector_multi_big():
    five_qubit_plus_state = np.array([0.1767767] * 32)
    desired_simple = np.array([1, 0, 0])
    for qubit in range(5):
        bloch_i = cirq.bloch_vector_from_state_vector(five_qubit_plus_state, qubit)
        np.testing.assert_array_almost_equal(bloch_i, desired_simple)


def test_bloch_vector_invalid():
    with pytest.raises(ValueError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5]), 0)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5, 0.5]), -1)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5, 0.5]), 2)


def test_density_matrix_from_state_vector():
    test_state = np.array(
        [
            0.0 - 0.35355339j,
            0.0 + 0.35355339j,
            0.0 - 0.35355339j,
            0.0 + 0.35355339j,
            0.0 + 0.35355339j,
            0.0 - 0.35355339j,
            0.0 + 0.35355339j,
            0.0 - 0.35355339j,
        ]
    )

    full_rho = cirq.density_matrix_from_state_vector(test_state)
    np.testing.assert_array_almost_equal(full_rho, np.outer(test_state, np.conj(test_state)))

    rho_one = cirq.density_matrix_from_state_vector(test_state, [1])
    true_one = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])
    np.testing.assert_array_almost_equal(rho_one, true_one)

    rho_two_zero = cirq.density_matrix_from_state_vector(test_state, [0, 2])
    true_two_zero = np.array(
        [
            [0.25 + 0.0j, -0.25 + 0.0j, -0.25 + 0.0j, 0.25 + 0.0j],
            [-0.25 + 0.0j, 0.25 + 0.0j, 0.25 + 0.0j, -0.25 + 0.0j],
            [-0.25 + 0.0j, 0.25 + 0.0j, 0.25 + 0.0j, -0.25 + 0.0j],
            [0.25 + 0.0j, -0.25 + 0.0j, -0.25 + 0.0j, 0.25 + 0.0j],
        ]
    )
    np.testing.assert_array_almost_equal(rho_two_zero, true_two_zero)

    # two and zero will have same single qubit density matrix.
    rho_two = cirq.density_matrix_from_state_vector(test_state, [2])
    true_two = np.array([[0.5 + 0.0j, -0.5 + 0.0j], [-0.5 + 0.0j, 0.5 + 0.0j]])
    np.testing.assert_array_almost_equal(rho_two, true_two)
    rho_zero = cirq.density_matrix_from_state_vector(test_state, [0])
    np.testing.assert_array_almost_equal(rho_zero, true_two)


def test_density_matrix_invalid():
    bad_state = np.array([0.5, 0.5, 0.5])
    good_state = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state)
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state, [0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1, 0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1])


def test_dirac_notation():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation_numpy([0, 0], "0")
    assert_dirac_notation_python([1], "|⟩")
    assert_dirac_notation_numpy([sqrt, sqrt], "0.71|0⟩ + 0.71|1⟩")
    assert_dirac_notation_python([-sqrt, sqrt], "-0.71|0⟩ + 0.71|1⟩")
    assert_dirac_notation_numpy([sqrt, -sqrt], "0.71|0⟩ - 0.71|1⟩")
    assert_dirac_notation_python([-sqrt, -sqrt], "-0.71|0⟩ - 0.71|1⟩")
    assert_dirac_notation_numpy([sqrt, 1j * sqrt], "0.71|0⟩ + 0.71j|1⟩")
    assert_dirac_notation_python([sqrt, exp_pi_2], "0.71|0⟩ + (0.5+0.5j)|1⟩")
    assert_dirac_notation_numpy([exp_pi_2, -sqrt], "(0.5+0.5j)|0⟩ - 0.71|1⟩")
    assert_dirac_notation_python([exp_pi_2, 0.5 - 0.5j], "(0.5+0.5j)|0⟩ + (0.5-0.5j)|1⟩")
    assert_dirac_notation_numpy([0.5, 0.5, -0.5, -0.5], "0.5|00⟩ + 0.5|01⟩ - 0.5|10⟩ - 0.5|11⟩")
    assert_dirac_notation_python([0.71j, 0.71j], "0.71j|0⟩ + 0.71j|1⟩")


def test_dirac_notation_partial_state():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation_numpy([1, 0], "|0⟩")
    assert_dirac_notation_python([1j, 0], "1j|0⟩")
    assert_dirac_notation_numpy([0, 1], "|1⟩")
    assert_dirac_notation_python([0, 1j], "1j|1⟩")
    assert_dirac_notation_numpy([sqrt, 0, 0, sqrt], "0.71|00⟩ + 0.71|11⟩")
    assert_dirac_notation_python([sqrt, sqrt, 0, 0], "0.71|00⟩ + 0.71|01⟩")
    assert_dirac_notation_numpy([exp_pi_2, 0, 0, exp_pi_2], "(0.5+0.5j)|00⟩ + (0.5+0.5j)|11⟩")
    assert_dirac_notation_python([0, 0, 0, 1], "|11⟩")


def test_dirac_notation_precision():
    sqrt = np.sqrt(0.5)
    assert_dirac_notation_numpy([sqrt, sqrt], "0.7|0⟩ + 0.7|1⟩", decimals=1)
    assert_dirac_notation_python([sqrt, sqrt], "0.707|0⟩ + 0.707|1⟩", decimals=3)


def test_dirac_notation_invalid():
    with pytest.raises(ValueError, match='state_vector has incorrect size'):
        _ = cirq.dirac_notation([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='state_vector has incorrect size'):
        _ = cirq.dirac_notation([1.0, 1.0], qid_shape=(3,))


def test_to_valid_state_vector():
    with pytest.raises(ValueError, match='Computational basis state is out of range'):
        cirq.to_valid_state_vector(2, 1)
    np.testing.assert_almost_equal(
        cirq.to_valid_state_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_state_vector(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([0.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(0, 2), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(1, 2), np.array([0.0, 1.0, 0.0, 0.0]))

    v = cirq.to_valid_state_vector([0, 1, 2, 0], qid_shape=(3, 3, 3, 3))
    assert v.shape == (3**4,)
    assert v[6 + 9] == 1

    v = cirq.to_valid_state_vector([False, True, False, False], num_qubits=4)
    assert v.shape == (16,)
    assert v[4] == 1

    v = cirq.to_valid_state_vector([0, 1, 0, 0], num_qubits=2)
    assert v.shape == (4,)
    assert v[1] == 1

    v = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.complex64), qid_shape=(2, 1))
    assert v.shape == (2,)
    assert v[0] == 1


def test_to_valid_state_vector_creates_new_copy():
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    out = cirq.to_valid_state_vector(state, 2)
    assert out is not state


def test_invalid_to_valid_state_vector():
    with pytest.raises(ValueError, match="Please specify"):
        _ = cirq.to_valid_state_vector(np.array([1]))

    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(np.array([1.0, 0.0], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(-1, 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(5, 2)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector('0000', 2)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector('not an int', 2)
    with pytest.raises(ValueError, match=r'num_qubits != len\(qid_shape\)'):
        _ = cirq.to_valid_state_vector(0, 5, qid_shape=(1, 2, 3))

    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([3], qid_shape=(3,))
    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([-1], qid_shape=(3,))
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector([], qid_shape=(3,))
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector([0, 1], num_qubits=3)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector([1, 0], qid_shape=(2, 1))
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.int64), qid_shape=(2, 1))


def test_validate_normalized_state():
    cirq.validate_normalized_state_vector(cirq.testing.random_superposition(2), qid_shape=(2,))
    cirq.validate_normalized_state_vector(
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64), qid_shape=(2, 2)
    )
    with pytest.raises(ValueError, match='invalid dtype'):
        cirq.validate_normalized_state_vector(
            np.array([1, 1], dtype=np.complex64), qid_shape=(2, 2), dtype=np.complex128
        )
    with pytest.raises(ValueError, match='incorrect size'):
        cirq.validate_normalized_state_vector(
            np.array([1, 1], dtype=np.complex64), qid_shape=(2, 2)
        )
    with pytest.raises(ValueError, match='not normalized'):
        cirq.validate_normalized_state_vector(
            np.array([1.0, 0.2, 0.0, 0.0], dtype=np.complex64), qid_shape=(2, 2)
        )


def test_validate_density_matrix():
    cirq.validate_density_matrix(cirq.testing.random_density_matrix(2), qid_shape=(2,))
    with pytest.raises(ValueError, match='dtype'):
        cirq.to_valid_density_matrix(
            np.array([[1, 0], [0, 0]], dtype=np.complex64), qid_shape=(2,), dtype=np.complex128
        )
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[1, 0]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[1, 0.1], [0, 0]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0.1]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(
            np.array([[1.1, 0], [0, -0.1]], dtype=np.complex64), qid_shape=(2,)
        )


def test_to_valid_density_matrix_from_density_matrix():
    assert_valid_density_matrix(np.array([[1, 0], [0, 0]]))
    assert_valid_density_matrix(np.array([[0.5, 0], [0, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.2], [0.2, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.2 - 0.2j], [0.2 + 0.2j, 0.5]]))
    assert_valid_density_matrix(np.eye(4) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([1, 0, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(np.ones([4, 4]) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([0.2, 0.8, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(
        np.array([[0.2, 0, 0, 0.2 - 0.3j], [0, 0, 0, 0], [0, 0, 0, 0], [0.2 + 0.3j, 0, 0, 0.8]]),
        num_qubits=2,
    )

    assert_valid_density_matrix(np.array([[1, 0, 0]] + [[0, 0, 0]] * 2), qid_shape=(3,))
    assert_valid_density_matrix(
        np.array([[0, 0, 0], [0, 0.5, 0.5j], [0, -0.5j, 0.5]]), qid_shape=(3,)
    )
    assert_valid_density_matrix(np.eye(9) / 9.0, qid_shape=(3, 3))
    assert_valid_density_matrix(np.eye(12) / 12.0, qid_shape=(3, 4))
    assert_valid_density_matrix(np.ones([9, 9]) / 9.0, qid_shape=(3, 3))
    assert_valid_density_matrix(np.diag([0.2, 0.8, 0, 0]), qid_shape=(4,))


def test_to_valid_density_matrix_from_density_matrix_tensor():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            cirq.one_hot(shape=(2, 2, 2, 2, 2, 2), dtype=np.complex64), num_qubits=3
        ),
        cirq.one_hot(shape=(8, 8), dtype=np.complex64),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            cirq.one_hot(shape=(2, 3, 4, 2, 3, 4), dtype=np.complex64), qid_shape=(2, 3, 4)
        ),
        cirq.one_hot(shape=(24, 24), dtype=np.complex64),
    )


def test_to_valid_density_matrix_not_square():
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[1], [0]]), num_qubits=1)


def test_to_valid_density_matrix_size_mismatch_num_qubits():
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]]), num_qubits=2)
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.eye(4) / 4.0, num_qubits=1)


def test_to_valid_density_matrix_not_hermitian():
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[0.5, 0.5j], [0.5, 0.5j]]), num_qubits=1)
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(
            np.array(
                [[0.2, 0, 0, -0.2 - 0.3j], [0, 0, 0, 0], [0, 0, 0, 0], [0.2 + 0.3j, 0, 0, 0.8]]
            ),
            num_qubits=2,
        )


def test_to_valid_density_matrix_mismatched_qid_shape():
    with pytest.raises(ValueError, match=r'num_qubits != len\(qid_shape\)'):
        cirq.to_valid_density_matrix(np.eye(4) / 4, num_qubits=1, qid_shape=(2, 2))
    with pytest.raises(ValueError, match=r'num_qubits != len\(qid_shape\)'):
        cirq.to_valid_density_matrix(np.eye(4) / 4, num_qubits=2, qid_shape=(4,))
    with pytest.raises(ValueError, match='Both were None'):
        cirq.to_valid_density_matrix(np.eye(4) / 4)


def test_to_valid_density_matrix_not_unit_trace():
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, -0.1]]), num_qubits=1)
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.zeros([2, 2]), num_qubits=1)


def test_to_valid_density_matrix_not_positive_semidefinite():
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(
            np.array([[0.6, 0.5], [0.5, 0.4]], dtype=np.complex64), num_qubits=1
        )


def test_to_valid_density_matrix_wrong_dtype():
    with pytest.raises(ValueError, match='dtype'):
        cirq.to_valid_density_matrix(
            np.array([[1, 0], [0, 0]], dtype=np.complex64), num_qubits=1, dtype=np.complex128
        )


def test_to_valid_density_matrix_from_state_vector():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([1, 0], dtype=np.complex64), num_qubits=1
        ),
        np.array([[1, 0], [0, 0]]),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex64),
            num_qubits=1,
        ),
        np.array([[0.3, np.sqrt(0.3 * 0.7)], [np.sqrt(0.3 * 0.7), 0.7]]),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([np.sqrt(0.5), np.sqrt(0.5) * 1j], dtype=np.complex64),
            num_qubits=1,
        ),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array([0.5] * 4, dtype=np.complex64), num_qubits=2
        ),
        0.25 * np.ones((4, 4)),
    )


def test_to_valid_density_matrix_from_state_vector_tensor():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(
            density_matrix_rep=np.array(np.full((2, 2), 0.5), dtype=np.complex64), num_qubits=2
        ),
        0.25 * np.ones((4, 4)),
    )


def test_to_valid_density_matrix_from_state_invalid_state():
    with pytest.raises(ValueError, match="Invalid quantum state"):
        cirq.to_valid_density_matrix(np.array([1, 0, 0]), num_qubits=2)


def test_to_valid_density_matrix_from_computational_basis():
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=0, num_qubits=1), np.array([[1, 0], [0, 0]])
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=1, num_qubits=1), np.array([[0, 0], [0, 1]])
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=2, num_qubits=2),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
    )
    np.testing.assert_almost_equal(
        cirq.to_valid_density_matrix(density_matrix_rep=0, num_qubits=0), np.array([[1]])
    )


def test_to_valid_density_matrix_from_state_invalid_computational_basis():
    with pytest.raises(ValueError, match="out of range"):
        cirq.to_valid_density_matrix(-1, num_qubits=2)


def test_one_hot():
    result = cirq.one_hot(shape=4, dtype=np.int32)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, [1, 0, 0, 0])

    np.testing.assert_array_equal(
        cirq.one_hot(shape=[2, 3], dtype=np.complex64), [[1, 0, 0], [0, 0, 0]]
    )

    np.testing.assert_array_equal(
        cirq.one_hot(shape=[2, 3], dtype=np.complex64, index=(0, 2)), [[0, 0, 1], [0, 0, 0]]
    )

    np.testing.assert_array_equal(
        cirq.one_hot(shape=5, dtype=np.complex128, index=3), [0, 0, 0, 1, 0]
    )


def test_eye_tensor():
    assert np.all(cirq.eye_tensor((), dtype=int) == np.array(1))
    assert np.all(cirq.eye_tensor((1,), dtype=int) == np.array([[1]]))
    assert np.all(cirq.eye_tensor((2,), dtype=int) == np.array([[1, 0], [0, 1]]))  # yapf: disable
    assert np.all(
        cirq.eye_tensor((2, 2), dtype=int)
        == np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    )  # yapf: disable
    assert np.all(
        cirq.eye_tensor((2, 3), dtype=int)
        == np.array(
            [
                [[[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]],
                [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]],
            ]
        )
    )  # yapf: disable
    assert np.all(
        cirq.eye_tensor((3, 2), dtype=int)
        == np.array(
            [
                [[[1, 0], [0, 0], [0, 0]], [[0, 1], [0, 0], [0, 0]]],
                [[[0, 0], [1, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]]],
                [[[0, 0], [0, 0], [1, 0]], [[0, 0], [0, 0], [0, 1]]],
            ]
        )
    )  # yapf: disable
