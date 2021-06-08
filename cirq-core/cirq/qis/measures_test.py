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
"""Tests for measures."""
import numpy as np
import pytest

import cirq

N = 15
VEC1 = cirq.testing.random_superposition(N)
VEC2 = cirq.testing.random_superposition(N)
MAT1 = cirq.testing.random_density_matrix(N)
MAT2 = cirq.testing.random_density_matrix(N)
U = cirq.testing.random_unitary(N)


def test_fidelity_symmetric():
    np.testing.assert_allclose(cirq.fidelity(VEC1, VEC2), cirq.fidelity(VEC2, VEC1))
    np.testing.assert_allclose(cirq.fidelity(VEC1, MAT1), cirq.fidelity(MAT1, VEC1))
    np.testing.assert_allclose(
        cirq.fidelity(cirq.density_matrix(MAT1), MAT2),
        cirq.fidelity(cirq.density_matrix(MAT2), MAT1),
    )


def test_fidelity_between_zero_and_one():
    assert 0 <= cirq.fidelity(VEC1, VEC2) <= 1
    assert 0 <= cirq.fidelity(VEC1, MAT1) <= 1
    assert 0 <= cirq.fidelity(cirq.density_matrix(MAT1), MAT2) <= 1


def test_fidelity_invariant_under_unitary_transformation():
    np.testing.assert_allclose(
        cirq.fidelity(cirq.density_matrix(MAT1), MAT2),
        cirq.fidelity(cirq.density_matrix(U @ MAT1 @ U.T.conj()), U @ MAT2 @ U.T.conj()),
    )


def test_fidelity_commuting_matrices():
    d1 = np.random.uniform(size=N)
    d1 /= np.sum(d1)
    d2 = np.random.uniform(size=N)
    d2 /= np.sum(d2)
    mat1 = cirq.density_matrix(U @ np.diag(d1) @ U.T.conj())
    mat2 = U @ np.diag(d2) @ U.T.conj()

    np.testing.assert_allclose(
        cirq.fidelity(mat1, mat2, qid_shape=(15,)), np.sum(np.sqrt(d1 * d2)) ** 2
    )


def test_fidelity_known_values():
    vec1 = np.array([1, 1j, -1, -1j]) * 0.5
    vec2 = np.array([1, -1, 1, -1], dtype=np.complex128) * 0.5
    vec3 = np.array([1, 0, 0, 0], dtype=np.complex128)
    tensor1 = np.reshape(vec1, (2, 2))
    mat1 = cirq.density_matrix(np.outer(vec1, vec1.conj()))
    mat2 = cirq.density_matrix(np.outer(vec2, vec2.conj()))
    mat3 = 0.3 * mat1.density_matrix() + 0.7 * mat2.density_matrix()

    np.testing.assert_allclose(cirq.fidelity(vec1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, vec3), 0.25)
    np.testing.assert_allclose(cirq.fidelity(vec1, tensor1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, mat2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(mat2, vec2), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat2), 0)
    np.testing.assert_allclose(cirq.fidelity(mat1, vec2), 0)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat3), 0.3)
    np.testing.assert_allclose(cirq.fidelity(tensor1, mat3), 0.3)
    np.testing.assert_allclose(cirq.fidelity(mat3, tensor1), 0.3)
    np.testing.assert_allclose(cirq.fidelity(mat3, vec2), 0.7)


def test_fidelity_numpy_arrays():
    vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
    vec2 = np.array([1, 0, 0, 0], dtype=np.complex64)
    tensor1 = np.reshape(vec1, (2, 2, 2))
    tensor2 = np.reshape(vec2, (2, 2))
    mat1 = np.outer(vec1, vec1.conj())

    np.testing.assert_allclose(cirq.fidelity(vec1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, tensor1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor2, tensor2, qid_shape=(2, 2)), 1)
    np.testing.assert_allclose(cirq.fidelity(mat1, mat1, qid_shape=(8,)), 1)

    with pytest.raises(ValueError, match='dimension'):
        _ = cirq.fidelity(vec1, vec1, qid_shape=(4,))

    with pytest.raises(ValueError, match='Mismatched'):
        _ = cirq.fidelity(vec1, vec2)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.fidelity(tensor2, tensor2)

    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.fidelity(mat1, mat1)


def test_fidelity_ints():
    assert cirq.fidelity(3, 4) == 0.0
    assert cirq.fidelity(4, 4) == 1.0

    with pytest.raises(ValueError, match='non-negative'):
        _ = cirq.fidelity(-1, 2)
    with pytest.raises(ValueError, match='maximum'):
        _ = cirq.fidelity(4, 1, qid_shape=(2,))
    with pytest.raises(ValueError, match='maximum'):
        _ = cirq.fidelity(1, 4, qid_shape=(2,))


def test_fidelity_product_states():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_ZERO(b)), 1.0
    )
    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_ONE(b)),
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_PLUS(b)), 0.5
    )
    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_ONE(a) * cirq.KET_ONE(b), cirq.KET_MINUS(a) * cirq.KET_PLUS(b)), 0.25
    )
    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_MINUS(a) * cirq.KET_PLUS(b), cirq.KET_MINUS(a) * cirq.KET_PLUS(b)),
        1.0,
    )
    np.testing.assert_allclose(
        cirq.fidelity(cirq.KET_MINUS(a) * cirq.KET_PLUS(b), cirq.KET_PLUS(a) * cirq.KET_MINUS(b)),
        0.0,
        atol=1e-7,
    )

    with pytest.raises(ValueError, match='Mismatched'):
        _ = cirq.fidelity(cirq.KET_MINUS(a), cirq.KET_PLUS(a) * cirq.KET_MINUS(b))
    with pytest.raises(ValueError, match='qid shape'):
        _ = cirq.fidelity(
            cirq.KET_MINUS(a) * cirq.KET_PLUS(b),
            cirq.KET_PLUS(a) * cirq.KET_MINUS(b),
            qid_shape=(4,),
        )


def test_fidelity_fail_inference():
    state_vector = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor = np.reshape(state_vector, (2, 2))
    with pytest.raises(ValueError, match='Please specify'):
        _ = cirq.fidelity(state_tensor, 4)


def test_fidelity_bad_shape():
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.fidelity(np.array([[[1.0]]]), np.array([[[1.0]]]), qid_shape=(1,))


def test_von_neumann_entropy():
    # 1x1 matrix
    assert cirq.von_neumann_entropy(np.array([[1]])) == 0
    # An EPR pair state (|00> + |11>)(<00| + <11|)
    assert (
        cirq.von_neumann_entropy(0.5 * np.array([1, 0, 0, 1] * np.array([[1], [0], [0], [1]]))) == 0
    )
    # Maximally mixed state
    # yapf: disable
    assert cirq.von_neumann_entropy(np.array(
        [[0.5, 0],
        [0, 0.5]])) == 1
    # yapf: enable
    # 2x2 random unitary, each column as a ket, each ket as a density matrix,
    # linear combination of the two with coefficients 0.1 and 0.9
    res = cirq.testing.random_unitary(2)
    first_column = res[:, 0]
    first_density_matrix = 0.1 * np.outer(first_column, np.conj(first_column))
    second_column = res[:, 1]
    second_density_matrix = 0.9 * np.outer(second_column, np.conj(second_column))
    assert np.isclose(
        cirq.von_neumann_entropy(first_density_matrix + second_density_matrix), 0.4689, atol=1e-04
    )

    assert np.isclose(
        cirq.von_neumann_entropy(np.diag([0, 0, 0.1, 0, 0.2, 0.3, 0.4, 0])), 1.8464, atol=1e-04
    )
    # Random NxN matrix
    probs = np.random.exponential(size=N)
    probs /= np.sum(probs)
    mat = U @ (probs * U).T.conj()

    np.testing.assert_allclose(
        cirq.von_neumann_entropy(mat), -np.sum(probs * np.log(probs) / np.log(2))
    )
    # QuantumState object
    assert (
        cirq.von_neumann_entropy(cirq.quantum_state(np.array([[0.5, 0], [0, 0.5]]), qid_shape=(2,)))
        == 1
    )
    assert (
        cirq.von_neumann_entropy(
            cirq.quantum_state(np.array([[0.5, 0.5], [0.5, 0.5]]), qid_shape=(2, 2))
        )
        == 0
    )


@pytest.mark.parametrize(
    'gate, expected_entanglement_fidelity',
    (
        (cirq.I, 1),
        (cirq.X, 0),
        (cirq.Y, 0),
        (cirq.Z, 0),
        (cirq.S, 1 / 2),
        (cirq.CNOT, 1 / 4),
        (cirq.TOFFOLI, 9 / 16),
    ),
)
def test_entanglement_fidelity_of_unitary_channels(gate, expected_entanglement_fidelity):
    assert np.isclose(cirq.entanglement_fidelity(gate), expected_entanglement_fidelity)


@pytest.mark.parametrize('p', (0, 0.1, 0.2, 0.5, 0.8, 0.9, 1))
@pytest.mark.parametrize(
    'channel_factory, entanglement_fidelity_formula',
    (
        # Each Pauli error turns the maximally entangled state into an orthogonal state, so only
        # the error-free term, whose pre-factor is 1 - p, contributes to entanglement fidelity.
        (cirq.depolarize, lambda p: 1 - p),
        (lambda p: cirq.depolarize(p, n_qubits=2), lambda p: 1 - p),
        (lambda p: cirq.depolarize(p, n_qubits=3), lambda p: 1 - p),
        # See e.g. https://quantumcomputing.stackexchange.com/questions/16074 for average fidelity,
        # then use Horodecki formula F_avg = (N F_e + 1) / (N + 1) to find entanglement fidelity.
        (cirq.amplitude_damp, lambda gamma: 1 / 2 - gamma / 4 + np.sqrt(1 - gamma) / 2),
    ),
)
def test_entanglement_fidelity_of_noisy_channels(p, channel_factory, entanglement_fidelity_formula):
    channel = channel_factory(p)
    actual_entanglement_fidelity = cirq.entanglement_fidelity(channel)
    expected_entanglement_fidelity = entanglement_fidelity_formula(p)
    assert np.isclose(actual_entanglement_fidelity, expected_entanglement_fidelity)
