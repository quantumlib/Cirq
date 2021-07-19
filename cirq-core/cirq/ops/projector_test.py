from itertools import permutations

import numpy as np
import pytest

import cirq


def test_projector_qid():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})

    np.testing.assert_allclose(zero_projector.matrix(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector.matrix(), [[0.0, 0.0], [0.0, 1.0]])


def test_projector_from_np_array():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    np.testing.assert_allclose(zero_projector.matrix(), [[1.0, 0.0], [0.0, 0.0]])


def test_projector_matrix_missing_qid():
    q0, q1 = cirq.LineQubit.range(2)
    proj = cirq.ProjectorString({q0: 0})

    np.testing.assert_allclose(proj.matrix(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q0]), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1]), np.diag([1.0, 1.0]))

    np.testing.assert_allclose(proj.matrix([q0, q1]), np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]), np.diag([1.0, 0.0, 1.0, 0.0]))


def test_projector_from_state_missing_qid():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    d = cirq.ProjectorString({q0: 0})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_state_vector(np.array([[0.0, 0.0]]), qid_map={q1: 0})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 0.0]]), qid_map={q1: 0})


def test_equality():
    q0 = cirq.NamedQubit('q0')

    obj1 = cirq.ProjectorString({q0: 0})
    obj2 = cirq.ProjectorString({q0: 1})

    assert obj1 == obj1
    assert obj1 != obj2
    assert hash(obj1) == hash(obj1)
    assert hash(obj1) != hash(obj2)


def test_projector_qutrit():
    (q0,) = cirq.LineQid.range(1, dimension=3)

    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})
    two_projector = cirq.ProjectorString({q0: 2})

    np.testing.assert_allclose(zero_projector.matrix(), np.diag([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(one_projector.matrix(), np.diag([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(two_projector.matrix(), np.diag([0.0, 0.0, 1.0]))


def test_get_values():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0})

    assert len(d._projector_dict_()) == 1
    assert np.allclose(d._projector_dict_()[q0], 0)


def test_expectation_from_state_vector_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({})

    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)


def test_expectation_from_state_vector_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0})

    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)
    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([0.0, 1.0]), {q0: 0}), 0.0)


def test_expectation_from_state_vector_basis_states_three_qubits():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    d = cirq.ProjectorString({q0: 0, q1: 1})

    np.testing.assert_allclose(
        d.expectation_from_state_vector(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), {q0: 0, q1: 1, q2: 2}
        ),
        0.0,
    )

    np.testing.assert_allclose(
        d.expectation_from_state_vector(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), {q0: 0, q1: 2, q2: 1}
        ),
        1.0,
    )


def test_expectation_higher_dims():
    q0 = cirq.NamedQid('q0', dimension=2)
    q1 = cirq.NamedQid('q1', dimension=3)
    q2 = cirq.NamedQid('q2', dimension=5)
    d = cirq.ProjectorString({q2: 3, q1: 1})

    phis = [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]

    for perm in permutations([0, 1, 2]):
        inv_perm = [-1] * len(perm)
        for i, j in enumerate(perm):
            inv_perm[j] = i

        state_vector = np.kron(phis[perm[0]], np.kron(phis[perm[1]], phis[perm[2]]))
        state = np.einsum('i,j->ij', state_vector, state_vector.T.conj())

        np.testing.assert_allclose(
            d.expectation_from_state_vector(
                state_vector, {q0: inv_perm[0], q1: inv_perm[1], q2: inv_perm[2]}
            ),
            1.0,
        )

        np.testing.assert_allclose(
            d.expectation_from_density_matrix(
                state, {q0: inv_perm[0], q1: inv_perm[1], q2: inv_perm[2]}
            ),
            1.0,
        )


def test_expectation_from_density_matrix_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 1.0
    )


def test_expectation_from_density_matrix_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 1.0
    )
    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 1.0]]), {q0: 0}), 0.0
    )


def test_projector_sum_basic():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})

    proj_sum = cirq.ProjectorSum.from_projector_strings(
        zero_projector
    ) + cirq.ProjectorSum.from_projector_strings(one_projector)

    np.testing.assert_allclose(proj_sum.matrix(), [[1.0, 0.0], [0.0, 1.0]])
