from itertools import permutations
import math

import numpy as np
import pytest

import cirq


def test_ket_bra_qid():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    zero_projector = cirq.KetBraSum({q0: [(0, 0)]})
    one_projector = cirq.KetBraSum({q0: [(1, 1)]})
    not_a_projector = cirq.KetBraSum({q0: [(0, 1)]})
    two_quids = cirq.KetBraSum({(q0, q1): [(1, 3)]})

    np.testing.assert_allclose(zero_projector.matrix(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector.matrix(), [[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(not_a_projector.matrix(), [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(two_quids.matrix(), [[0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]])


def test_ket_bra_from_np_array():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.KetBraSum({q0: [(0, 0)]})
    np.testing.assert_allclose(zero_projector.matrix(), [[1.0, 0.0], [0.0, 0.0]])


def test_ket_bra_plus():
    q0 = cirq.NamedQubit('q0')

    plus = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
    plus_projector = cirq.KetBraSum({q0: [(plus, plus)]})

    np.testing.assert_allclose(plus_projector.matrix(), [[0.5, 0.5], [0.5, 0.5]])


def test_ket_bra_matrix_missing_qid():
    q0, q1 = cirq.LineQubit.range(2)
    proj = cirq.KetBraSum({q0: [(0, 0)]})

    np.testing.assert_allclose(proj.matrix(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(proj.matrix([q0]), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(proj.matrix([q1]), [[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(proj.matrix([q0, q1]), np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]), np.diag([1.0, 0.0, 1.0, 0.0]))


def test_ket_bra_from_state_missing_qid():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    d = cirq.KetBraSum({q0: [(0, 0)]})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_state_vector(np.array([[0.0, 0.0]]), qid_map={q1: 0})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 0.0]]), qid_map={q1: 0})


def test_ket_bra_from_state_missing_proj_key():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    d = cirq.KetBraSum({(q0, q1): [(0, 0)]})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_state_vector(np.array([[0.0, 0.0]]), qid_map={q1: 0})

    with pytest.raises(ValueError, match="Missing qid: q0"):
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 0.0]]), qid_map={q1: 0})


def test_equality():
    q0 = cirq.NamedQubit('q0')

    obj1 = cirq.KetBraSum({q0: [(0, 0)]})
    obj2 = cirq.KetBraSum({q0: [(1, 1)]})

    assert obj1 == obj1
    assert obj1 != obj2
    assert hash(obj1) == hash(obj1)
    assert hash(obj1) != hash(obj2)


def test_ket_bra_qutrit():
    (q0,) = cirq.LineQid.range(1, dimension=3)

    zero_projector = cirq.KetBraSum({q0: [(0, 0)]})
    one_projector = cirq.KetBraSum({q0: [(1, 1)]})
    two_projector = cirq.KetBraSum({q0: [(2, 2)]})

    np.testing.assert_allclose(
        zero_projector.matrix(), [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    np.testing.assert_allclose(
        one_projector.matrix(), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )

    np.testing.assert_allclose(
        two_projector.matrix(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )


def test_get_values():
    q0 = cirq.NamedQubit('q0')
    d = cirq.KetBraSum({q0: [(0, 0)]})

    assert len(d._ket_bra_dict_()) == 1
    assert np.allclose(d._ket_bra_dict_()[q0], [(0, 0)])


def test_repr():
    q0 = cirq.NamedQubit('q0')
    d = cirq.KetBraSum({q0: [[1.0, 0.0]]})

    assert d.__repr__() == (
        "cirq.KetBraSum(ket_bra_dict={cirq.NamedQubit('q0'): [[1.0, 0.0]]})"
    )


def test_consistency_with_existing():
    a, b = cirq.LineQubit.range(2)
    mx = (cirq.KET_IMAG(a) * cirq.KET_IMAG(b)).projector()
    proj_vec = np.asarray([0.5, 0.5j, 0.5j, -0.5])
    ii_proj = cirq.KetBraSum({(a, b): [(proj_vec, proj_vec.conj())]})
    np.testing.assert_allclose(mx, ii_proj.matrix())


def test_expectation_from_state_vector_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.KetBraSum({})

    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)


def test_expectation_from_state_vector_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.KetBraSum({q0: [(0, 0)]})

    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)
    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([0.0, 1.0]), {q0: 0}), 0.0)


def test_expectation_from_state_vector_basis_states_three_qubits():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    d = cirq.KetBraSum({q0: [(0, 0)], q1: [(1, 1)]})

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
    d = cirq.KetBraSum({q2: [(3, 3)], q1: [(1, 1)]})

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
    d = cirq.KetBraSum({})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 1.0
    )


def test_expectation_from_density_matrix_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.KetBraSum({q0: [(0, 0)]})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 1.0
    )
    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 1.0]]), {q0: 0}), 0.0
    )


def test_internal_consistency():
    q0 = cirq.NamedQid('q0', dimension=2)
    q1 = cirq.NamedQid('q1', dimension=3)

    phi0 = np.asarray([1.0, -3.0j])
    phi1 = np.asarray([-0.5j, 1.0 + 2.0j, 1.2])

    state_vector = np.asarray([1.0, 2.0j, -3.0, -4.0j, -5.0j, 0.0])

    phi0 = phi0 / np.linalg.norm(phi0)
    phi1 = phi1 / np.linalg.norm(phi1)
    state_vector = state_vector / np.linalg.norm(state_vector)
    state = np.einsum('i,j->ij', state_vector, state_vector.T.conj())

    d = cirq.KetBraSum({q0: [(phi0, phi0)], q1: [(phi1, phi1)]})
    P = d.matrix(ket_bra_keys=[q1, q0])

    projected_state = np.matmul(P, state_vector)
    actual0 = np.linalg.norm(projected_state, ord=2) ** 2

    actual1 = d.expectation_from_state_vector(state_vector, qid_map={q1: 0, q0: 1})

    actual2 = d.expectation_from_density_matrix(state, qid_map={q1: 0, q0: 1})

    np.testing.assert_allclose(actual0, actual1, atol=1e-6)
    np.testing.assert_allclose(actual0, actual2, atol=1e-6)


def test_ket_bra_split_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    phi = np.asarray([1.0 / math.sqrt(2), 0.0, 0.0, 1.0 / math.sqrt(2)])
    d = cirq.KetBraSum({(q0, q2): [(phi, phi.conj())]})

    qid_map = {q0: 0, q1: 1, q2: 2}

    state_vector = np.asarray([0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5])
    state = np.einsum('i,j->ij', state_vector, state_vector.T.conj())

    actual1 = d.expectation_from_state_vector(state_vector, qid_map=qid_map)
    actual2 = d.expectation_from_density_matrix(state, qid_map=qid_map)

    np.testing.assert_allclose(actual1, 0.25, atol=1e-6)
    np.testing.assert_allclose(actual2, 0.25, atol=1e-6)
