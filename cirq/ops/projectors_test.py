from itertools import permutations
import math

import numpy as np
import pytest

import cirq


def test_projector_qubit():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.Projector({q0: [[1.0, 0.0]]})
    one_projector = cirq.Projector({q0: [[0.0, 1.0]]})

    np.testing.assert_allclose(zero_projector.matrix(),
                               [[1.0, 0.0], [0.0, 0.0]])

    np.testing.assert_allclose(one_projector.matrix(), [[0.0, 0.0], [0.0, 1.0]])


def test_projector_from_np_array():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.Projector({q0: np.array([[1.0, 0.0]])})
    np.testing.assert_allclose(zero_projector.matrix(),
                               [[1.0, 0.0], [0.0, 0.0]])


def test_projector_plus():
    q0 = cirq.NamedQubit('q0')

    plus_projector = cirq.Projector(
        {q0: [[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]})

    np.testing.assert_allclose(plus_projector.matrix(),
                               [[0.5, 0.5], [0.5, 0.5]])


def test_projector_overcomplete_basis():
    q0 = cirq.NamedQubit('q0')

    with pytest.raises(ValueError,
                       match="Vectors in basis must be linearly independent"):
        cirq.Projector({
            q0: [[1.0, 0.0], [0.0, 1.0],
                 [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]
        })


def test_projector_non_orthonormal_basis():
    q0 = cirq.NamedQubit('q0')

    cirq.Projector({q0: [[1.0, 0.0]]}, enforce_orthonormal_basis=True)
    cirq.Projector({q0: [[1.0, 0.0], [0.0, 1.0]]},
                   enforce_orthonormal_basis=True)
    cirq.Projector(
        {
            q0: [[1.0j / math.sqrt(2), 1.0 / math.sqrt(2)],
                 [1.0 / math.sqrt(2), 1.0j / math.sqrt(2)]]
        },
        enforce_orthonormal_basis=True)

    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector(
            {q0: [[1.0, 0.0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]},
            enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector(
            {
                q0: [[1.0j / math.sqrt(2), 1.0 / math.sqrt(2)],
                     [1.0 / math.sqrt(2), -1.0j / math.sqrt(2)]]
            },
            enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="State_vector is not normalized"):
        cirq.Projector({q0: [[2.0, 0.0]]}, enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector(
            {q0: [[1.0, 0.0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]},
            enforce_orthonormal_basis=True)


def test_projector_missing_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    proj = cirq.Projector({q0: [[1.0, 0.0]]})

    np.testing.assert_allclose(proj.matrix(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(proj.matrix([q0]), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(proj.matrix([q1]), [[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(proj.matrix([q0, q1]),
                               np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]),
                               np.diag([1.0, 0.0, 1.0, 0.0]))


def test_equality():
    q0 = cirq.NamedQubit('q0')

    obj1 = cirq.Projector({q0: [[1.0, 0.0]]})
    obj2 = cirq.Projector({q0: [[0.0, 1.0]]})

    assert obj1 == obj1
    assert obj1 != obj2
    assert hash(obj1) == hash(obj1)
    assert hash(obj1) != hash(obj2)


def test_projector_dim2_qubit():
    q0 = cirq.NamedQubit('q0')

    dim2_projector = cirq.Projector({q0: [[1.0, 0.0], [0.0, 1.0]]})
    not_colinear_projector = cirq.Projector(
        {q0: [[1.0, 0.0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]]})
    complex_projector = cirq.Projector(
        {q0: [[1.0j, 0.0], [1. / math.sqrt(2), 1.0 / math.sqrt(2)]]})

    np.testing.assert_allclose(dim2_projector.matrix(),
                               [[1.0, 0.0], [0.0, 1.0]],
                               atol=1e-6)

    np.testing.assert_allclose(not_colinear_projector.matrix(),
                               [[1.0, 0.0], [0.0, 1.0]],
                               atol=1e-6)

    np.testing.assert_allclose(complex_projector.matrix(),
                               [[1.0, 0.0], [0.0, 1.0]],
                               atol=1e-6)


def test_projector_qutrit():
    q0, = cirq.LineQid.range(1, dimension=3)

    zero_projector = cirq.Projector({q0: [[1.0, 0.0, 0.0]]})
    one_projector = cirq.Projector({q0: [[0.0, 1.0, 0.0]]})
    two_projector = cirq.Projector({q0: [[0.0, 0.0, 1.0]]})

    np.testing.assert_allclose(
        zero_projector.matrix(),
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    np.testing.assert_allclose(
        one_projector.matrix(),
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    np.testing.assert_allclose(
        two_projector.matrix(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_get_values():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({q0: [[1.0, 0.0]]})

    assert len(d._projection_bases_()) == 1
    assert np.allclose(d._projection_bases_()[q0], [[1.0, 0.0]])


def test_repr():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({q0: [[1.0, 0.0]]})

    assert d.__repr__() == (
        "cirq.Projector(projection_bases={" +
        "cirq.NamedQubit('q0'): array([[1.+0.j, 0.+0.j]], dtype=complex64)})")


def test_consistency_with_existing():
    a, b = cirq.LineQubit.range(2)
    mx = (cirq.KET_IMAG(a) * cirq.KET_IMAG(b)).projector()
    ii_proj = cirq.Projector({(a, b): [[.5, .5j, .5j, -.5]]})
    np.testing.assert_allclose(mx, ii_proj.matrix())


def test_expectation_from_state_vector_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({})

    np.testing.assert_allclose(
        d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)


def test_expectation_from_state_vector_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({q0: [[1.0, 0.0]]})

    np.testing.assert_allclose(
        d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)
    np.testing.assert_allclose(
        d.expectation_from_state_vector(np.array([0.0, 1.0]), {q0: 0}), 0.0)


def test_expectation_from_state_vector_basis_states_three_qubits():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    d = cirq.Projector({q0: [[1.0, 0.0]], q1: [[0.0, 1.0]]})

    np.testing.assert_allclose(
        d.expectation_from_state_vector(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), {
                q0: 0,
                q1: 1,
                q2: 2
            }), 0.0)

    np.testing.assert_allclose(
        d.expectation_from_state_vector(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), {
                q0: 0,
                q1: 2,
                q2: 1
            }), 1.0)


def test_expectation_higher_dims():
    q0 = cirq.NamedQid('q0', dimension=2)
    q1 = cirq.NamedQid('q1', dimension=3)
    q2 = cirq.NamedQid('q2', dimension=5)
    d = cirq.Projector({q2: [[0.0, 0.0, 0.0, 1.0, 0.0]], q1: [[0.0, 1.0, 0.0]]})

    phis = [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]

    for perm in permutations([0, 1, 2]):
        inv_perm = [-1] * len(perm)
        for i, j in enumerate(perm):
            inv_perm[j] = i

        state_vector = np.kron(phis[perm[0]],
                               np.kron(phis[perm[1]], phis[perm[2]]))
        state = np.einsum('i,j->ij', state_vector, state_vector.T.conj())

        np.testing.assert_allclose(
            d.expectation_from_state_vector(state_vector, {
                q0: inv_perm[0],
                q1: inv_perm[1],
                q2: inv_perm[2]
            }), 1.0)

        np.testing.assert_allclose(
            d.expectation_from_density_matrix(state, {
                q0: inv_perm[0],
                q1: inv_perm[1],
                q2: inv_perm[2]
            }), 1.0)


def test_expectation_from_density_matrix_basis_states_empty():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]),
                                          {q0: 0}), 1.0)


def test_expectation_from_density_matrix_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.Projector({q0: [[1.0, 0.0]]})

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]),
                                          {q0: 0}), 1.0)
    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[0.0, 0.0], [0.0, 1.0]]),
                                          {q0: 0}), 0.0)
