import numpy as np
import pytest

import cirq


def test_projector_matrix():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})
    coeff_projector = cirq.ProjectorString({q0: 0}, 1.23 + 4.56j)

    np.testing.assert_allclose(zero_projector.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        coeff_projector.matrix().toarray(), [[1.23 + 4.56j, 0.0], [0.0, 0.0]]
    )


def test_projector_repr():
    q0 = cirq.NamedQubit('q0')

    assert (
        repr(cirq.ProjectorString({q0: 0}))
        == "cirq.ProjectorString(projector_dict={cirq.NamedQubit('q0'): 0},coefficient=(1+0j))"
    )


def test_projector_from_np_array():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    np.testing.assert_allclose(zero_projector.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])


def test_projector_matrix_missing_qid():
    q0, q1 = cirq.LineQubit.range(2)
    proj = cirq.ProjectorString({q0: 0})
    proj_with_coefficient = cirq.ProjectorString({q0: 0}, 1.23 + 4.56j)

    np.testing.assert_allclose(proj.matrix().toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q0]).toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1]).toarray(), np.diag([1.0, 1.0]))

    np.testing.assert_allclose(proj.matrix([q0, q1]).toarray(), np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]).toarray(), np.diag([1.0, 0.0, 1.0, 0.0]))

    np.testing.assert_allclose(
        proj_with_coefficient.matrix([q1, q0]).toarray(),
        np.diag([1.23 + 4.56j, 0.0, 1.23 + 4.56j, 0.0]),
    )


def test_equality():
    q0 = cirq.NamedQubit('q0')

    obj1a = cirq.ProjectorString({q0: 0})
    obj1b = cirq.ProjectorString({q0: 0})
    obj2 = cirq.ProjectorString({q0: 1})
    obj3 = cirq.ProjectorString({q0: 1}, coefficient=0.20160913)

    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(obj1a, obj1b)
    eq.add_equality_group(obj2)
    eq.add_equality_group(obj3)


def test_get_values():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0}, 1.23 + 4.56j)

    assert len(d.projector_dict) == 1
    assert d.projector_dict[q0] == 0
    assert d.coefficient == 1.23 + 4.56j


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
    d_1qbit = cirq.ProjectorString({q1: 1})
    d_2qbits = cirq.ProjectorString({q0: 0, q1: 1})

    state_vector = cirq.testing.random_superposition(8)

    # If the mapping of state_vector is {q0: 0, q1: 1, q2: 2}, then the coefficients are:
    # 0: (q0, q1, q2) = (0, 0, 0)
    # 1: (q0, q1, q2) = (0, 0, 1)
    # 2: (q0, q1, q2) = (0, 1, 0) -> Projected on
    # 3: (q0, q1, q2) = (0, 1, 1) -> Projected on
    # 4: (q0, q1, q2) = (1, 0, 0)
    # 5: (q0, q1, q2) = (1, 0, 1)
    # 6: (q0, q1, q2) = (1, 1, 0)
    # 7: (q0, q1, q2) = (1, 1, 1)
    np.testing.assert_allclose(
        d_2qbits.expectation_from_state_vector(state_vector, {q0: 0, q1: 1, q2: 2}),
        sum(abs(state_vector[i]) ** 2 for i in [2, 3]),
    )

    # Same as above except it's only for q1=1, which happens for indices 2, 3, 6, and 7:
    np.testing.assert_allclose(
        d_1qbit.expectation_from_state_vector(state_vector, {q0: 0, q1: 1, q2: 2}),
        sum(abs(state_vector[i]) ** 2 for i in [2, 3, 6, 7]),
    )

    # Here we have a different mapping, but the idea is the same.
    # 0: (q0 ,q2, q1) = (0, 0, 0)
    # 1: (q0, q2, q1) = (0, 0, 1) -> Projected on
    # 2: (q0, q2, q1) = (0, 1, 0)
    # 3: (q0, q2, q1) = (0, 1, 1) -> Projected on
    # 4: (q0, q2, q1) = (1, 0, 0)
    # 5: (q0, q2, q1) = (1, 0, 1)
    # 6: (q0, q2, q1) = (1, 1, 0)
    # 7: (q0, q2, q1) = (1, 1, 1)
    np.testing.assert_allclose(
        d_2qbits.expectation_from_state_vector(state_vector, {q0: 0, q1: 2, q2: 1}),
        sum(abs(state_vector[i]) ** 2 for i in [1, 3]),
    )

    # Same as above except it's only for q1=1, which happens for indices 1, 3, 5, and 7:
    np.testing.assert_allclose(
        d_1qbit.expectation_from_state_vector(state_vector, {q0: 0, q1: 2, q2: 1}),
        sum(abs(state_vector[i]) ** 2 for i in [1, 3, 5, 7]),
    )


def test_expectation_from_density_matrix_three_qubits():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    d_1qbit = cirq.ProjectorString({q1: 1})
    d_2qbits = cirq.ProjectorString({q0: 0, q1: 1})

    state = cirq.testing.random_density_matrix(8)

    # If the mapping of state is {q0: 0, q1: 1, q2: 2}, then the coefficients are:
    # 0: (q0, q1, q2) = (0, 0, 0)
    # 1: (q0, q1, q2) = (0, 0, 1)
    # 2: (q0, q1, q2) = (0, 1, 0) -> Projected on
    # 3: (q0, q1, q2) = (0, 1, 1) -> Projected on
    # 4: (q0, q1, q2) = (1, 0, 0)
    # 5: (q0, q1, q2) = (1, 0, 1)
    # 6: (q0, q1, q2) = (1, 1, 0)
    # 7: (q0, q1, q2) = (1, 1, 1)
    np.testing.assert_allclose(
        d_2qbits.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}),
        sum(state[i][i].real for i in [2, 3]),
    )

    # Same as above except it's only for q1=1, which happens for indices 2, 3, 6, and 7:
    np.testing.assert_allclose(
        d_1qbit.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}),
        sum(state[i][i].real for i in [2, 3, 6, 7]),
    )

    # Here we have a different mapping, but the idea is the same.
    # 0: (q0 ,q2, q1) = (0, 0, 0)
    # 1: (q0, q2, q1) = (0, 0, 1) -> Projected on
    # 2: (q0, q2, q1) = (0, 1, 0)
    # 3: (q0, q2, q1) = (0, 1, 1) -> Projected on
    # 4: (q0, q2, q1) = (1, 0, 0)
    # 5: (q0, q2, q1) = (1, 0, 1)
    # 6: (q0, q2, q1) = (1, 1, 0)
    # 7: (q0, q2, q1) = (1, 1, 1)
    np.testing.assert_allclose(
        d_2qbits.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}),
        sum(state[i][i].real for i in [1, 3]),
    )

    # Same as above except it's only for q1=1, which happens for indices 1, 3, 5, and 7:
    np.testing.assert_allclose(
        d_1qbit.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}),
        sum(state[i][i].real for i in [1, 3, 5, 7]),
    )


def test_consistency_state_vector_and_density_matrix():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')

    state_vector = cirq.testing.random_superposition(8)
    state = np.einsum('i,j->ij', state_vector, np.conj(state_vector))

    for proj_qubit in q0, q1, q2:
        for proj_idx in [0, 1]:
            d = cirq.ProjectorString({proj_qubit: proj_idx})

            np.testing.assert_allclose(
                d.expectation_from_state_vector(state_vector, {q0: 0, q1: 1, q2: 2}),
                d.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}),
            )


def test_expectation_higher_dims():
    qubit = cirq.NamedQid('q0', dimension=2)
    qutrit = cirq.NamedQid('q1', dimension=3)

    with pytest.raises(ValueError, match="Only qubits are supported"):
        cirq.ProjectorString({qutrit: 0})

    d = cirq.ProjectorString({qubit: 0})
    with pytest.raises(ValueError, match="Only qubits are supported"):
        _ = (d.expectation_from_state_vector(np.zeros(2 * 3), {qubit: 0, qutrit: 0}),)


def test_expectation_with_coefficient():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0}, coefficient=(0.6 + 0.4j))

    np.testing.assert_allclose(
        d.expectation_from_state_vector(np.array([[1.0, 0.0]]), qid_map={q0: 0}), 0.6 + 0.4j
    )

    np.testing.assert_allclose(
        d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 0.6 + 0.4j
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
