import numpy as np
import pytest

import cirq


def test_projector_qid():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})

    np.testing.assert_allclose(zero_projector.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])


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

    np.testing.assert_allclose(proj.matrix().toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q0]).toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1]).toarray(), np.diag([1.0, 1.0]))

    np.testing.assert_allclose(proj.matrix([q0, q1]).toarray(), np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]).toarray(), np.diag([1.0, 0.0, 1.0, 0.0]))


def test_equality():
    q0 = cirq.NamedQubit('q0')

    obj1 = cirq.ProjectorString({q0: 0})
    obj2 = cirq.ProjectorString({q0: 1})

    assert obj1 == obj1
    assert obj1 != obj2
    assert hash(obj1) == hash(obj1)
    assert hash(obj1) != hash(obj2)


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
    d = cirq.ProjectorString({q0: 0, q1: 1})

    wf = cirq.testing.random_superposition(8)

    # If the mapping of wf is {q0: 0, q1: 1, q2: 2}, then the coefficients are:
    # 0: (q0, q1, q2) = (0, 0, 0)
    # 1: (q0, q1, q2) = (0, 0, 1)
    # 2: (q0, q1, q2) = (0, 1, 0) -> Projected on
    # 3: (q0, q1, q2) = (0, 1, 1) -> Projected on
    # 4: (q0, q1, q2) = (1, 0, 0)
    # 5: (q0, q1, q2) = (1, 0, 1)
    # 6: (q0, q1, q2) = (1, 1, 0)
    # 7: (q0, q1, q2) = (1, 1, 1)
    np.testing.assert_allclose(
        d.expectation_from_state_vector(wf, {q0: 0, q1: 1, q2: 2}),
        abs(wf[2]) ** 2 + abs(wf[3]) ** 2,
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
        d.expectation_from_state_vector(wf, {q0: 0, q1: 2, q2: 1}),
        abs(wf[1]) ** 2 + abs(wf[3]) ** 2,
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


def test_projector_sum_expectations():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    one_projector = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 1}))

    proj_sum = 0.6 * zero_projector + 0.4 * one_projector
    np.testing.assert_allclose(proj_sum.matrix().toarray(), [[0.6, 0.0], [0.0, 0.4]])
    np.testing.assert_allclose(
        proj_sum.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 0.6
    )
    np.testing.assert_allclose(
        proj_sum.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 0.6
    )


def test_projector_sum_operations():
    q0 = cirq.NamedQubit('q0')

    zero_projector = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 0}))
    one_projector = cirq.ProjectorSum.from_projector_strings(cirq.ProjectorString({q0: 1}))

    simple_addition = zero_projector + one_projector
    np.testing.assert_allclose(simple_addition.matrix().toarray(), [[1.0, 0.0], [0.0, 1.0]])

    incrementation = zero_projector
    incrementation += one_projector
    np.testing.assert_allclose(incrementation.matrix().toarray(), [[1.0, 0.0], [0.0, 1.0]])

    weighted_sum = 0.6 * zero_projector + 0.4 * one_projector
    np.testing.assert_allclose(weighted_sum.matrix().toarray(), [[0.6, 0.0], [0.0, 0.4]])
