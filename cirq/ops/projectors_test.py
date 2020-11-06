import math
import re

import numpy as np
import pytest

import cirq


def test_projector1d_circuit_diagram():
    q = cirq.NamedQubit('q')
    projector = cirq.Projector([[1.0, 0.0]])
    cirq.testing.assert_has_diagram(cirq.Circuit(projector(q)),
                                    "q: ───Proj([[1.0, 0.0]])───",
                                    precision=None)
    cirq.testing.assert_has_diagram(cirq.Circuit(projector(q)),
                                    "q: ───Proj([[1.0, 0.0]])───",
                                    precision=2)


def test_projector2d_circuit_diagram():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    projector = cirq.Projector([[1.0, 0.0, 0.0, 0.0]], qid_shape=(
        2,
        2,
    ))
    cirq.testing.assert_has_diagram(cirq.Circuit(projector(
        q1, q2)), ("q1: ───Proj([[1.0, 0.0, 0.0, 0.0]])───\n" + "       │\n" +
                   "q2: ───Proj───────────────────────────"))


def test_projector_qubit():
    zero_projector = cirq.Projector([[1.0, 0.0]])
    one_projector = cirq.Projector([[0.0, 1.0]])

    np.testing.assert_allclose(cirq.channel(zero_projector),
                               ([[1.0, 0.0], [0.0, 0.0]],))

    np.testing.assert_allclose(cirq.channel(one_projector),
                               ([[0.0, 0.0], [0.0, 1.0]],))


def test_projector_from_np_array():
    zero_projector = cirq.Projector(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(cirq.channel(zero_projector),
                               ([[1.0, 0.0], [0.0, 0.0]],))


def test_projector_bad_rank():
    with pytest.raises(ValueError,
                       match="The input projection_basis must be a 2D array"):
        cirq.Projector(np.array([1.0, 0.0]))


def test_projector_plus():
    plus_projector = cirq.Projector([[1.0, 1.0]])

    np.testing.assert_allclose(cirq.channel(plus_projector),
                               ([[0.5, 0.5], [0.5, 0.5]],))


def test_projector_overcomplete_basis():
    with pytest.raises(ValueError,
                       match="Vectors in basis must be linearly independent"):
        cirq.Projector([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_projector_non_orthonormal_basis():
    cirq.Projector([[1.0, 0.0]], enforce_orthonormal_basis=True)
    cirq.Projector([[1.0, 0.0], [0.0, 1.0]], enforce_orthonormal_basis=True)
    cirq.Projector([[1.0j / math.sqrt(2), 1.0 / math.sqrt(2)],
                    [1.0 / math.sqrt(2), 1.0j / math.sqrt(2)]],
                   enforce_orthonormal_basis=True)

    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector([[1.0, 0.0], [1.0, 1.0]], enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector([[1.0j / math.sqrt(2), 1.0 / math.sqrt(2)],
                        [1.0 / math.sqrt(2), -1.0j / math.sqrt(2)]],
                       enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector([[2.0, 0.0]], enforce_orthonormal_basis=True)
    with pytest.raises(ValueError, match="The basis must be orthonormal"):
        cirq.Projector([[1.0, 0.0], [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]],
                       enforce_orthonormal_basis=True)


def test_equality():
    obj1 = cirq.Projector([[1.0, 0.0]])
    obj2 = cirq.Projector([[0.0, 1.0]])

    assert obj1 == obj1
    assert obj1 != obj2
    assert hash(obj1) == hash(obj1)
    assert hash(obj1) != hash(obj2)


def test_projector_dim2_qubit():
    dim2_projector = cirq.Projector([[1.0, 0.0], [0.0, 1.0]])
    not_colinear_projector = cirq.Projector([[1.0, 0.0], [1.0, 1.0]])
    complex_projector = cirq.Projector([[1.0j, 0.0], [1.0, 1.0]])

    np.testing.assert_allclose(cirq.channel(dim2_projector),
                               ([[1.0, 0.0], [0.0, 1.0]],),
                               atol=1e-12)

    np.testing.assert_allclose(cirq.channel(not_colinear_projector),
                               ([[1.0, 0.0], [0.0, 1.0]],),
                               atol=1e-12)

    np.testing.assert_allclose(cirq.channel(complex_projector),
                               ([[1.0, 0.0], [0.0, 1.0]],),
                               atol=1e-12)


def test_projector_qutrit():
    zero_projector = cirq.Projector([[1.0, 0.0, 0.0]], qid_shape=[3])
    one_projector = cirq.Projector([[0.0, 1.0, 0.0]], qid_shape=[3])
    two_projector = cirq.Projector([[0.0, 0.0, 1.0]], qid_shape=[3])

    np.testing.assert_allclose(
        cirq.channel(zero_projector),
        ([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],))

    np.testing.assert_allclose(
        cirq.channel(one_projector),
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],))

    np.testing.assert_allclose(
        cirq.channel(two_projector),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],))


def test_bad_constructors():
    with pytest.raises(
            ValueError,
            match=re.escape("Invalid shape (1, 2) for qid_shape [2, 2]")):
        cirq.Projector([[1.0, 0.0]], qid_shape=[2, 2])


def test_get_values():
    d = cirq.Projector([[1.0, 0.0]])

    np.testing.assert_allclose(d._projection_basis_(), [[1.0, 0.0]])
    assert d._qid_shape_() == (2,)
    assert not d._has_unitary_()
    assert not d._is_parameterized_()
    assert d._has_channel_()


def test_repr():
    d = cirq.Projector([[1.0, 0.0]])

    assert d.__repr__(
    ) == "cirq.Projector(projection_basis=[[1.0, 0.0]]),qid_shape=(2,))"


def test_consistency_with_existing():
    a, b = cirq.LineQubit.range(2)
    mx = (cirq.KET_IMAG(a) * cirq.KET_IMAG(b)).projector()
    ii_proj = cirq.Projector([[.5, .5j, .5j, -.5]], qid_shape=(
        2,
        2,
    ))
    np.testing.assert_allclose(mx, cirq.channel(ii_proj)[0])
