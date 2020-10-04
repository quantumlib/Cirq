import re

import numpy as np
import pytest

import cirq


def test_projector_qubit():
    zero_projector = cirq.Projector(projector_id=0)
    one_projector = cirq.Projector(projector_id=1)

    np.testing.assert_allclose(cirq.channel(zero_projector),
                               ([[1.0, 0.0], [0.0, 0.0]],))

    np.testing.assert_allclose(cirq.channel(one_projector),
                               ([[0.0, 0.0], [0.0, 1.0]],))


def test_projector_qutrit():
    zero_projector = cirq.Projector(projector_id=0, qid_shape=[3])
    one_projector = cirq.Projector(projector_id=1, qid_shape=[3])
    two_projector = cirq.Projector(projector_id=2, qid_shape=[3])

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
    with pytest.raises(ValueError, match="qid_shape must have a single entry."):
        cirq.Projector(projector_id=0, qid_shape=[2, 2])

    with pytest.raises(
            ValueError,
            match=re.escape("projector_id=2 must be less than qid_shape[0]=2")):
        cirq.Projector(projector_id=2, qid_shape=[2])


def test_get_values():
    d = cirq.Projector(projector_id=0)

    assert d._projector_id_() == 0
    assert d._qid_shape_() == (2,)
    assert not d._has_unitary_()
    assert not d._is_parameterized_()
    assert d._has_channel_()


def test_repr():
    d = cirq.Projector(projector_id=0)

    assert d.__repr__() == "cirq.Projector(projector_id=0,qid_shape=(2,))"
