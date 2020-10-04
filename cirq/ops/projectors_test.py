import numpy as np

import cirq


def test_projector_qubit():
    zero_projector = cirq.Projector(projector_id=0)
    one_projector = cirq.Projector(projector_id=1)

    np.testing.assert_allclose(cirq.channel(zero_projector),
                               ([[1.0, 0.0], [0.0, 0.0]],))

    np.testing.assert_allclose(cirq.channel(one_projector),
                               ([[0.0, 0.0], [0.0, 1.0]],))


def test_projector_qutrit():
    zero_projector = cirq.Projector(projector_id=0, qid_shape=3)
    one_projector = cirq.Projector(projector_id=1, qid_shape=3)
    two_projector = cirq.Projector(projector_id=2, qid_shape=3)

    np.testing.assert_allclose(
        cirq.channel(zero_projector),
        ([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],))

    np.testing.assert_allclose(
        cirq.channel(one_projector),
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],))

    np.testing.assert_allclose(
        cirq.channel(two_projector),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],))
