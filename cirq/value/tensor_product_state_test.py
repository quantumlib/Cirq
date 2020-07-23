import numpy as np
import pytest

import cirq


def test_name():
    names = [str(state) for state in cirq.PAULI_STATES]
    assert names == [
        '+X',
        '-X',
        '+Y',
        '-Y',
        '+Z',
        '-Z',
    ]


def test_repr():
    for o in cirq.PAULI_STATES:
        assert o == eval(repr(o))


def test_equality():
    assert cirq.KET_PLUS == cirq.KET_PLUS
    assert cirq.KET_PLUS != cirq.KET_MINUS
    assert cirq.KET_PLUS != cirq.KET_ZERO

    assert hash(cirq.KET_PLUS) == hash(cirq.KET_PLUS)


def test_stabilized():
    for state in cirq.PAULI_STATES:
        val, gate = state.stabilized_by()
        matrix = cirq.unitary(gate)
        vec = state.state_vector()

        np.testing.assert_allclose(matrix @ vec, val * vec)


def test_projector():
    np.testing.assert_equal(cirq.KET_ZERO.projector(), [[1, 0], [0, 0]])
    np.testing.assert_equal(cirq.KET_ONE.projector(), [[0, 0], [0, 1]])
    np.testing.assert_allclose(cirq.KET_PLUS.projector(),
                               np.array([[1, 1], [1, 1]]) / 2)
    np.testing.assert_allclose(cirq.KET_MINUS.projector(),
                               np.array([[1, -1], [-1, 1]]) / 2)


def test_oneq_state():
    q0, q1 = cirq.LineQubit.range(2)
    st0 = cirq.KET_PLUS(q0)
    assert str(st0) == '+X(0)'

    st1 = cirq.KET_PLUS(q1)
    assert st0 != st1

    assert st0 == cirq.KET_PLUS.on(q0)


def test_tensor_product_state():
    q0, q1, q2 = cirq.LineQubit.range(3)

    tps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)
    assert str(tps) == "+X(0) * +X(1)"

    tps *= cirq.KET_ONE(q2)
    assert str(tps) == "+X(0) * +X(1) * -Z(2)"

    with pytest.raises(ValueError) as e:
        # Re-use q2
        tps *= cirq.KET_PLUS(q2)
    assert e.match(r'.*both contain factors for these qubits: '
                   r'\[cirq.LineQubit\(2\)\]')


def test_tensor_product_state_2():
    q0, q1, q2 = cirq.LineQubit.range(3)

    with pytest.raises(ValueError):
        # No coefficient
        tps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * -1
    with pytest.raises(ValueError):
        # Not a state
        tps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO


def test_tensor_product_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    tps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO(q2)
    assert tps.qubits == [q0, q1, q2]
    assert tps[q0] == cirq.KET_PLUS


def test_tensor_product_iter():
    q0, q1, q2 = cirq.LineQubit.range(3)
    tps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO(q2)

    should_be = [
        (q0, cirq.KET_PLUS),
        (q1, cirq.KET_PLUS),
        (q2, cirq.KET_ZERO),
    ]
    assert list(tps) == should_be
    assert len(tps) == 3


def test_tensor_product_state_equality():
    q0, q1, q2 = cirq.LineQubit.range(3)

    assert cirq.KET_PLUS(q0) == cirq.KET_PLUS(q0)
    assert cirq.KET_PLUS(q0) != cirq.KET_PLUS(q1)
    assert cirq.KET_PLUS(q0) != cirq.KET_MINUS(q0)

    assert cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1) == cirq.KET_PLUS(
        q0) * cirq.KET_MINUS(q1)
    assert cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1) != cirq.KET_PLUS(
        q0) * cirq.KET_MINUS(q2)

    assert hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)) == hash(
        cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1))
    assert hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)) != hash(
        cirq.KET_PLUS(q0) * cirq.KET_MINUS(q2))
    assert cirq.KET_PLUS(q0) != '+X(0)'


def test_tp_state_vector():
    q0, q1 = cirq.LineQubit.range(2)
    s00 = cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)
    np.testing.assert_equal(s00.state_vector(), [1, 0, 0, 0])
    np.testing.assert_equal(s00.state_vector(qubit_order=(q1, q0)),
                            [1, 0, 0, 0])

    s01 = cirq.KET_ZERO(q0) * cirq.KET_ONE(q1)
    np.testing.assert_equal(s01.state_vector(), [0, 1, 0, 0])
    np.testing.assert_equal(s01.state_vector(qubit_order=(q1, q0)),
                            [0, 0, 1, 0])


def test_tp_initial_state():
    q0, q1, q2 = cirq.LineQubit.range(3)
    psi1 = cirq.final_state_vector(
        cirq.Circuit([cirq.I.on_each(q0, q1),
                      cirq.X(q1)]))

    s01 = cirq.KET_ZERO(q0) * cirq.KET_ONE(q1)
    psi2 = cirq.final_state_vector(cirq.Circuit([cirq.I.on_each(q0, q1)]),
                                   initial_state=s01)

    np.testing.assert_allclose(psi1, psi2)


def test_tp_projector():
    q0, q1 = cirq.LineQubit.range(2)
    p00 = (cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)))
    np.testing.assert_allclose(rho, p00)

    p01 = (cirq.KET_ZERO(q0) * cirq.KET_ONE(q1)).projector()
    rho = cirq.final_density_matrix(
        cirq.Circuit([cirq.I.on_each(q0, q1),
                      cirq.X(q1)]))
    np.testing.assert_allclose(rho, p01)

    ppp = (cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit([
        cirq.H.on_each(q0, q1),
    ]))
    np.testing.assert_allclose(rho, ppp, atol=1e-7)

    ppm = (cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)).projector()
    rho = cirq.final_density_matrix(
        cirq.Circuit([cirq.H.on_each(q0, q1),
                      cirq.Z(q1)]))
    np.testing.assert_allclose(rho, ppm, atol=1e-7)
