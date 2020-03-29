from random import random

import numpy as np

import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag

import cirq
from cirq.contrib.three_qubit.qsd_opt import _multiplexed_angles, \
    _cs_to_ops, _middle_multiplexor_to_ops, \
    _two_qubit_matrix_to_diagonal_and_circuit, _is_three_cnot_two_qubit_unitary, \
    _to_special, _extract_right_diag, _gamma


@pytest.mark.parametrize(["theta", "num_czs"], [
    (np.array([0.5, 0.6, 0.7, 0.8]), 4),
    (np.array([0., 0., np.pi / 2, np.pi / 2]), 2),
    (np.zeros(4), 0),
    (np.repeat(np.pi / 4, repeats=4), 0),
    (np.array([0.5 * np.pi, -0.5 * np.pi, 0.7 * np.pi, -0.7 * np.pi]), 4),
    (np.array([0.3, -0.3, 0.3, -0.3]), 2),
    (np.array([0.3, 0.3, -0.3, -0.3]), 2),
])
def test_cs_to_ops(theta, num_czs):
    a, b, c = cirq.LineQubit.range(3)
    CS = _theta_to_CS(theta)
    circuit_CS = cirq.Circuit(_cs_to_ops(a, b, c, theta))

    assert_almost_equal(circuit_CS.unitary(
        qubits_that_should_be_present=[a, b, c]), CS, 10)

    assert (len([cz for cz in list(circuit_CS.all_operations())
                 if isinstance(cz.gate, cirq.CZPowGate)]) == num_czs), \
        "expected {} CZs got \n {} \n {}".format(num_czs, circuit_CS,
                                                 circuit_CS.unitary())


def _theta_to_CS(theta):
    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    return np.block([[C, -S],
                     [S, C]])


def test_multiplexed_angles():
    theta = [random() * np.pi,
             random() * np.pi,
             random() * np.pi,
             random() * np.pi]

    angles = _multiplexed_angles(theta)

    # assuming the following structure
    #
    # ---a(0)-X---a(1)--X--a(2)-X--a(3)--X
    #         |         |       |        |
    # --------@---------|-------@--------|
    #                   |                |
    # ------------------@----------------@

    # |00> on the select qubits
    #
    # ---a(0)----a(1)----a(2)---a(3)---
    #
    # ---------------------------------
    #
    # ---------------------------------
    assert np.isclose(theta[0],
                      (angles[0] + angles[1] + angles[2] + angles[3]))

    # |01> on the select qubits
    #
    # ---a(0)----a(1)--X--a(2)---a(3)-X
    #                  |              |
    # -----------------|--------------|
    #                  |              |
    # -----------------@--------------@
    assert np.isclose(theta[1],
                      (angles[0] + angles[1] - angles[2] - angles[3]))

    # |10> on the select qubits
    #
    # ---a(0)-X---a(1)---a(2)-X--a(3)
    #         |               |
    # --------@---------------@------
    #
    # ---------------------------------
    assert np.isclose(theta[2],
                      (angles[0] - angles[1] - angles[2] + angles[3]))

    # |11> on the select qubits
    #
    # ---a(0)-X---a(1)--X--a(2)-X--a(3)--X
    #         |         |       |        |
    # --------@---------|-------@--------|
    #                   |                |
    # ------------------@----------------@
    assert np.isclose(theta[3],
                      (angles[0] - angles[1] + angles[2] - angles[3]))


@pytest.mark.parametrize(["angles", "num_cnots"],
                         [
                             [([-0.2312, 0.2312, 1.43, -2.2322]), 4],
                             [([0, 0, 0, 0]), 0],
                             [([0.3, 0.3, 0.3, 0.3]), 0],
                             [([0.3, -0.3, 0.3, -0.3]), 2],
                             [([0.3, 0.3, -0.3, -0.3]), 2],
                             [([-0.3, 0.3, 0.3, -0.3]), 4],
                             [([-0.3, 0.3, -0.3, 0.3]), 2],
                             [([0.3, -0.3, -0.3, -0.3]), 4],
                             [([-0.3, 0.3, -0.3, -0.3]), 4],
                         ])
def test_middle_multiplexor(angles, num_cnots):
    a, b, c = cirq.LineQubit.range(3)
    eigvals = np.exp(np.array(angles) * np.pi * 1j)
    d = np.diag(np.sqrt(eigvals))
    mid = block_diag(d, d.conj().T)
    circuit_u1u2_mid = cirq.Circuit(
        _middle_multiplexor_to_ops(a, b, c, eigvals))
    np.testing.assert_almost_equal(mid, circuit_u1u2_mid.unitary(
        qubits_that_should_be_present=[a, b, c]))
    assert (len([cnot for cnot in list(circuit_u1u2_mid.all_operations())
                 if isinstance(cnot.gate, cirq.CNotPowGate)]) == num_cnots), \
        "expected {} CNOTs got \n {} \n {}".format(num_cnots, circuit_u1u2_mid,
                                                   circuit_u1u2_mid.unitary())


def test_two_qubit_matrix_to_diagonal_and_circuit():
    b, c = cirq.LineQubit.range(2)
    V = cirq.unitary(_two_qubit_circuit_with_cnots(3))
    circ, diagonal = _two_qubit_matrix_to_diagonal_and_circuit(V, b, c)
    cirq.testing.assert_allclose_up_to_global_phase(
        circ.unitary(qubits_that_should_be_present=[b, c]) @
        diagonal.conj().T, V, atol=1e-14)
    # TODO test diagonal branch
    # TODO test less than two qubit branch


def test_is_three_cnot_two_qubit_unitary():
    assert _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(3)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(2)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(1)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(
        np.eye(4))


def _two_qubit_circuit_with_cnots(num_cnots=3):
    a, b = cirq.LineQubit.range(2)
    random_one_qubit_gate = lambda: cirq.PhasedXPowGate(phase_exponent=random(),
                                                        exponent=random())
    one_cz = lambda: [random_one_qubit_gate().on(a),
                      random_one_qubit_gate().on(b),
                      cirq.CZ.on(a, b)]
    return cirq.Circuit([random_one_qubit_gate().on(a),
                         random_one_qubit_gate().on(b),
                         [one_cz() for _ in range(num_cnots)]])


def test_to_special():
    u = cirq.testing.random_unitary(4)
    su = _to_special(u)
    assert not cirq.is_special_unitary(u)
    assert cirq.is_special_unitary(su)


@pytest.mark.parametrize("U", [
    _two_qubit_circuit_with_cnots(3).unitary(),
    # an example where gamma(special(u))=I, so the denominator becomes 0
    np.array([[1,0,0,1],
              [0,0,1,0],
              [0,1,0,0],
              [0,0,0,1]], dtype=np.complex128)
])
def test_extract_right_diag(U):
    assert _num_two_qubit_gates_in_two_qubit_unitary(U) == 3
    diag = _extract_right_diag(U)
    assert cirq.is_diagonal(diag)
    assert _num_two_qubit_gates_in_two_qubit_unitary(U @ diag) == 2


def _num_two_qubit_gates_in_two_qubit_unitary(U):
    """
    See Proposition III.1, III.2, III.3 in Shende et al. “Recognizing Small-
    Circuit Structure in Two-Qubit Operators and Timing Hamiltonians to Compute
    Controlled-Not Gates”. In: Quant-Ph/0308045 (2003)'
    :param U: a two-qubit unitary
    :return: the number of two-qubit gates required to implement the unitary
    """
    assert np.shape(U) == (4, 4)
    poly = np.poly(_gamma(_to_special(U)))
    # characteristic polynomial = (x+1)^4 or (x-1)^4
    if np.allclose(poly, [1, 4, 6, 4, 1]) or np.allclose(poly,
                                                         [1, -4, 6, -4, 1]):
        return 0
    # characteristic polynomial = (x+i)^2 * (x-i)^2
    if np.allclose(poly, [1, 0, 2, 0, 1]):
        return 1
    # characteristic polynomial coefficients are all real
    if np.alltrue(np.isclose(0, np.imag(poly))):
        return 2
    return 3
