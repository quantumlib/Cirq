from random import random

import numpy as np

import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag

import cirq
from cirq.contrib.three_qubit.qsd_opt import _multiplexed_angles, \
    _cs_to_circuit, _middle_multiplexor, \
    _two_qubit_matrix_to_diagonal_and_circuit, _is_three_cnot_two_qubit_unitary


@pytest.mark.parametrize(["theta", "num_czs"], [
    (np.array([0.5, 0.6, 0.7, 0.8]), 4),
    (np.array([0., 0., np.pi / 2, np.pi / 2]), 4),
    (np.zeros(4), 0),
    (np.repeat(0.000015, repeats=4), 0),
    (np.repeat(np.pi / 4, repeats=4), 0),
])
def test_cs_to_circuit(theta, num_czs):
    a, b, c = cirq.LineQubit.range(3)
    CS = _theta_to_CS(theta)
    circuit_CS = _cs_to_circuit(a, b, c, theta)

    assert_almost_equal(circuit_CS.unitary(
        qubits_that_should_be_present=[a, b, c]), CS, 15)

    assert (len([cz for cz in list(circuit_CS.all_operations())
                 if isinstance(cz.gate, cirq.CZPowGate)]) == num_czs), \
        "expected {} CZs got \n {} \n {}".format(num_czs, circuit_CS,
                                                 circuit_CS.unitary())


def _theta_to_CS(theta):
    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    return np.vstack((np.hstack((C, -S)), np.hstack((S, C))))


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


def test_middle_multiplexor():
    a, b, c = cirq.LineQubit.range(3)
    eigvals = np.exp([-0.2312j, 0.2312j, 1.43j, -2.2322j])
    d = np.diag(np.sqrt(eigvals))
    mid = block_diag(d, d.conj().T)
    circuit_u1u2_mid = _middle_multiplexor(a, b, c, eigvals)
    np.testing.assert_almost_equal(mid, circuit_u1u2_mid.unitary(
        qubits_that_should_be_present=[a, b, c]))


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
    return cirq.Circuit([one_cz() for _ in range(num_cnots)])
