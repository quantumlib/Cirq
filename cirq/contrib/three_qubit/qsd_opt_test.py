from random import random

import numpy as np

import pytest
from numpy.testing import assert_almost_equal

import cirq
from cirq.contrib.three_qubit.qsd_opt import _multiplexed_angles, \
    _cs_to_circuit


@pytest.mark.parametrize("theta", [
    np.array([0.5, 0.6, 0.7, 0.8]),
    np.array([0., 0., np.pi / 2, np.pi / 2]),
    np.array([0., 0., 0., 0.]),
    np.repeat(np.pi / 4, repeats=4),
])
def test_cs_to_circuit(theta):
    a, b, c = cirq.LineQubit.range(3)
    CS = _theta_to_CS(theta)
    circuit_CS = _cs_to_circuit(a, b, c, theta)
    rightmost_CZ = cirq.Circuit(cirq.CZ(c, a)).unitary(
        qubits_that_should_be_present=[a, b, c])
    assert_almost_equal(
        rightmost_CZ @ circuit_CS.unitary(
            qubits_that_should_be_present=[a, b, c]), CS, 15)


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
