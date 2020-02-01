from random import random

import numpy as np

from cirq.contrib.three_qubit.qsd_opt import multiplexed_angles


def test_multiplexed_angles():
    theta = [random() * np.pi,
             random() * np.pi,
             random() * np.pi,
             random() * np.pi]

    angles = multiplexed_angles(theta)

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
