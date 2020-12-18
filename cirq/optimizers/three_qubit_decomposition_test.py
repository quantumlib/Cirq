# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import random

import numpy as np

import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag

import cirq
from cirq.optimizers.three_qubit_decomposition import (
    _multiplexed_angles,
    _cs_to_ops,
    _middle_multiplexor_to_ops,
    _two_qubit_multiplexor_to_ops,
)


@pytest.mark.parametrize(
    "u",
    [
        cirq.testing.random_unitary(8),
        np.eye(8),
        cirq.ControlledGate(cirq.ISWAP)._unitary_(),
        cirq.CCX._unitary_(),
    ],
)
def test_three_qubit_matrix_to_operations(u):
    a, b, c = cirq.LineQubit.range(3)
    operations = cirq.three_qubit_matrix_to_operations(a, b, c, u)
    final_circuit = cirq.Circuit(operations)
    final_unitary = final_circuit.unitary(qubits_that_should_be_present=[a, b, c])
    cirq.testing.assert_allclose_up_to_global_phase(u, final_unitary, atol=1e-9)
    num_two_qubit_gates = len(
        [
            op
            for op in list(final_circuit.all_operations())
            if isinstance(op.gate, (cirq.CZPowGate, cirq.CNotPowGate))
        ]
    )
    assert num_two_qubit_gates <= 20, f"expected at most 20 CZ/CNOTs got {num_two_qubit_gates}"


def test_three_qubit_matrix_to_operations_errors():
    a, b, c = cirq.LineQubit.range(3)
    with pytest.raises(ValueError, match="(8,8)"):
        cirq.three_qubit_matrix_to_operations(a, b, c, np.eye(2))
    with pytest.raises(ValueError, match="not unitary"):
        cirq.three_qubit_matrix_to_operations(a, b, c, cirq.unitary(cirq.CCX) * 2)


@pytest.mark.parametrize(
    ["theta", "num_czs"],
    [
        (np.array([0.5, 0.6, 0.7, 0.8]), 4),
        (np.array([0.0, 0.0, np.pi / 2, np.pi / 2]), 2),
        (np.zeros(4), 0),
        (np.repeat(np.pi / 4, repeats=4), 0),
        (np.array([0.5 * np.pi, -0.5 * np.pi, 0.7 * np.pi, -0.7 * np.pi]), 4),
        (np.array([0.3, -0.3, 0.3, -0.3]), 2),
        (np.array([0.3, 0.3, -0.3, -0.3]), 2),
    ],
)
def test_cs_to_ops(theta, num_czs):
    a, b, c = cirq.LineQubit.range(3)
    cs = _theta_to_cs(theta)
    circuit_cs = cirq.Circuit(_cs_to_ops(a, b, c, theta))

    assert_almost_equal(circuit_cs.unitary(qubits_that_should_be_present=[a, b, c]), cs, 10)

    assert (
        len([cz for cz in list(circuit_cs.all_operations()) if isinstance(cz.gate, cirq.CZPowGate)])
        == num_czs
    ), "expected {} CZs got \n {} \n {}".format(num_czs, circuit_cs, circuit_cs.unitary())


def _theta_to_cs(theta: np.ndarray) -> np.ndarray:
    """Returns the CS matrix from the cosine sine decomposition.

    Args:
        theta: the 4 angles that result from the CS decomposition
    Returns:
        the CS matrix
    """
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    return np.block([[c, -s], [s, c]])


def test_multiplexed_angles():
    theta = [random() * np.pi, random() * np.pi, random() * np.pi, random() * np.pi]

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
    assert np.isclose(theta[0], (angles[0] + angles[1] + angles[2] + angles[3]))

    # |01> on the select qubits
    #
    # ---a(0)----a(1)--X--a(2)---a(3)-X
    #                  |              |
    # -----------------|--------------|
    #                  |              |
    # -----------------@--------------@
    assert np.isclose(theta[1], (angles[0] + angles[1] - angles[2] - angles[3]))

    # |10> on the select qubits
    #
    # ---a(0)-X---a(1)---a(2)-X--a(3)
    #         |               |
    # --------@---------------@------
    #
    # ---------------------------------
    assert np.isclose(theta[2], (angles[0] - angles[1] - angles[2] + angles[3]))

    # |11> on the select qubits
    #
    # ---a(0)-X---a(1)--X--a(2)-X--a(3)--X
    #         |         |       |        |
    # --------@---------|-------@--------|
    #                   |                |
    # ------------------@----------------@
    assert np.isclose(theta[3], (angles[0] - angles[1] + angles[2] - angles[3]))


@pytest.mark.parametrize(
    ["angles", "num_cnots"],
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
    ],
)
def test_middle_multiplexor(angles, num_cnots):
    a, b, c = cirq.LineQubit.range(3)
    eigvals = np.exp(np.array(angles) * np.pi * 1j)
    d = np.diag(np.sqrt(eigvals))
    mid = block_diag(d, d.conj().T)
    circuit_u1u2_mid = cirq.Circuit(_middle_multiplexor_to_ops(a, b, c, eigvals))
    np.testing.assert_almost_equal(
        mid, circuit_u1u2_mid.unitary(qubits_that_should_be_present=[a, b, c])
    )
    assert (
        len(
            [
                cnot
                for cnot in list(circuit_u1u2_mid.all_operations())
                if isinstance(cnot.gate, cirq.CNotPowGate)
            ]
        )
        == num_cnots
    ), "expected {} CNOTs got \n {} \n {}".format(
        num_cnots, circuit_u1u2_mid, circuit_u1u2_mid.unitary()
    )


@pytest.mark.parametrize("shift_left", [True, False])
def test_two_qubit_multiplexor_to_circuit(shift_left):
    a, b, c = cirq.LineQubit.range(3)
    u1 = cirq.testing.random_unitary(4)
    u2 = cirq.testing.random_unitary(4)
    d_ud, ud_ops = _two_qubit_multiplexor_to_ops(a, b, c, u1, u2, shift_left=shift_left)
    expected = block_diag(u1, u2)
    diagonal = np.kron(np.eye(2), d_ud) if d_ud is not None else np.eye(8)
    actual = cirq.Circuit(ud_ops).unitary(qubits_that_should_be_present=[a, b, c]) @ diagonal
    cirq.testing.assert_allclose_up_to_global_phase(expected, actual, atol=1e-8)
