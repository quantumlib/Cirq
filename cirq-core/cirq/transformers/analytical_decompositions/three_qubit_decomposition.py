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

"""Utility methods for decomposing three-qubit unitaries."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import cirq
from cirq import ops, transformers as opt


def three_qubit_matrix_to_operations(
    q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, u: np.ndarray, atol: float = 1e-8
) -> List[ops.Operation]:
    """Returns operations for a 3 qubit unitary.

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        u: unitary matrix
        atol: A limit on the amount of absolute error introduced by the
            construction.

    Returns:
        The resulting operations will have only known two-qubit and one-qubit
        gates based operations, namely CZ, CNOT and rx, ry, PhasedXPow gates.

    Raises:
        ValueError: If the u matrix is non-unitary or not of shape (8,8).
        ImportError: If the decomposition cannot be done because the SciPy version is less than
            1.5.0 and so does not contain the required `cossin` method.
    """
    if np.shape(u) != (8, 8):
        raise ValueError(f"Expected unitary matrix with shape (8,8) got {np.shape(u)}")
    if not cirq.is_unitary(u, atol=atol):
        raise ValueError(f"Matrix is not unitary: {u}")

    try:
        from scipy.linalg import cossin
    except ImportError:  # pragma: no cover
        raise ImportError(
            "cirq.three_qubit_unitary_to_operations requires "
            "SciPy 1.5.0+, as it uses the cossin function. Please"
            " upgrade scipy in your environment to use this "
            "function!"
        )
    (u1, u2), theta, (v1h, v2h) = cossin(u, 4, 4, separate=True)

    cs_ops = _cs_to_ops(q0, q1, q2, theta)
    if len(cs_ops) > 0 and cs_ops[-1] == cirq.CZ(q2, q0):
        # optimization A.1 - merging the last CZ from the end of CS into UD
        # cz = cirq.Circuit([cs_ops[-1]]).unitary()
        # CZ(c,a) = CZ(a,c) as CZ is symmetric
        # for the u1⊕u2 multiplexor operator:
        # as u1(b,c) is the operator in case a = \0>,
        # and u2(b,c) is the operator for (b,c) in case a = |1>
        # we can represent the merge by phasing u2 with I ⊗ Z
        u2 = u2 @ np.diag([1, -1, 1, -1])
        cs_ops = cs_ops[:-1]

    d_ud, ud_ops = _two_qubit_multiplexor_to_ops(q0, q1, q2, u1, u2, shift_left=True, atol=atol)

    _, vdh_ops = _two_qubit_multiplexor_to_ops(
        q0, q1, q2, v1h, v2h, shift_left=False, diagonal=d_ud, atol=atol
    )

    return list(cirq.Circuit(vdh_ops + cs_ops + ud_ops).all_operations())


def _cs_to_ops(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, theta: np.ndarray) -> List[ops.Operation]:
    """Converts theta angles based Cosine Sine matrix to operations.

    Using the optimization as per Appendix A.1, it uses CZ gates instead of
    CNOT gates and returns a circuit that skips the terminal CZ gate.

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        theta: theta returned from the Cosine Sine decomposition

    Returns:
         the operations
    """
    # Note: we are using *2 as the thetas are already half angles from the
    # CSD decomposition, but cirq.ry takes full angles.
    angles = _multiplexed_angles(theta * 2)
    rys = [cirq.ry(angle).on(q0) for angle in angles]
    ops = [
        rys[0],
        cirq.CZ(q1, q0),
        rys[1],
        cirq.CZ(q2, q0),
        rys[2],
        cirq.CZ(q1, q0),
        rys[3],
        cirq.CZ(q2, q0),
    ]
    return _optimize_multiplexed_angles_circuit(ops)


def _two_qubit_multiplexor_to_ops(
    q0: ops.Qid,
    q1: ops.Qid,
    q2: ops.Qid,
    u1: np.ndarray,
    u2: np.ndarray,
    shift_left: bool = True,
    diagonal: Optional[np.ndarray] = None,
    atol: float = 1e-8,
) -> Tuple[Optional[np.ndarray], List[ops.Operation]]:
    r"""Converts a two qubit double multiplexor to circuit.
    Input: U_1 ⊕ U_2, with select qubit a (i.e. a = |0> => U_1(b,c),
    a = |1> => U_2(b,c).

    We want this:
        $$
        U_1 ⊕ U_2 = (V ⊕ V) @ (D ⊕ D^{\dagger}) @ (W ⊕ W)
        $$
    We can get it via:
        $$
        U_1 = V @ D @ W       (1)
        U_2 = V @ D^{\dagger} @ W (2)
        $$

    We can derive
        $$
        U_1 U_2^{\dagger}= V @ D^2 @ V^{\dagger}, (3)
        $$

    i.e the eigendecomposition of $U_1 U_2^{\dagger}$ will give us D and V.
    W is easy to derive from (2).

    This function, after calculating V, D and W, also returns the circuit that
    implements these unitaries: V, W on qubits b, c and the middle diagonal
    multiplexer on a,b,c qubits.

    The resulting circuit will have only known two-qubit and one-qubit gates,
    namely CZ, CNOT and rx, ry, PhasedXPow gates.

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        u1: two-qubit operation on b,c for a = |0>
        u2: two-qubit operation on b,c for a = |1>
        shift_left: return the extracted diagonal or not
        diagonal: an incoming diagonal to be merged with
        atol: the absolute tolerance for the two-qubit sub-decompositions.

    Returns:
        The circuit implementing the two qubit multiplexor consisting only of
        known two-qubit and single qubit gates
    """
    u1u2 = u1 @ u2.conj().T
    eigvals, v = cirq.unitary_eig(u1u2)
    d = np.diag(np.sqrt(eigvals))

    w = d @ v.conj().T @ u2

    circuit_u1u2_mid = _middle_multiplexor_to_ops(q0, q1, q2, eigvals)

    if diagonal is not None:
        v = diagonal @ v

    d_v, circuit_u1u2_r = opt.two_qubit_matrix_to_diagonal_and_cz_operations(q1, q2, v, atol=atol)

    w = d_v @ w

    d_w: Optional[np.ndarray]

    # if it's interesting to extract the diagonal then let's do it
    if shift_left:
        d_w, circuit_u1u2_l = opt.two_qubit_matrix_to_diagonal_and_cz_operations(
            q1, q2, w, atol=atol
        )
    # if we are at the end of the circuit, then just fall back to KAK
    else:
        d_w = None
        circuit_u1u2_l = opt.two_qubit_matrix_to_cz_operations(
            q1, q2, w, allow_partial_czs=False, atol=atol
        )

    return d_w, circuit_u1u2_l + circuit_u1u2_mid + circuit_u1u2_r


def _optimize_multiplexed_angles_circuit(operations: Sequence[ops.Operation]):
    """Removes two qubit gates that amount to identity.
    Exploiting the specific multiplexed structure, this methods looks ahead
    to find stripes of 3 or 4 consecutive CZ or CNOT gates and removes them.

    Args:
        operations: operations to be optimized
    Returns:
        the optimized operations
    """
    circuit = cirq.Circuit(operations)
    circuit = cirq.transformers.drop_negligible_operations(circuit)
    if np.allclose(circuit.unitary(), np.eye(8), atol=1e-14):
        return cirq.Circuit([])

    # the only way we can get identity here is if all four CZs are
    # next to each other
    def num_conseq_2qbit_gates(i):
        j = i
        while j < len(operations) and operations[j].gate.num_qubits() == 2:
            j += 1
        return j - i

    operations = list(circuit.all_operations())

    i = 0
    while i < len(operations):
        num_czs = num_conseq_2qbit_gates(i)
        if num_czs == 4:
            operations = operations[:1]
            break
        elif num_czs == 3:
            operations = operations[:i] + [operations[i + 1]] + operations[i + 3 :]
            break
        else:
            i += 1
    return operations


def _middle_multiplexor_to_ops(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, eigvals: np.ndarray):
    theta = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    angles = _multiplexed_angles(theta)
    rzs = [cirq.rz(angle).on(q0) for angle in angles]
    ops = [
        rzs[0],
        cirq.CNOT(q1, q0),
        rzs[1],
        cirq.CNOT(q2, q0),
        rzs[2],
        cirq.CNOT(q1, q0),
        rzs[3],
        cirq.CNOT(q2, q0),
    ]
    return _optimize_multiplexed_angles_circuit(ops)


def _multiplexed_angles(theta: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Calculates the angles for a 4-way multiplexed rotation.

    For example, if we want rz(theta[i]) if the select qubits are in state
    |i>, then, multiplexed_angles returns a[i] that can be used in a circuit
    similar to this:

    ---rz(a[0])-X---rz(a[1])--X--rz(a[2])-X--rz(a[3])--X
                |             |           |            |
    ------------@-------------|-----------@------------|
                              |                        |
    --------------------------@------------------------@

    Args:
        theta: the desired angles for each basis state of the select qubits
    Returns:
        the angles to be used in actual rotations in the circuit implementation
    """
    return (
        np.array(
            [
                (theta[0] + theta[1] + theta[2] + theta[3]),
                (theta[0] + theta[1] - theta[2] - theta[3]),
                (theta[0] - theta[1] - theta[2] + theta[3]),
                (theta[0] - theta[1] + theta[2] - theta[3]),
            ]
        )
        / 4
    )
