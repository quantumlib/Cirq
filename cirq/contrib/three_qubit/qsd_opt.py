from typing import List

import numpy as np
from scipy.linalg import cossin

import cirq
from cirq import Circuit, two_qubit_matrix_to_operations, linalg


def three_qubit_unitary_to_operations(a: cirq.Qid, b: cirq.Qid, c: cirq.Qid,
                                      U: np.ndarray) -> List[cirq.Operation]:
    """Returns operations for a 3 qubit unitary.

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Args:
        a: first qubit
        b: second qubit
        c: third qubit
        U: unitary matrix

    Returns:
        The resulting operations will have only known two-qubit and one-qubit
        gates based operations, namely CZ, CNOT and rx, ry, PhasedXPow gates.
    """
    assert np.shape(U) == (8, 8)
    assert cirq.is_unitary(U)

    (u1, u2), theta, (v1h, v2h) = cossin(U, 4, 4, separate=True)

    CS_ops = _cs_to_ops(a, b, c, theta)
    if len(CS_ops) > 0 and CS_ops[-1] == cirq.CZ(c, a):
        # optimization A.1 - merging the last CZ from the end of CS into UD
        # cz = cirq.Circuit([CS_ops[-1]]).unitary()
        # CZ(c,a) = CZ(a,c) as CZ is symmetric
        # for the u1⊕u2 multiplexor operator:
        # as u1(b,c) is the operator in case a = \0>,
        # and u2(b,c) is the operator for (b,c) in case a = |1>
        # we can represent the merge by phasing u2 with I ⊗ Z
        u2 = u2 @ np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))
        CS_ops = CS_ops[:-1]

    dUD, c_UD = _two_qubit_multiplexor_to_circuit(a,
                                                  b,
                                                  c,
                                                  u1,
                                                  u2,
                                                  shiftLeft=True)

    _, c_VDH = _two_qubit_multiplexor_to_circuit(a,
                                                 b,
                                                 c,
                                                 v1h,
                                                 v2h,
                                                 shiftLeft=False,
                                                 diagonal=dUD)

    return list(cirq.Circuit([c_VDH, CS_ops, c_UD]).all_operations())


def _cs_to_ops(a: cirq.Qid, b: cirq.Qid, c: cirq.Qid,
               theta: np.ndarray) -> List[cirq.Operation]:
    """Converts theta angles based Cosine Sine matrix to operations.

    Using the optimization as per Appendix A.1, it uses CZ gates instead of
    CNOT gates and returns a circuit that skips the terminal CZ gate.

    Args:
        a, b, c: the 3 qubits in order
        theta: theta returned from the Cosine Sine decomposition
    Returns:
         the operations
    """
    # Note: we are using *2 as the thetas are already half angles from the
    # CSD decomposition, but cirq.ry takes full angles.
    angles = _multiplexed_angles(theta * 2)
    rys = [cirq.ry(angle).on(a) for angle in angles]
    ops = [
        rys[0],
        cirq.CZ(b, a), rys[1],
        cirq.CZ(c, a), rys[2],
        cirq.CZ(b, a), rys[3],
        cirq.CZ(c, a)
    ]
    return _optimize_multiplexed_angles_circuit(ops)


def _optimize_multiplexed_angles_circuit(ops):
    """Removes two qubit gates that amount to identity.
    Exploiting the specific multiplexed structure, this methods looks ahead
    to find stripes of 3 or 4 consecutive CZ or CNOT gates and removes them.

    Args:
        ops: operations to be optimized
    Returns:
        the optimized operations
    """
    circuit = cirq.Circuit(ops)
    cirq.optimizers.DropNegligible().optimize_circuit(circuit)
    if np.allclose(circuit.unitary(), np.eye(8), atol=1e-14):
        return cirq.Circuit([])

    # the only way we can get identity here is if all four CZs are
    # next to each other
    def num_conseq_2qbit_gates(i):
        j = i
        while j < len(ops) and ops[j].gate.num_qubits() == 2:
            j += 1
        return j - i

    ops = list(circuit.all_operations())

    i = 0
    while i < len(ops):
        num_czs = num_conseq_2qbit_gates(i)
        if num_czs == 4:
            ops = ops[:1]
            break
        elif num_czs == 3:
            ops = ops[:i] + [ops[i + 1]] + ops[i + 3:]
            break
        else:
            i += 1
    return ops


def _two_qubit_multiplexor_to_circuit(a,
                                      b,
                                      c,
                                      u1,
                                      u2,
                                      shiftLeft=True,
                                      diagonal=np.eye(4)):
    """Converts a two qubit double multiplexor to circuit.
    Input: u1 ⊕ u2, with select qubit a (i.e. a = |0> => u1(b,c),
    a = |1> => u2(b,c).

    We want this:

        u1 ⊕ u2 = V ⊕ V @ D ⊕ D^{adj} @ W ⊕ W

    We can get it via:

        u1 = V @ D @ W       (1)
        u2 = V @ D^{adj} @ W (2)

    We can derive
        u1u2^{adj}= V @ D^2 @ V^{adj}, (3)

    i.e the eigendecomposition of u1u2^{adj} will give us D and V.
    W is easy to derive from (2).

    This function, after calculating V, D and W, also returns the circuit that
    implements these unitaries: V, W on qubits b, c and the middle diagonal
    multiplexer on a,b,c qubits.

    The resulting circuit will have only known two-qubit and one-qubit gates,
    namely CZ, CNOT and rx, ry, PhasedXPow gates.

    Args:
        a: first qubit
        b: second qubit
        c: third qubit
        u1: two-qubit operation on b,c for a = |0>
        u2: two-qubit operation on b,c for a = |1>
        shiftLeft: return the extracted diagonal or not
        diagonal: an incoming diagonal to be merged with
    Returns:
        The circuit implementing the two qubit multiplexor consisting only of
        known two-qubit and single qubit gates
    """
    u1u2 = u1 @ u2.conj().T
    eigvals, V = cirq.unitary_eig(u1u2)
    d = np.diag(np.sqrt(eigvals))

    W = d @ V.conj().T @ u2

    circuit_u1u2_mid = _middle_multiplexor_to_ops(a, b, c, eigvals)

    V = diagonal @ V

    circuit_u1u2_R, dV = _decompose_to_diagonal_and_circuit(b, c, V)

    W = dV @ W

    # if it's interesting to extract the diagonal then let's do it
    if shiftLeft:
        circuit_u1u2_L, dW = _decompose_to_diagonal_and_circuit(b, c, W)
    # if we are at the end of the circuit, then just fall back to KAK
    else:
        dW = np.eye(4)
        circuit_u1u2_L = cirq.Circuit(
            two_qubit_matrix_to_operations(b, c, W, allow_partial_czs=False))

    return dW, cirq.Circuit([circuit_u1u2_L, circuit_u1u2_mid, circuit_u1u2_R])


def _middle_multiplexor_to_ops(a, b, c, eigvals):
    theta = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    angles = _multiplexed_angles(theta)
    rzs = [cirq.rz(angle).on(a) for angle in angles]
    ops = [
        rzs[0],
        cirq.CNOT(b, a), rzs[1],
        cirq.CNOT(c, a), rzs[2],
        cirq.CNOT(b, a), rzs[3],
        cirq.CNOT(c, a)
    ]
    return _optimize_multiplexed_angles_circuit(ops)

def _multiplexed_angles(theta):
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
    return np.array([(theta[0] + theta[1] + theta[2] + theta[3]),
                     (theta[0] + theta[1] - theta[2] - theta[3]),
                     (theta[0] - theta[1] - theta[2] + theta[3]),
                     (theta[0] - theta[1] + theta[2] - theta[3])]) / 4


def _decompose_to_diagonal_and_circuit(b, c, V):
    """Returns a circuit and a diagonal for 2-qubit unitary.

    For a 2-qubit unitary V, return c cirq.Circuit and D diagonal unitary,
    so that:
        V = c.unitary() @ D

    Args:
        b, c: qubits
        V: the input unitary
    Returns:
        tuple(c,D): circuit c, and the diagonal D
    """
    if cirq.is_diagonal(V, atol=1e-15):
        circuit = cirq.Circuit([])
        d = V
    elif linalg.num_cnots_required(V) == 3:
        right_diag = linalg.extract_right_diag(V)
        # two-CNOT unitary
        two_CNOT = V @ right_diag
        # thus, we can see that two_CNOT @ d = V
        d = right_diag.conj().T
        circuit = Circuit(
            two_qubit_matrix_to_operations(b,
                                           c,
                                           two_CNOT,
                                           allow_partial_czs=False))
    else:
        d = np.eye(4)
        circuit = Circuit(
            two_qubit_matrix_to_operations(b, c, V, allow_partial_czs=False))
    return circuit, d
