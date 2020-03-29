from scipy.linalg import cossin
from scipy.linalg import block_diag

import cirq
import numpy as np


def three_qubit_unitary_to_operations(U):
    assert np.shape(U) == (8, 8)
    (u1, u2), theta, (v1h, v2h) = cossin(U, 4, 4, separate=True)

    a, b, c = cirq.LineQubit.range(3)

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

    dUD, c_UD = _two_qubit_multiplexor_to_circuit(a, b, c, u1, u2,
                                                  shiftLeft=True)

    UD = block_diag(u1, u2)
    cirq.testing.assert_allclose_up_to_global_phase(UD,
                                                    c_UD.unitary(
                                                        qubits_that_should_be_present=[
                                                            a, b, c]) @ np.kron(
                                                        np.eye(2), dUD),
                                                    atol=1e-8)

    dVDH, c_VDH = _two_qubit_multiplexor_to_circuit(a, b, c, v1h, v2h,
                                                    shiftLeft=False,
                                                    diagonal=dUD)

    VDH = block_diag(v1h, v2h)
    cirq.testing.assert_allclose_up_to_global_phase(
        np.kron(np.eye(2), dUD) @ VDH,
        c_VDH.unitary(qubits_that_should_be_present=[a, b, c]),
        atol=1e-8)

    final_circuit = cirq.Circuit([c_VDH, CS_ops, c_UD])
    cirq.testing.assert_allclose_up_to_global_phase(U,
                                                    final_circuit.unitary(
                                                        qubits_that_should_be_present=[
                                                            a, b, c]),
                                                    atol=1e-9)
    return final_circuit


def _cs_to_ops(a, b, c: cirq.Qid, theta: np.ndarray):
    """ Converts theta angles based Cosine Sine matrix to operations.

    Using the optimization as per Appendix A.1, it uses CZ gates instead of
    CNOT gates and returns a circuit that skips the terminal CZ gate.

    :param a, b, c: the 3 qubits in order
    :param theta: theta returned from the Cosine Sine decomposition
    :return: the circuit
    """
    # Note: we are using *2 as the thetas are already half angles from the
    # CSD decomposition, but cirq.ry takes full angles.
    angles = _multiplexed_angles(theta * 2)
    rys = [cirq.ry(angle).on(a) for angle in angles]
    ops = [rys[0],
           cirq.CZ(b, a),
           rys[1],
           cirq.CZ(c, a),
           rys[2],
           cirq.CZ(b, a),
           rys[3],
           cirq.CZ(c, a)]
    return _optimize_multiplexed_angles_circuit(ops)


def _optimize_multiplexed_angles_circuit(ops):
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


def _two_qubit_multiplexor_to_circuit(a, b, c, u1, u2, shiftLeft=True,
                                      diagonal=np.eye(4)):
    u1u2 = u1 @ u2.conj().T
    eigvals, V = cirq.unitary_eig(u1u2)
    d = np.diag(np.sqrt(eigvals))

    W = d @ V.conj().T @ u2

    circuit_u1u2_mid = _middle_multiplexor_to_ops(a, b, c, eigvals)

    V = diagonal @ V

    circuit_u1u2_R, dV = _two_qubit_matrix_to_diagonal_and_circuit(V, b, c)

    W = dV.conj().T @ W

    # if it's interesting to extract the diagonal then let's do it
    if shiftLeft:
        circuit_u1u2_L, dW = _two_qubit_matrix_to_diagonal_and_circuit(W, b, c)
    # if we are at the end of the circuit, then just fall back to KAK
    else:
        dW = np.eye(4)
        circuit_u1u2_L = cirq.Circuit(
            cirq.optimizers.two_qubit_matrix_to_operations(b, c, W,
                                                           allow_partial_czs=False))

    return dW.conj().T, cirq.Circuit(
        [circuit_u1u2_L,
         circuit_u1u2_mid,
         circuit_u1u2_R])


def _middle_multiplexor_to_ops(a, b, c, eigvals):
    theta = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    angles = _multiplexed_angles(theta)
    rzs = [cirq.rz(angle).on(a) for angle in angles]
    ops = [rzs[0],
           cirq.CNOT(b, a),
           rzs[1],
           cirq.CNOT(c, a),
           rzs[2],
           cirq.CNOT(b, a),
           rzs[3],
           cirq.CNOT(c, a)]
    return _optimize_multiplexed_angles_circuit(ops)


def _to_special(u):
    """Converts a unitary matrix to a special unitary operator.
    All unitary matrix u have |det(u)| = 1.
    Also for all d dimensional unitary matrix u, and scalar s:
        det(u * s) = det(u) * s^(d)
    To find a special unitary matrix from u:
        u * det(u)^{-1/d}
    :param u: the unitary matrix
    :return: the special unitary matrix
    """
    return u * (np.linalg.det(u) ** (-1 / len(u)))


def _gamma(g):
    """Gamma function to convert u to the magic basis.

    See Definition IV.1 in Minimal Universal Two-Qubit CNOT-based Circuits.
    https://arxiv.org/abs/quant-ph/0308033
    :param g:
    :return:
    """
    yy = np.kron(cirq.Y._unitary_(), cirq.Y._unitary_())
    return g @ yy @ g.T @ yy


def _extract_right_diag(U):
    """ Extract a diagonal unitary from a 3-CNOT two-qubit unitary.

    Returns a 2-CNOT unitary D that is diagonal, so that U @ D needs only
    two CNOT gates.

    See Proposition V.2 in Minimal Universal Two-Qubit CNOT-based Circuits.
    https://arxiv.org/abs/quant-ph/0308033

    :param U: three-CNOT two-qubit unitary
    :return: diagonal extarcted from U
    """
    t = _gamma(_to_special(U).T).T.diagonal()
    k = np.real(t[0] + t[3] - t[1] - t[2])

    if k == 0:
        # TODO: why does this work?
        psi = np.pi/2
    else:
        psi = np.arctan(np.imag(np.sum(t)) / k)

    a, b = cirq.LineQubit.range(2)
    c_d = cirq.Circuit([cirq.CNOT(a, b), cirq.rz(psi)(b), cirq.CNOT(a, b)])
    return c_d._unitary_()


def _is_three_cnot_two_qubit_unitary(U):
    assert np.shape(U) == (4, 4)
    poly = np.poly(_gamma(_to_special(U)))
    return not np.alltrue(np.isclose(0, np.imag(poly)))


def _multiplexed_angles(theta):
    """
    Calculates the angles for a 4-way multiplexed rotation.

    For example, if we want rz(theta[i]) if the select qubits are in state
    |i>, then, multiplexed_angles returns a[i] that can be used in a circuit
    similar to this:

    ---rz(a[0])-X---rz(a[1])--X--rz(a[2])-X--rz(a[3])--X
                |             |           |            |
    ------------@-------------|-----------@------------|
                              |                        |
    --------------------------@------------------------@

    :param theta: the desired angles for each basis state of the select qubits
    :return: the angles to be used in actual rotations in the
     circuit implementation
    """
    return np.array(
        [(theta[0] + theta[1] + theta[2] + theta[3]),
         (theta[0] + theta[1] - theta[2] - theta[3]),
         (theta[0] - theta[1] - theta[2] + theta[3]),
         (theta[0] - theta[1] + theta[2] - theta[3])
         ]) / 4


def _two_qubit_matrix_to_diagonal_and_circuit(V, b, c):
    if cirq.is_diagonal(V, atol=1e-15):
        circuit_u1u2_R = []
        dV = V.conj().T
    elif _is_three_cnot_two_qubit_unitary(V):
        dV = _extract_right_diag(V)
        V = V @ dV
        circuit_u1u2_R = cirq.Circuit(
            cirq.optimizers.two_qubit_matrix_to_operations(b, c, V,
                                                           allow_partial_czs=False))
    else:
        dV = np.eye(4)
        circuit_u1u2_R = cirq.Circuit(
            cirq.optimizers.two_qubit_matrix_to_operations(b, c, V,
                                                           allow_partial_czs=False))
    return circuit_u1u2_R, dV
