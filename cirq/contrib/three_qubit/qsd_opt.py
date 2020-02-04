from random import random, randint

import cirq
import numpy as np
from numpy.linalg import eigh
from numpy.testing import assert_almost_equal
from pycsd import cs_decomp


def special(u):
    return u / (np.linalg.det(u) ** (1 / 4))


def g(u):
    yy = np.kron(cirq.Y._unitary_(), cirq.Y._unitary_())
    return u @ yy @ u.T @ yy


def extract_right_diag(a, b, U):
    u = special(U)
    t = g(u.T).T.diagonal()
    psi = np.arctan(np.imag(np.sum(t)) / np.real(t[0] + t[3] - t[1] - t[2]))
    if np.real(t[0] + t[3] - t[1] - t[2]) == 0:
        psi = np.pi / 2
    c_d = cirq.Circuit([cirq.CNOT(a, b), cirq.rz(psi)(b), cirq.CNOT(a, b)])
    return c_d._unitary_()


def closest_unitary(A):
    V, __, W = np.linalg.svd(A)
    return np.array(V @ W)


def isThreeCNOTUnitary(U):
    poly = np.poly(g(special(U)))
    return not np.alltrue(np.isclose(0, np.imag(poly)))


def mxs(U):
    return "np.array({})".format(np.array2string(a=U, separator=","))


def multiplexor_to_circuit(a, b, c, u1, u2, shiftLeft=True, diagonal=np.eye(4)):
    u1u2 = u1 @ u2.conj().T
    eigvals, V = np.linalg.eig(u1u2)
    ## sometimes V becomes non-unitary due to rounding errors, this fixes it
    V = closest_unitary(V)

    d = np.diag(np.sqrt(eigvals))
    cirq.testing.assert_allclose_up_to_global_phase(V @ d @ d @ V.conj().T,
                                                    u1u2, atol=1e-8)

    z = np.zeros((4, 4))

    combo = np.vstack(
        (
            np.hstack((u1, z)),
            np.hstack((z, u2))
        )
    )
    mid = np.vstack(
        (
            np.hstack((d, z)),
            np.hstack((z, d.conj().T))
        )
    )

    vs = np.vstack(
        (
            np.hstack((V, z)),
            np.hstack((z, V))
        )
    )
    W = d @ V.conj().T @ u2
    ws = np.vstack(
        (
            np.hstack((W, z)),
            np.hstack((z, W))
        )
    )

    np.testing.assert_almost_equal(vs @ mid @ ws, combo)

    theta = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)

    angles = multiplexed_angles(theta)

    rzs = [cirq.rz(angle).on(a) if angle != 0 else []
           for angle in angles]

    if rzs[1] == [] and rzs[2] == []:
        if rzs[3] == []:
            ops = [
                rzs[0],
            ]
        else:
            ops = [
                rzs[0],
                cirq.CNOT(c, a),
                rzs[3],
                cirq.CNOT(c, a)]
    else:
        ops = [rzs[0],
               cirq.CNOT(b, a),
               rzs[1],
               cirq.CNOT(c, a),
               rzs[2],
               cirq.CNOT(b, a),
               rzs[3],
               cirq.CNOT(c, a)]

    circuit_u1u2_mid = cirq.Circuit(ops)

    np.testing.assert_almost_equal(mid, circuit_u1u2_mid.unitary(
        qubits_that_should_be_present=[a, b, c]))

    V = diagonal @ V

    circuit_u1u2_R, dV = two_qubit_matrix_to_diagonal_and_circuit(V, b, c)

    W = dV.conj().T @ W

    # if it's interesting to extract the diagonal then let's do it
    if shiftLeft:
        circuit_u1u2_L, dW = two_qubit_matrix_to_diagonal_and_circuit(W, b, c)
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


def multiplexed_angles(theta):
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


def two_qubit_matrix_to_diagonal_and_circuit(V, b, c):
    if cirq.is_diagonal(V, atol=1e-15):
        circuit_u1u2_R = []
        dV = V.conj().T
    elif isThreeCNOTUnitary(V):
        dV = extract_right_diag(b, c, V)
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


def three_qubit_unitary_to_operations(U):
    CS, UD, VDH, theta, u1, u2, v1h, v2h = csd(U)

    a, b, c = cirq.LineQubit.range(3)

    circuit_CS, rightmost_CZ = cs_to_circuit_and_rightmost_cz(CS, a, b, c,
                                                              theta)

    # optimization A.1 - merging the CZ(c,a) from the end of CS into UD
    u2 = u2 @ np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))

    UD = UD @ rightmost_CZ

    dUD, c_UD = multiplexor_to_circuit(a, b, c, u1, u2, shiftLeft=True)
    cirq.testing.assert_allclose_up_to_global_phase(UD,
                                                    c_UD.unitary(
                                                        qubits_that_should_be_present=[
                                                            a, b, c]) @ np.kron(
                                                        np.eye(2), dUD),
                                                    atol=1e-8)

    dVDH, c_VDH = multiplexor_to_circuit(a, b, c, v1h, v2h, shiftLeft=False,
                                         diagonal=dUD)

    cirq.testing.assert_allclose_up_to_global_phase(
        np.kron(np.eye(2), dUD) @ VDH,
        c_VDH.unitary(qubits_that_should_be_present=[a, b, c]),
        atol=1e-8)

    final_circuit = cirq.Circuit([c_VDH, circuit_CS, c_UD])
    cirq.testing.assert_allclose_up_to_global_phase(U,
                                                    final_circuit.unitary(
                                                        qubits_that_should_be_present=[
                                                            a, b, c]),
                                                    atol=1e-9)
    return final_circuit


def csd(U):
    n_qubits = 3
    M = 2 ** n_qubits  # size of unitary matrix
    P = int(M / 2)  # number of rows in upper left block
    u1, u2, v1h, v2h, theta = cs_decomp(U, P, P)
    z = np.zeros((P, P))
    UD = np.vstack(
        (
            np.hstack((u1, z)),
            np.hstack((z, u2))
        )
    )
    VDH = np.vstack((np.hstack((v1h, z)),
                     np.hstack((z, v2h))))
    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    CS = np.vstack((np.hstack((C, -S)), np.hstack((S, C))))
    assert_almost_equal(U, UD @ CS @ VDH)
    return CS, UD, VDH, theta, u1, u2, v1h, v2h


def cs_to_circuit_and_rightmost_cz(CS, a, b, c, theta):
    # Note: we are using *2 as the thetas are already half angles from the
    # CSD decomposition, but ry takes full angles.
    angles = multiplexed_angles(theta * 2)
    rys = [cirq.ry(angle).on(a) if angle != 0 else []
           for angle in angles]

    # we are using CZ's as an optimization as per Appendix A.1
    if rys[1] == [] and rys[2] == []:
        ops = [
            rys[0],
            cirq.CZ(c, a),
            rys[3]]
    else:
        ops = [rys[0],
               cirq.CZ(b, a),
               rys[1],
               cirq.CZ(c, a),
               rys[2],
               cirq.CZ(b, a),
               rys[3]]

    circuit_CS = cirq.Circuit(ops)

    # the rightmost CZ gate is to be merged with the UD multiplexor later
    # as per Appendix A.1
    rightmost_CZ = cirq.Circuit(cirq.CZ(c, a)).unitary(
        qubits_that_should_be_present=[a, b, c])

    assert_almost_equal(
        rightmost_CZ @ circuit_CS.unitary(
            qubits_that_should_be_present=[a, b, c]), CS, 8)
    return circuit_CS, rightmost_CZ


def random_unitary(n_qubits=3):
    M = 2 ** n_qubits  # size of unitary matrix
    H = np.random.rand(M, M) + 1.9j * np.random.rand(M, M)
    H = H + H.conj().T
    D, U = eigh(H)
    return U


np.set_printoptions(precision=16, suppress=False, linewidth=300,
                    floatmode='maxprec_equal')


def decompose_circuit(circuit):
    print()
    print()
    print("--------------- START -------------- ")
    print()
    print()
    print(circuit)
    print("cirq.Circuit({})".format(list(circuit.all_operations())))
    decompose_unitary(circuit._unitary_())


def decompose_unitary(U):
    print(U)
    final_circuit = three_qubit_unitary_to_operations(U)
    print("result: ")
    print(final_circuit)
    num_two_qubits = sum([1 for op in final_circuit.all_operations() if
                          op.gate.num_qubits() == 2])
    print("CNOT/CZs: {}, All gates: {}".format(
        num_two_qubits,
        len(list(final_circuit.all_operations()))))
    if num_two_qubits > 20: exit(1)
    print("cirq.Circuit({})".format(list(final_circuit.all_operations())))
    print("XMON optimized: ")
    final_circuit = cirq.google.optimizers.optimized_for_xmon(final_circuit)
    print(final_circuit)
    num_two_qubits = sum([1 for op in final_circuit.all_operations() if
                          op.gate.num_qubits() == 2])
    print("CNOT/CZs: {}, All gates: {}".format(
        num_two_qubits,
        len(list(final_circuit.all_operations()))))

    print(cirq.contrib.quirk.circuit_to_quirk_url((final_circuit)))


if __name__ == '__main__':
    a, b, c = cirq.LineQubit.range(3)
    decompose_unitary(
        cirq.Circuit(cirq.ControlledGate(cirq.ISWAP).on(a, b, c))._unitary_())

    U = random_unitary()
    decompose_unitary(U)
    for i in range(1000):
        a, b, c = cirq.LineQubit.range(3)
        circuit = cirq.testing.random_circuit([a, b, c], 10, 0.75)
        decompose_unitary(
            circuit.unitary(qubits_that_should_be_present=[a, b, c]))
