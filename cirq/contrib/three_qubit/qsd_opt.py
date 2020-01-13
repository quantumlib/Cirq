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
        psi = np.pi/2
    c_d = cirq.Circuit([cirq.CNOT(a, b), cirq.rz(psi)(b), cirq.CNOT(a, b)])
    return c_d._unitary_()


def closest_unitary(A):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.

        Return U as a numpy matrix.
    """
    V, __, Wh = np.linalg.svd(A)
    U = np.array(V @ Wh)
    return U


def isThreeCNOTUnitary(U):
    poly = np.poly(g(special(U)))
    return not np.alltrue(np.isclose(0, np.imag(poly)))


def mxs(U):
    return "np.array({})".format(np.array2string(a=U, separator=","))


def multiplexor_to_circuit(a, b, c, u1, u2, shiftLeft=True, diagonal=np.eye(4)):
    print("u1", u1)
    print("u2", u2)
    u1u2 = u1 @ u2.conj().T

    print(u1u2)
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
    print("W", W)
    #     print(combo)
    #     print(vs @ mid @ ws)
    np.testing.assert_almost_equal(vs @ mid @ ws, combo)

    theta2 = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    print(theta2)

    #     print(np.array2string(theta2, precision=2,suppress_small=True))
    #     print(np.exp(theta2 / 2 * 1j))
    #     f or th in theta2:
    #         print(cirq.rz(th)._unitary_())
    circuit_u1u2_mid = cirq.Circuit([
        cirq.rz((theta2[0] + theta2[1] + theta2[2] + theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.rz((theta2[0] + theta2[1] - theta2[2] - theta2[3]) / 4).on(a),
        cirq.CNOT(c, a),
        cirq.rz((theta2[0] - theta2[1] - theta2[2] + theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.rz((theta2[0] - theta2[1] + theta2[2] - theta2[3]) / 4).on(a),
        cirq.CNOT(c, a)])

    #     print(circuit_u1u2_mid)
    #     print(circuit_u1u2_mid._unitary_())
    np.testing.assert_almost_equal(mid, circuit_u1u2_mid._unitary_())

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


def two_qubit_matrix_to_diagonal_and_circuit(V, b, c):
    if cirq.is_diagonal(V, atol=1e-15):
        print("Unitary is already diagonal!")
        print("V", V)
        circuit_u1u2_R = []
        dV = V.conj().T
    elif isThreeCNOTUnitary(V):
        print("Three CNOT unitary, extracting diagonal...")
        print("V", V)
        dV = extract_right_diag(b, c, V)
        print("V", mxs(V))
        V = V @ dV
        print("V", V)
        circuit_u1u2_R = cirq.Circuit(
            cirq.optimizers.two_qubit_matrix_to_operations(b, c, V,
                                                           allow_partial_czs=False))
    else:
        print("Less than 3 CNOT unitary falling back to KAK")
        print("V", mxs(V))
        dV = np.eye(4)
        circuit_u1u2_R = cirq.Circuit(
            cirq.optimizers.two_qubit_matrix_to_operations(b, c, V,
                                                           allow_partial_czs=False))
    return circuit_u1u2_R, dV


def three_qubit_unitary_to_operations(U):
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

    a, b, c = cirq.LineQubit.range(3)
    # Note: we are using / 2 as the thetas are already half angles - and ry takes
    # full angles.
    # we are using CZ's as an optimization as per Appendix A.1 in
    circuit_CS = cirq.Circuit([
        cirq.ry((theta[0] + theta[1] + theta[2] + theta[3]) / 2).on(a),
        cirq.CZ(b, a),
        cirq.ry((theta[0] + theta[1] - theta[2] - theta[3]) / 2).on(a),
        cirq.CZ(c, a),
        cirq.ry((theta[0] - theta[1] - theta[2] + theta[3]) / 2).on(a),
        cirq.CZ(b, a),
        cirq.ry((theta[0] - theta[1] + theta[2] - theta[3]) / 2).on(a)])

    # print(circuit_CS)

    ##
    ## What we need is ..
    ## a + b + c + d = t0
    ## a - b + c - d = t1
    ## a + b - c - d = t2
    ## a - b - c + d  = t3
    ##
    ##
    # print (circuit_CS._unitary_())
    # print(CS)
    # assert_almost_equal(circuit_CS._unitary_(), CS, decimals)

    # print(np.cos(theta))
    # print(np.sin(theta))
    # for th in theta:
    #     print(cirq.ry(th*2)._unitary_())

    # optimization A.1 - merging the CZ(c,a) from the end of CS into UD
    u2 = u2 @ np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))
    UD = UD @ cirq.Circuit(cirq.CZ(c, a),
                           cirq.IdentityGate(1).on(b))._unitary_()

    dUD, c_UD = multiplexor_to_circuit(a, b, c, u1, u2, shiftLeft=True)
    cirq.testing.assert_allclose_up_to_global_phase(UD,
                                                    c_UD._unitary_() @ np.kron(
                                                        np.eye(2), dUD),
                                                    atol=1e-8)

    dVDH, c_VDH = multiplexor_to_circuit(a, b, c, v1h, v2h, shiftLeft=False,
                                         diagonal=dUD)

    cirq.testing.assert_allclose_up_to_global_phase(
        np.kron(np.eye(2), dUD) @ VDH, c_VDH._unitary_(),
        atol=1e-8)

    final_circuit = cirq.Circuit([c_VDH, circuit_CS, c_UD])
    cirq.testing.assert_allclose_up_to_global_phase(U,
                                                    final_circuit._unitary_(),
                                                    atol=1e-9)
    return final_circuit


def random_unitary(n_qubits=3):
    M = 2 ** n_qubits  # size of unitary matrix
    H = np.random.rand(M, M) + 1.9j * np.random.rand(M, M)
    H = H + H.conj().T
    D, U = eigh(H)
    return U


np.set_printoptions(precision=16, suppress=False, linewidth=300,
                    floatmode='maxprec_equal')


def decompose(circuit):
    print()
    print()
    print("--------------- START -------------- ")
    print()
    print()
    print(circuit)
    print("cirq.Circuit({})".format(list(circuit.all_operations())))
    U = circuit._unitary_()
    print(U)
    if len(U) < 8:
        print("less than 3 qubit circuit, skipping...")
        return
    # print(circuit)
    final_circuit = three_qubit_unitary_to_operations(U)
    print("result: ")
    print(final_circuit)
    num_two_qubits = sum([1 for op in final_circuit.all_operations() if
                          op.gate.num_qubits() == 2])
    print("CNOT/CZs: {}, All gates: {}".format(
        num_two_qubits,
        len(list(final_circuit.all_operations()))))
    if num_two_qubits > 20: exit(1)
    print("XMON optimized: ")
    final_circuit = cirq.google.optimizers.optimized_for_xmon(final_circuit)
    print(final_circuit)
    num_two_qubits = sum([1 for op in final_circuit.all_operations() if
                          op.gate.num_qubits() == 2])
    cirq.PhasedXPowGate()
    print("CNOT/CZs: {}, All gates: {}".format(
        num_two_qubits,
        len(list(final_circuit.all_operations()))))

    print(cirq.contrib.quirk.circuit_to_quirk_url((final_circuit)))


a, b, c = cirq.LineQubit.range(3)
decompose(cirq.Circuit(cirq.ControlledGate(cirq.ISWAP).on(a, b, c)))


# #
# # U = random_unitary()
# for i in range(1000):
#     a, b, c = cirq.LineQubit.range(3)
#
#     # circuit = cirq.Circuit(
#     #     [cirq.PhasedXPowGate(exponent=random(), phase_exponent=random())(a),
#     #      cirq.PhasedXPowGate(exponent=random(), phase_exponent=random())(b),
#     #      cirq.CNOT(a, b),
#     #      cirq.PhasedXPowGate(exponent=random(), phase_exponent=random())(c),
#     #       ])
#
#     circuit = cirq.testing.random_circuit([a,b,c], 10, 0.75)
#     decompose(circuit)
#
#
