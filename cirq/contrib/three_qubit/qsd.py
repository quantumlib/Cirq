import cirq
import numpy as np
from numpy.linalg import eigh
from numpy.testing import assert_almost_equal
from pycsd import cs_decomp

n_qubits = 3
M = 2 ** n_qubits  # size of unitary matrix

H = np.random.rand(M, M) + 1.9j * np.random.rand(M, M)
H = H + H.conj().T
D, U = eigh(H)
np.set_printoptions(precision=2, suppress=False, linewidth=300,
                    floatmode='maxprec_equal')
print(U)

P = int(M / 2)  # number of rows in upper left block
u1, u2, v1h, v2h, theta = cs_decomp(U, P, P)
print(u1)
print(u2)
print(v1h)
print(v2h)
print(theta)

decimals = 14  # desired precision

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

assert_almost_equal(U, UD @ CS @ VDH, decimals)

a, b, c = cirq.LineQubit.range(3)
# Note: we are using / 2 as the thetas are half angles
# we are using CZ's as an optimization as per Appendix A.1 in
circuit_CS = cirq.Circuit([
    cirq.Ry((theta[0] + theta[1] + theta[2] + theta[3]) / 2).on(a),
    cirq.CZ(b, a),
    cirq.Ry((theta[0] + theta[1] - theta[2] - theta[3]) / 2).on(a),
    cirq.CZ(b, a),
    cirq.CZ(c, a),
    cirq.Ry((theta[0] - theta[1] + theta[2] - theta[3]) / 2).on(a),
    cirq.CZ(b, a),
    cirq.Ry((theta[0] - theta[1] - theta[2] + theta[3]) / 2).on(a),
    cirq.CZ(b, a),
    cirq.CZ(c, a)])

print(circuit_CS)

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
assert_almost_equal(circuit_CS._unitary_(), CS, decimals)


# print(np.cos(theta))
# print(np.sin(theta))
# for th in theta:
#     print(cirq.Ry(th*2)._unitary_())

def multiplexor_to_circuit(u1, u2):
    u1u2 = u1 @ u2.conj().T
    eigvals, V = np.linalg.eig(u1u2)

    d = np.diag(np.sqrt(eigvals))
    np.testing.assert_almost_equal(V @ d @ d @ V.conj().T, u1u2)

    z = np.zeros((P, P))

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

    #     print(combo)
    #     print(vs @ mid @ ws)
    np.testing.assert_almost_equal(vs @ mid @ ws, combo)

    theta2 = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    #     print(np.array2string(theta2, precision=2,suppress_small=True))
    #     print(np.exp(theta2 / 2 * 1j))
    #     for th in theta2:
    #         print(cirq.Rz(th)._unitary_())
    circuit_u1u2_mid = cirq.Circuit([
        cirq.Rz((theta2[0] + theta2[1] + theta2[2] + theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.Rz((theta2[0] + theta2[1] - theta2[2] - theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.CNOT(c, a),
        cirq.Rz((theta2[0] - theta2[1] + theta2[2] - theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.Rz((theta2[0] - theta2[1] - theta2[2] + theta2[3]) / 4).on(a),
        cirq.CNOT(b, a),
        cirq.CNOT(c, a)])

    #     print(circuit_u1u2_mid)
    #     print(circuit_u1u2_mid._unitary_())
    np.testing.assert_almost_equal(mid, circuit_u1u2_mid._unitary_())

    circuit_u1u2_left = cirq.Circuit(
        cirq.optimizers.two_qubit_matrix_to_operations(b, c, V,
                                                       allow_partial_czs=False))
    cirq.testing.assert_allclose_up_to_global_phase(V,
                                                    circuit_u1u2_left._unitary_(),
                                                    atol=1e-8)
    cirq.linalg.match_global_phase(V, circuit_u1u2_left._unitary_())

    circuit_u1u2_right = cirq.Circuit(
        cirq.optimizers.two_qubit_matrix_to_operations(b, c, W,
                                                       allow_partial_czs=False))
    cirq.testing.assert_allclose_up_to_global_phase(W,
                                                    circuit_u1u2_right._unitary_(),
                                                    atol=1e-8)
    cirq.linalg.match_global_phase(W, circuit_u1u2_right._unitary_())

    return cirq.Circuit(
        [circuit_u1u2_right, circuit_u1u2_mid, circuit_u1u2_left])


c_UD = multiplexor_to_circuit(u1, u2)
cirq.testing.assert_allclose_up_to_global_phase(UD, c_UD._unitary_(), atol=1e-8)

c_VDH = multiplexor_to_circuit(v1h, v2h)
cirq.testing.assert_allclose_up_to_global_phase(VDH, c_VDH._unitary_(),
                                                atol=1e-8)

final_circuit = cirq.Circuit([c_VDH, circuit_CS, c_UD])
cirq.testing.assert_allclose_up_to_global_phase(U, final_circuit._unitary_(),
                                                atol=1e-8)

print(final_circuit)
print(sum(
    [1 for op in final_circuit.all_operations() if op.gate.num_qubits() == 2]),
      len(final_circuit))
for op in final_circuit.all_operations():
    if op.gate.num_qubits() == 2:  print(op)
