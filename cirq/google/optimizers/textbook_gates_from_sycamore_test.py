import functools
import itertools
import numpy as np

import cirq
import cirq.google.optimizers.textbook_gates_from_sycamore as cgot

import scipy.linalg


def _assert_equivalent_op_tree(x: cirq.OP_TREE, y: cirq.OP_TREE):
    a = list(cirq.flatten_op_tree(x))
    b = list(cirq.flatten_op_tree(y))
    assert a == b


def random_single_qubit_unitary():
    a, b, c = np.random.random(3) * 2 * np.pi
    circuit = cirq.unitary(cirq.Rz(a)) @ cirq.unitary(
        cirq.Ry(b)) @ cirq.unitary(cirq.Rz(c))
    assert np.allclose(circuit.conj().T @ circuit, np.eye(2))
    return circuit


def test_unitary_decomp():
    q = cirq.GridQubit(0, 0)
    random_unitary = random_single_qubit_unitary()
    circuit = cirq.Circuit([
        term.on(q) for term in cirq.single_qubit_matrix_to_gates(random_unitary)
    ])
    assert np.isclose(
        abs(np.trace(cirq.unitary(circuit).conj().T @ random_unitary)), 2.0)


def test_sycamorecz():
    qubits = cirq.LineQubit.range(2)
    sycamore_cz = cirq.Circuit(cgot.decompose_cz_into_syc(qubits[0], qubits[1]))
    test_cz_gate = cirq.unitary(sycamore_cz)
    true_cz_gate = np.diag([1, 1, 1, -1])
    overlap = abs(np.trace(test_cz_gate.conj().T @ true_cz_gate))
    assert np.isclose(overlap, 4.0)


def test_sycamore_iswap():
    qubits = cirq.LineQubit.range(2)
    sycamore_iswap = cirq.Circuit(
        cgot.decompose_iswap_into_syc(qubits[0], qubits[1]))
    test_iswap_gate = cirq.unitary(sycamore_iswap)
    true_iswap_gate = cirq.unitary(cirq.ISWAP)
    overlap = abs(np.trace(test_iswap_gate.conj().T @ true_iswap_gate))
    assert np.isclose(overlap, 4.0)


def test_sycamore_swap():
    qubits = cirq.LineQubit.range(2)
    sycamore_swap = cirq.Circuit(
        cgot.decompose_swap_into_syc(qubits[0], qubits[1]))
    test_swap_gate = cirq.unitary(sycamore_swap)
    true_swap_gate = cirq.unitary(cirq.SWAP)
    overlap = abs(np.trace(test_swap_gate.conj().T @ true_swap_gate))
    assert np.isclose(overlap, 4.0)


def test_operator_decomp():
    theta1 = np.pi / 3
    theta2 = np.pi / 4
    x = cirq.unitary(cirq.X)
    G1 = np.cos(theta1) * np.eye(2) + 1j * np.sin(theta1) * x
    G2 = np.cos(theta2) * np.eye(2) + 1j * np.sin(theta2) * x
    true_matrix = np.zeros((4, 4), dtype=np.complex128)
    true_matrix[0, 0] = np.cos(theta1) * np.cos(theta2)
    true_matrix[1, 1] = -np.sin(theta1) * np.sin(theta2)
    true_matrix[0, 1] = 1j * np.cos(theta1) * np.sin(theta2)
    true_matrix[1, 0] = 1j * np.sin(theta1) * np.cos(theta2)

    op = np.kron(G2, G1)
    coeff_vec = cgot.operator_decomp(op)
    test_matrix = coeff_vec.reshape((4, 4), order='F')
    assert np.allclose(test_matrix, true_matrix)


def test_cz_reconstruction():
    cz = np.diag([1, 1, 1, -1])
    cz_decomp = cgot.operator_decomp(cz)

    pauli_ops = [
        np.eye(2),
        cirq.unitary(cirq.X),
        cirq.unitary(cirq.Y),
        cirq.unitary(cirq.Z)
    ]
    num_qubits = 2
    operator = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    for idx, vec_index in enumerate(
            itertools.product(range(4), repeat=num_qubits)):
        op_basis = functools.reduce(np.kron,
                                    map(lambda x: pauli_ops[x], vec_index))
        operator += cz_decomp[idx] * op_basis

    assert np.allclose(operator, cz)


def test_schmidt_decomp():
    cz = np.diag([1, 1, 1, -1])
    zz = np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))
    phi = -np.pi / 24
    theta1 = np.pi / 7
    theta2 = np.pi / 3
    x = cirq.unitary(cirq.X)
    y = cirq.unitary(cirq.Y)
    z = cirq.unitary(cirq.Z)
    G1 = np.cos(theta1) * np.eye(2) + 1j * np.sin(theta1) * cirq.unitary(cirq.X)
    G2 = np.cos(theta2) * np.eye(2) + 1j * np.sin(theta2) * cirq.unitary(cirq.X)
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    A = np.zeros((4, 4), dtype=np.complex128)
    A[0, 0] = c1 * c2
    A[1, 1] = -s1 * s2
    A[0, 1] = 1j * c1 * s2
    A[1, 0] = 1j * s1 * c2
    g_decomp = cgot.operator_decomp(np.kron(G2, G1)).reshape((4, 4), order='F')
    assert np.allclose(A, g_decomp)

    Bop = cz.conj().T @ np.kron(G2, G1) @ cz

    trial_Bop = (c1 * c2 * np.kron(np.eye(2), np.eye(2)) +
                 1j * c1 * s2 * np.kron(x, z) + 1j * s1 * c2 * np.kron(z, x) +
                 -1 * s1 * s2 * np.kron(y, y))
    assert np.allclose(Bop, trial_Bop)

    B = np.zeros((4, 4), dtype=np.complex128)
    b_decomp = cgot.operator_decomp(Bop)

    pauli_ops = [np.eye(2), x, y, z]
    num_qubits = 2
    operator = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    for idx, vec_index in enumerate(
            itertools.product(range(4), repeat=num_qubits)):
        op_basis = functools.reduce(np.kron,
                                    map(lambda x: pauli_ops[x], vec_index))
        operator += b_decomp[idx] * op_basis
    assert np.allclose(Bop, operator)

    B[0, 0] = c1 * c2
    B[1, 3] = 1j * s1 * c2
    B[3, 1] = 1j * c1 * s2
    B[2, 2] = -s1 * s2
    assert np.allclose(B, b_decomp.reshape((4, 4), order='F'))

    Cop = 0.5 * (zz @ Bop + Bop @ zz)
    c_decomp = cgot.operator_decomp(Cop)
    C = np.zeros((4, 4), dtype=np.complex128)
    C[1, 1] = s1 * s2
    C[3, 3] = c1 * c2
    assert np.allclose(c_decomp.reshape((4, 4), order='F'), C)

    Dop = zz @ Bop @ zz
    d_decomp = cgot.operator_decomp(Dop)
    D = np.zeros((4, 4), dtype=np.complex128)
    D[0, 0] = c1 * c2
    D[3, 1] = -1j * c1 * s2
    D[2, 2] = -1 * s1 * s2
    D[1, 3] = -1j * s1 * c2
    assert np.allclose(D, d_decomp.reshape((4, 4), order='F'))

    M = (np.cos(phi)**2) * B + 1j * np.sin(2 * phi) * C - (np.sin(phi)**2) * D

    bisycamore = scipy.linalg.expm(1j * phi * zz) @ cz @ np.kron(
        G2, G1) @ cz @ scipy.linalg.expm(1j * phi * zz)
    bisycamore_decomp = cgot.operator_decomp(bisycamore)
    assert np.allclose(bisycamore_decomp.reshape((4, 4), order='F'), M)


def test_decompose_cphase():
    np.set_printoptions(precision=10)
    phi = -np.pi / 24
    zz = np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))
    thetas = np.linspace(0, 2 * np.pi, 1000)
    for theta in thetas:
        u = scipy.linalg.expm(1j * theta * zz)
        u_decomp = cgot.operator_decomp(u)
        u_decomp_square = u_decomp.reshape((4, 4), order='F')
        u, s, _ = np.linalg.svd(u_decomp.reshape((4, 4), order='F'))
        assert np.allclose(
            sorted([abs(np.cos(theta)), abs(np.sin(theta))], reverse=True),
            s[:2])
        if abs(u_decomp_square[0, 0]) > np.cos(2 * phi):
            c2 = abs(np.sin(theta)) / np.cos(2 * phi)
        else:
            c2 = abs(np.cos(theta)) / np.cos(2 * phi)

        assert c2 <= 1.0
        eta = 0.5 - (0.5 * (np.cos(2 * phi)**2) * (c2**2))
        eta = eta.real
        s_trial = [
            abs(np.cos(2 * phi) * c2),
            np.sqrt(1 - (abs(np.cos(2 * phi) * c2))**2), 0, 0
        ]
        s_trial = np.array(sorted(s_trial, reverse=True))
        assert np.allclose(s_trial, s)


def test_zztheta():
    zz = np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))
    qubits = cirq.LineQubit.range(2)
    for theta in np.linspace(0, 2 * np.pi, 10):
        expected_unitary = scipy.linalg.expm(-1j * theta * zz)
        circuit = cirq.Circuit(cgot.rzz(theta, qubits[0], qubits[1]))
        actual_unitary = cirq.unitary(circuit)
        cirq.testing.assert_allclose_up_to_global_phase(actual_unitary,
                                                        expected_unitary,
                                                        atol=1e-7)


def test_zztheta_zzpow():
    qubits = cirq.LineQubit.range(2)
    for theta in np.linspace(0, 2 * np.pi, 10):
        syc_circuit = cirq.Circuit(cgot.rzz(theta, qubits[0], qubits[1]))
        cirq_circuit = cirq.Circuit([
            cirq.ZZPowGate(exponent=2 * theta / np.pi,
                           global_shift=-0.5).on(*qubits)
        ])
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(cirq_circuit), cirq.unitary(syc_circuit), atol=1e-7)


def test_zztheta_qaoa_like():

    class ConvertZZToSycamore(cirq.PointOptimizer):

        def optimization_at(self, circuit, index, op):
            if cirq.op_gate_of_type(op, cirq.ZZPowGate):
                return cirq.PointOptimizationSummary(
                    clear_span=1,
                    clear_qubits=op.qubits,
                    new_operations=cgot.rzz(theta=np.pi * op.gate.exponent / 2,
                                            q0=op.qubits[0],
                                            q1=op.qubits[1]))

    qubits = cirq.LineQubit.range(4)
    for exponent in np.linspace(-1, 1, 10):
        cirq_circuit = cirq.Circuit([
            cirq.H.on_each(qubits),
            cirq.ZZPowGate(exponent=exponent)(qubits[0], qubits[1]),
            cirq.ZZPowGate(exponent=exponent)(qubits[2], qubits[3]),
            cirq.Rx(.123).on_each(qubits),
        ])
        syc_circuit = cirq_circuit.copy()
        ConvertZZToSycamore().optimize_circuit(syc_circuit)

        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(cirq_circuit), cirq.unitary(syc_circuit), atol=1e-7)


def test_zztheta_zzpow_unsorted_qubits():

    qubits = cirq.LineQubit(1), cirq.LineQubit(0)
    exponent = 0.06366197723675814
    cirq_circuit = cirq.Circuit(
        cirq.ZZPowGate(exponent=exponent,
                       global_shift=-0.5).on(qubits[0], qubits[1]),)
    syc_circuit = cirq.Circuit(
        cgot.rzz(theta=np.pi * exponent / 2, q0=qubits[0], q1=qubits[1]))

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq_circuit),
                                                    cirq.unitary(syc_circuit),
                                                    atol=1e-7)


def test_swap_zztheta():
    """
    Construct a Ising gate followed by a swap using a sycamore.
    """
    qubits = cirq.LineQubit.range(2)
    a, b = qubits
    for theta in np.linspace(0, 2 * np.pi, 10):
        expected_circuit = cirq.Circuit(
            cirq.SWAP(a, b),
            cirq.ZZPowGate(exponent=2 * theta / np.pi,
                           global_shift=-0.5).on(a, b))
        expected_unitary = cirq.unitary(expected_circuit)
        actual_circuit = cirq.Circuit(cgot.swap_rzz(theta, a, b))
        actual_unitary = cirq.unitary(actual_circuit)
        cirq.testing.assert_allclose_up_to_global_phase(actual_unitary,
                                                        expected_unitary,
                                                        atol=1e-7)


def test_cphase():
    """
    Test cphase synthesis

    cphase(phi) = diag([1, 1, 1, exp(1j * phi)])
    """
    for phi in np.linspace(0, 2 * np.pi, 100):
        true_cphase = cirq.unitary(cirq.CZPowGate(exponent=phi / np.pi))
        assert np.allclose(np.diag([1, 1, 1, np.exp(1j * phi)]), true_cphase)
        test_cphase = cirq.unitary(
            cirq.Circuit(
                cgot.cphase(phi, cirq.NamedQubit('a'), cirq.NamedQubit('b'))))
        overlap = abs(np.trace(test_cphase.conj().T @ true_cphase))
        assert np.isclose(4.0, overlap)


def test_known_two_q_operations_to_sycamore_operations_cz():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.CZ(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(
        cgot.known_two_q_operations_to_sycamore_operations(
            qubits[0], qubits[1], operation))
    u1 = cirq.unitary(operation)
    u2 = cirq.unitary(test_op_tree)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0)
    _assert_equivalent_op_tree(test_op_tree,
                               cgot.decompose_cz_into_syc(qubits[0], qubits[1]))


def test_known_two_q_operations_to_sycamore_operations_cnot():
    a, b = cirq.LineQubit.range(2)
    op = cirq.CNOT(a, b)
    decomposed = cirq.Circuit(
        cgot.known_two_q_operations_to_sycamore_operations(a, b, op))

    # Should be equivalent.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(op),
                                                    cirq.unitary(decomposed),
                                                    atol=1e-8)

    # Should have decomposed into two Sycamores.
    multi_qubit_ops = [
        e for e in decomposed.all_operations() if len(e.qubits) > 1
    ]
    assert len(multi_qubit_ops) == 2
    assert all(
        cirq.op_gate_isinstance(e, cirq.google.SycamoreGate)
        for e in multi_qubit_ops)


def test_known_two_q_operations_to_sycamore_operations_cphase():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    for phi in np.linspace(0, 2 * np.pi, 30):
        operation = cirq.CZPowGate(exponent=phi / np.pi).on(
            qubits[0], qubits[1])
        test_op_tree = cirq.Circuit(
            cgot.known_two_q_operations_to_sycamore_operations(
                qubits[0], qubits[1], operation))
        true_op_tree = cirq.Circuit(cgot.cphase(phi, qubits[0], qubits[1]))
        u1 = cirq.unitary(operation)
        u2 = cirq.unitary(test_op_tree)
        overlap = abs(np.trace(u1.conj().T @ u2))
        assert np.isclose(overlap, 4.0)
        _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_known_two_q_operations_to_sycamore_operations_swap():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.SWAP(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(
        cgot.known_two_q_operations_to_sycamore_operations(
            qubits[0], qubits[1], operation))
    u1 = cirq.unitary(operation)
    u2 = cirq.unitary(test_op_tree)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0)
    _assert_equivalent_op_tree(
        test_op_tree, cgot.decompose_swap_into_syc(qubits[0], qubits[1]))


def test_known_two_q_operations_to_sycamore_operations_iswap():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.ISWAP(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(
        cgot.known_two_q_operations_to_sycamore_operations(
            qubits[0], qubits[1], operation))
    u1 = cirq.unitary(operation)
    u2 = cirq.unitary(test_op_tree)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0)
    _assert_equivalent_op_tree(
        test_op_tree,
        cirq.Circuit(cgot.decompose_iswap_into_syc(qubits[0], qubits[1])))


def test_known_two_q_operations_to_sycamore_operations_matrix():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.TwoQubitMatrixGate(cirq.unitary(cirq.CX)).on(
        qubits[0], qubits[1])
    _ = cirq.Circuit(
        cgot.known_two_q_operations_to_sycamore_operations(
            qubits[0], qubits[1], operation))
    u1 = cirq.unitary(operation)
    u2 = cirq.unitary(cirq.CX)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0)
