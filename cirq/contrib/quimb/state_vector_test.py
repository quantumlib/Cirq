import functools
import operator

import numpy as np
import pytest

import cirq
import cirq.contrib.quimb as ccq


def test_tensor_state_vector_1():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))

    psi1 = cirq.final_state_vector(c, qubit_order=q, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(c, q)
    np.testing.assert_allclose(psi1, psi2, atol=1e-15)


def test_tensor_state_vector_implicit_qubits():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))

    psi1 = cirq.final_state_vector(c, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(c)
    np.testing.assert_allclose(psi1, psi2, atol=1e-15)


def test_tensor_state_vector_2():
    q = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    for _ in range(10):
        g = cirq.MatrixGate(cirq.testing.random_unitary(dim=2 ** len(q), random_state=rs))
        c = cirq.Circuit(g.on(*q))
        psi1 = cirq.final_state_vector(c, dtype=np.complex128)
        psi2 = ccq.tensor_state_vector(c, q)
        np.testing.assert_allclose(psi1, psi2, atol=1e-8)


def test_tensor_state_vector_3():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    psi1 = cirq.final_state_vector(circuit, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(circuit, qubits)
    np.testing.assert_allclose(psi1, psi2, atol=1e-8)


def test_tensor_state_vector_4():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=100, op_density=0.8)
    psi1 = cirq.final_state_vector(circuit, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(circuit, qubits)
    np.testing.assert_allclose(psi1, psi2, atol=1e-8)


def test_sandwich_operator_identity():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    tot_c = ccq.circuit_for_expectation_value(circuit, cirq.PauliString({}))
    np.testing.assert_allclose(cirq.unitary(tot_c), np.eye(2 ** len(qubits)), atol=1e-6)


def _random_pauli_string(qubits, rs, coefficients=False):
    ps = cirq.PauliString(
        {q: p}
        for q, p in zip(qubits, rs.choice([cirq.X, cirq.Y, cirq.Z, cirq.I], size=len(qubits)))
    )
    if coefficients:
        return rs.uniform(-1, 1) * ps
    return ps


def test_sandwich_operator_expect_val():
    rs = np.random.RandomState(52)
    qubits = cirq.LineQubit.range(5)
    for _ in range(10):  # try a bunch of different ones
        circuit = cirq.testing.random_circuit(
            qubits=qubits, n_moments=10, op_density=0.8, random_state=rs
        )
        operator = _random_pauli_string(qubits, rs)
        tot_c = ccq.circuit_for_expectation_value(circuit, operator)
        eval_sandwich = cirq.unitary(tot_c)[0, 0]
        wfn = cirq.Simulator().simulate(circuit)
        eval_normal = operator.expectation_from_wavefunction(wfn.final_state, wfn.qubit_map)
        np.testing.assert_allclose(eval_sandwich, eval_normal, atol=1e-5)


def test_tensor_unitary():
    rs = np.random.RandomState(52)
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.testing.random_circuit(
            qubits=qubits, n_moments=10, op_density=0.8, random_state=rs
        )
        operator = _random_pauli_string(qubits, rs)

        circuit_sand = ccq.circuit_for_expectation_value(circuit, operator)
        u_tn = ccq.tensor_unitary(circuit_sand, qubits)
        u_cirq = cirq.unitary(circuit_sand)
        np.testing.assert_allclose(u_tn, u_cirq, atol=1e-6)


def test_tensor_unitary_implicit_qubits():
    rs = np.random.RandomState(52)
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.testing.random_circuit(
            qubits=qubits, n_moments=10, op_density=0.8, random_state=rs
        )
        operator = _random_pauli_string(qubits, rs)

        circuit_sand = ccq.circuit_for_expectation_value(circuit, operator)
        u_tn = ccq.tensor_unitary(circuit_sand)
        u_cirq = cirq.unitary(circuit_sand)
        np.testing.assert_allclose(u_tn, u_cirq, atol=1e-6)


def test_tensor_expectation_value():
    rs = np.random.RandomState(52)
    for _ in range(10):
        for n_qubit in [2, 7]:
            qubits = cirq.LineQubit.range(n_qubit)
            for depth in [10, 20]:
                circuit = cirq.testing.random_circuit(
                    qubits=qubits, n_moments=depth, op_density=0.8, random_state=rs
                )
                operator = _random_pauli_string(qubits, rs, coefficients=True)
                eval_tn = ccq.tensor_expectation_value(circuit, operator)

                wfn = cirq.Simulator().simulate(circuit)
                eval_normal = operator.expectation_from_wavefunction(wfn.final_state, wfn.qubit_map)
                assert eval_normal.imag < 1e-6
                eval_normal = eval_normal.real
                np.testing.assert_allclose(eval_tn, eval_normal, atol=1e-3)


def test_bad_init_state():
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    with pytest.raises(ValueError):
        ccq.circuit_to_tensors(circuit=circuit, qubits=qubits, initial_state=1)


def test_too_much_ram():
    qubits = cirq.LineQubit.range(30)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=20, op_density=0.8)
    op = functools.reduce(operator.mul, [cirq.Z(q) for q in qubits], 1)
    with pytest.raises(MemoryError) as e:
        ccq.tensor_expectation_value(circuit=circuit, pauli_string=op)

    assert e.match(r'.*too much RAM!.*')
