import numpy as np

import cirq
import cirq.contrib.quimb as ccq


def test_tensor_density_matrix_1():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))

    rho1 = cirq.final_density_matrix(c, qubit_order=q, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(c, q)
    np.testing.assert_allclose(rho1, rho2, atol=1e-15)


def test_tensor_density_matrix_optional_qubits():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))

    rho1 = cirq.final_density_matrix(c, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(c)
    np.testing.assert_allclose(rho1, rho2, atol=1e-15)


def test_tensor_density_matrix_noise_1():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.YPowGate(exponent=0.25).on(q[0]),
        cirq.amplitude_damp(1e-2).on(q[0]),
        cirq.phase_damp(1e-3).on(q[0]),
    )

    rho1 = cirq.final_density_matrix(c, qubit_order=q, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(c, q)
    np.testing.assert_allclose(rho1, rho2, atol=1e-15)


def test_tensor_density_matrix_2():
    q = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    for _ in range(10):
        g = cirq.MatrixGate(cirq.testing.random_unitary(dim=2 ** len(q), random_state=rs))
        c = cirq.Circuit(g.on(*q))
        rho1 = cirq.final_density_matrix(c, dtype=np.complex128)
        rho2 = ccq.tensor_density_matrix(c, q)
        np.testing.assert_allclose(rho1, rho2, atol=1e-8)


def test_tensor_density_matrix_3():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    rho1 = cirq.final_density_matrix(circuit, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(circuit, qubits)
    np.testing.assert_allclose(rho1, rho2, atol=1e-8)


def test_tensor_density_matrix_4():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=100, op_density=0.8)
    cirq.DropEmptyMoments().optimize_circuit(circuit)
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=1e-3))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, qubits))
    rho1 = cirq.final_density_matrix(circuit, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(circuit, qubits)
    np.testing.assert_allclose(rho1, rho2, atol=1e-8)


def test_tensor_density_matrix_gridqubit():
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    cirq.DropEmptyMoments().optimize_circuit(circuit)
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=1e-3))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, qubits))
    rho1 = cirq.final_density_matrix(circuit, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(circuit, qubits)
    np.testing.assert_allclose(rho1, rho2, atol=1e-8)
