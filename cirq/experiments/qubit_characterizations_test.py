import numpy as np

from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (rabi_oscillations,
                              single_qubit_randomized_benchmarking,
                              two_qubit_randomized_benchmarking,
                              single_qubit_state_tomography,
                              two_qubit_state_tomography)


def test_rabi_oscillations():
    # Check that the excited state population matches the ideal case within a
    # small statistical error.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    results = rabi_oscillations(simulator, qubit, np.pi, repetitions=10000)
    data = np.asarray(results.data)
    angles = data[:, 0]
    actual_pops = data[:, 1]
    target_pops = 0.5 - 0.5 * np.cos(angles)
    rms_err = np.sqrt(np.mean((target_pops - actual_pops) ** 2))
    assert rms_err < 0.1


def test_single_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    num_cfds = range(5, 20, 5)
    results = single_qubit_randomized_benchmarking(simulator, qubit,
                                                   num_clifford_range=num_cfds)
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)


def test_two_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)
    num_cfds = range(5, 20, 5)
    results = two_qubit_randomized_benchmarking(simulator, q_0, q_1,
                                                num_clifford_range=num_cfds)
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)


def test_single_qubit_state_tomography():
    # Check that the density matrices of the output states of X/2, Y/2 and
    # H + Y gates closely match the ideal cases.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)

    circuit_1 = circuits.Circuit.from_ops(ops.X(qubit) ** 0.5)
    circuit_2 = circuits.Circuit.from_ops(ops.Y(qubit) ** 0.5)
    circuit_3 = circuits.Circuit.from_ops(ops.H(qubit), ops.Y(qubit))

    act_rho_1 = single_qubit_state_tomography(simulator, qubit, circuit_1,
                                              100000).data
    act_rho_2 = single_qubit_state_tomography(simulator, qubit, circuit_2,
                                              100000).data
    act_rho_3 = single_qubit_state_tomography(simulator, qubit, circuit_3,
                                              100000).data

    tar_rho_1 = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    tar_rho_2 = np.array([[0.5, 0.5], [0.5, 0.5]])
    tar_rho_3 = np.array([[0.5, -0.5], [-0.5, 0.5]])

    np.testing.assert_almost_equal(act_rho_1, tar_rho_1, decimal=1)
    np.testing.assert_almost_equal(act_rho_2, tar_rho_2, decimal=1)
    np.testing.assert_almost_equal(act_rho_3, tar_rho_3, decimal=1)


def test_two_qubit_state_tomography():
    # Check that the density matrices of the four Bell states closely match
    # the ideal cases.
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)

    circuit_00 = circuits.Circuit.from_ops(ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_01 = circuits.Circuit.from_ops(ops.X(q_1), ops.H(q_0),
                                           ops.CNOT(q_0, q_1))
    circuit_10 = circuits.Circuit.from_ops(ops.X(q_0), ops.H(q_0),
                                           ops.CNOT(q_0, q_1))
    circuit_11 = circuits.Circuit.from_ops(ops.X(q_0), ops.X(q_1), ops.H(q_0),
                                           ops.CNOT(q_0, q_1))

    act_rho_00 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_00,
                                            100000).data
    act_rho_01 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_01,
                                            100000).data
    act_rho_10 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_10,
                                            100000).data
    act_rho_11 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_11,
                                            100000).data

    tar_rho_00 = np.outer([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]) / 2.0
    tar_rho_01 = np.outer([0, 1.0, 1.0, 0], [0, 1.0, 1.0, 0]) / 2.0
    tar_rho_10 = np.outer([1.0, 0, 0, -1.0], [1.0, 0, 0, -1.0]) / 2.0
    tar_rho_11 = np.outer([0, 1.0, -1.0, 0], [0, 1.0, -1.0, 0]) / 2.0

    np.testing.assert_almost_equal(act_rho_00, tar_rho_00, decimal=1)
    np.testing.assert_almost_equal(act_rho_01, tar_rho_01, decimal=1)
    np.testing.assert_almost_equal(act_rho_10, tar_rho_10, decimal=1)
    np.testing.assert_almost_equal(act_rho_11, tar_rho_11, decimal=1)
