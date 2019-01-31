import numpy

from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import rabi_oscillations, \
    single_qubit_randomized_benchmarking, two_qubit_randomized_benchmarking, \
    single_qubit_state_tomography, two_qubit_state_tomography


def test_rabi_oscillations():
    # Check that the excited state population matches the ideal case within a
    # small statistical error.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    angles, actual_pops = rabi_oscillations(simulator, qubit, 1.0, 1000, 10,
                                            plot=False)
    target_pops = 0.5 - 0.5 * numpy.cos(angles * numpy.pi)
    rms_err = numpy.sqrt(numpy.mean((target_pops - actual_pops) ** 2))
    assert rms_err < 0.1


def test_single_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    num_cfds = range(5, 20, 5)
    g_pops = single_qubit_randomized_benchmarking(simulator, qubit, num_cfds,
                                                  10, 100, plot=False)
    assert numpy.mean(g_pops) == 1.0


def test_two_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)
    num_cfds = range(5, 20, 5)
    g_pops = two_qubit_randomized_benchmarking(simulator, q_0, q_1, num_cfds,
                                               10, 1000, plot=False)
    assert numpy.mean(g_pops) == 1.0


def test_single_qubit_state_tomography():
    # Check that the density matrices of the output states of X/2, Y/2 and
    # H + Y gates closely match the ideal cases.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)

    circuit_1 = circuits.Circuit()
    circuit_2 = circuits.Circuit()
    circuit_3 = circuits.Circuit()
    circuit_1.append(ops.XPowGate(exponent=0.5)(qubit))
    circuit_2.append(ops.YPowGate(exponent=0.5)(qubit))
    circuit_3.append(ops.H(qubit))
    circuit_3.append(ops.Y(qubit))

    act_rho_1 = single_qubit_state_tomography(simulator, qubit, circuit_1,
                                              10000, False)
    act_rho_2 = single_qubit_state_tomography(simulator, qubit, circuit_2,
                                              10000, False)
    act_rho_3 = single_qubit_state_tomography(simulator, qubit, circuit_3,
                                              10000, False)

    tar_rho_1 = numpy.array([[0.5, 0.5j], [-0.5j, 0.5]])
    tar_rho_2 = numpy.array([[0.5, 0.5], [0.5, 0.5]])
    tar_rho_3 = numpy.array([[0.5, -0.5], [-0.5, 0.5]])

    numpy.testing.assert_almost_equal(act_rho_1, tar_rho_1, decimal=1)
    numpy.testing.assert_almost_equal(act_rho_2, tar_rho_2, decimal=1)
    numpy.testing.assert_almost_equal(act_rho_3, tar_rho_3, decimal=1)


def test_two_qubit_state_tomography():
    # Check that the density matrices of the four Bell states closely match
    # the ideal cases.
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)

    circuit_00 = circuits.Circuit()
    circuit_00.append(ops.H(q_0))
    circuit_00.append(ops.CNOT(q_0, q_1))

    circuit_01 = circuits.Circuit()
    circuit_01.append(ops.X(q_1))
    circuit_01.append(ops.H(q_0))
    circuit_01.append(ops.CNOT(q_0, q_1))

    circuit_10 = circuits.Circuit()
    circuit_10.append(ops.X(q_0))
    circuit_10.append(ops.H(q_0))
    circuit_10.append(ops.CNOT(q_0, q_1))

    circuit_11 = circuits.Circuit()
    circuit_11.append(ops.X(q_0))
    circuit_11.append(ops.X(q_1))
    circuit_11.append(ops.H(q_0))
    circuit_11.append(ops.CNOT(q_0, q_1))

    act_rho_00 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_00,
                                            10000, False)
    act_rho_01 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_01,
                                            10000, False)
    act_rho_10 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_10,
                                            10000, False)
    act_rho_11 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_11,
                                            10000, False)

    tar_rho_00 = numpy.outer([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]) / 2.0
    tar_rho_01 = numpy.outer([0, 1.0, 1.0, 0], [0, 1.0, 1.0, 0]) / 2.0
    tar_rho_10 = numpy.outer([1.0, 0, 0, -1.0], [1.0, 0, 0, -1.0]) / 2.0
    tar_rho_11 = numpy.outer([0, 1.0, -1.0, 0], [0, 1.0, -1.0, 0]) / 2.0

    numpy.testing.assert_almost_equal(act_rho_00, tar_rho_00, decimal=1)
    numpy.testing.assert_almost_equal(act_rho_01, tar_rho_01, decimal=1)
    numpy.testing.assert_almost_equal(act_rho_10, tar_rho_10, decimal=1)
    numpy.testing.assert_almost_equal(act_rho_11, tar_rho_11, decimal=1)
