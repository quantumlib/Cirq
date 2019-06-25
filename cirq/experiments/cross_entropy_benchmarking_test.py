import numpy as np

from cirq import ops, sim, devices
from cirq.experiments import cross_entropy_benchmarking, build_entangling_layers


def test_cross_entropy_benchmarking():
    # Check that the fidelities returned from a four-qubit XEB simulation are
    # close to 1 (deviations from 1 is expected due to finite number of
    # measurements).
    simulator = sim.Simulator()
    qubits = [devices.GridQubit(0, 0), devices.GridQubit(0, 1),
              devices.GridQubit(1, 0), devices.GridQubit(1, 1)]

    # Build a sequence of CZ gates.
    interleaved_ops = build_entangling_layers(qubits, ops.CZ)

    # Specify a set of \pi and \pi/2 single-qubit rotations.
    single_qubit_rots = [[ops.X ** 0.5], [ops.Y ** 0.5, ops.X ** 0.5],
                         [ops.Y, ops.X], [ops.Y ** 0.5]]

    # Simulate XEB using the default single-qubit gate set without two-qubit
    # gates, XEB using the specified single-qubit gate set without two-qubit
    # gates, and XEB using the specified single-qubit gate set with two-qubit
    # gate. Check that the fidelities are close to 1.0 in all cases. Also,
    # check that a single XEB fidelity is returned if a single cycle number
    # is specified.
    results_0 = cross_entropy_benchmarking(
        simulator, qubits, num_circuits=5, repetitions=5000,
        cycles=range(2, 30, 5))
    results_1 = cross_entropy_benchmarking(
        simulator, qubits, num_circuits=5, repetitions=5000,
        cycles=range(2, 30, 5), single_qubit_gates=single_qubit_rots)
    results_2 = cross_entropy_benchmarking(
        simulator, qubits, benchmark_ops=interleaved_ops,
        num_circuits=5, repetitions=5000,
        cycles=range(2, 30, 5), single_qubit_gates=single_qubit_rots)
    results_3 = cross_entropy_benchmarking(
        simulator, qubits, benchmark_ops=interleaved_ops,
        num_circuits=5, repetitions=5000,
        cycles=20, single_qubit_gates=single_qubit_rots)
    fidelities_0 = [datum.xeb_fidelity for datum in results_0.data]
    fidelities_1 = [datum.xeb_fidelity for datum in results_1.data]
    fidelities_2 = [datum.xeb_fidelity for datum in results_2.data]
    fidelities_3 = [datum.xeb_fidelity for datum in results_3.data]
    assert np.isclose(np.mean(fidelities_0), 1.0, atol=0.1)
    assert np.isclose(np.mean(fidelities_1), 1.0, atol=0.1)
    assert np.isclose(np.mean(fidelities_2), 1.0, atol=0.1)
    assert len(fidelities_3) == 1
