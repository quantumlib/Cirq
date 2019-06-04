import numpy as np

from cirq import ops, sim, devices
from cirq.experiments import cross_entropy_benchmarking


def test_cross_entropy_benchmarking():
    # Check that the fidelities returned from a four-qubit XEB simulation are
    # close to 1 (deviations from 1 is expected due to finite number of
    # measurements).
    simulator = sim.Simulator()
    qubits = [devices.GridQubit(0, 0), devices.GridQubit(0, 1),
              devices.GridQubit(1, 0), devices.GridQubit(1, 1)]

    # Simulate XEB using two different choices of single-qubit gate-set.
    results_0 = cross_entropy_benchmarking(
        simulator, qubits, ops.CZ, num_circuits=5, repetitions=5000,
        num_cycle_range=range(2, 30, 5))
    results_1 = cross_entropy_benchmarking(
        simulator, qubits, ops.CZ, num_circuits=5, repetitions=5000,
        num_cycle_range=range(2, 30, 5), use_tetrahedral_group=True)
    fidelities_0 = np.asarray(results_0.data)[:, 1]
    fidelities_1 = np.asarray(results_1.data)[:, 1]
    assert np.isclose(np.mean(fidelities_0), 1.0, atol=0.1)
    assert np.isclose(np.mean(fidelities_1), 1.0, atol=0.1)
