"""Cross entropy benchmarking example.

Cross entropy benchmarking is a method for assessing the performance of
gates by applying random circuits and measuring the cross entropy of observed
bitstring measurements versus the expected probabilities for these bitstrings
obtained from simulation.

See documentation in `cirq.experiments.cross_entropy_benchmarking` for
details of this experiments.
"""

import cirq


def main(repetitions=5000, num_circuits=20, cycles=range(2, 103, 10)):
    # The sampler to run the experiment.
    simulator = cirq.Simulator()

    # Specify 4 qubits on a 2 by 2 grid for the experiment.
    test_qubits = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
    ]

    # Builds the sequence of operations to be interleaved with random
    # single-qubit gates.
    interleaved_ops = cirq.experiments.build_entangling_layers(test_qubits, cirq.CZ)

    # Run the XEB experiment.
    xeb_result = cirq.experiments.cross_entropy_benchmarking(
        simulator,
        test_qubits,
        num_circuits=num_circuits,
        cycles=cycles,
        benchmark_ops=interleaved_ops,
        repetitions=repetitions,
    )

    # Plot XEB fidelity vs number of cycles.
    xeb_result.plot()


if __name__ == '__main__':
    main()
