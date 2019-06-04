import cirq


def main():
    # The device to run the experiment.
    simulator = cirq.Simulator()

    # Specify 4 qubits on a 2 by 2 grid for the experiment.
    test_qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1),
                   cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]

    # Number of measurements at the end of each circuit for estimating
    # probabilities.
    num_measurements = 5000

    # Run the XEB experiment.
    xeb_result = cirq.experiments.cross_entropy_benchmarking(
        simulator, test_qubits, cirq.CZ, repetitions=num_measurements)

    # Plot XEB fidelity vs number of cycles.
    xeb_result.plot()


if __name__ == '__main__':
    main()
