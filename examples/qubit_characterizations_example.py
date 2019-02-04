from matplotlib import pyplot

import cirq


def main():
    # The device to run the experiment.
    simulator = cirq.Simulator()

    # The two qubits to be characterized in this example.
    q_0 = cirq.GridQubit(0, 0)
    q_1 = cirq.GridQubit(0, 1)

    # Measure Rabi oscillation of q_0.
    rabi_results = cirq.experiments.rabi_oscillations(simulator, q_0, 4,
                                                      1000, 200)
    rabi_results.plot()

    # Clifford-based randomized benchmarking of single-qubit gates on q_0.
    n_cfs_1q = range(10, 100, 10)
    rb_result_1q = cirq.experiments.single_qubit_randomized_benchmarking(
        simulator, q_0, n_cfs_1q, 20, 1000)
    rb_result_1q.plot()

    # Clifford-based randomized benchmarking of two-qubit gates on q_0 and q_1.
    n_cfs_2q = range(5, 50, 5)
    rb_result_2q = cirq.experiments.two_qubit_randomized_benchmarking(
        simulator, q_0, q_1, n_cfs_2q, 20, 1000)
    rb_result_2q.plot()

    # State-tomography of q_0 after application of an X/2 rotation.
    cir_1q = cirq.Circuit()
    cir_1q.append(cirq.X(q_0) ** 0.5)
    tomography_1q = cirq.experiments.single_qubit_state_tomography(simulator,
                                                                   q_0,
                                                                   cir_1q, 1000)
    tomography_1q.plot()

    # State-tomography of a Bell state between q_0 and q_1.
    cir_2q = cirq.Circuit()
    cir_2q.append(cirq.H(q_0))
    cir_2q.append(cirq.CNOT(q_0, q_1))
    tomography_2q = cirq.experiments.two_qubit_state_tomography(simulator,
                                                                q_0, q_1,
                                                                cir_2q, 1000)
    tomography_2q.plot()

    pyplot.show()


if __name__ == '__main__':
    main()
