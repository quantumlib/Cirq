import numpy as np
import cirq


def main():
    # The device to run the experiment.
    simulator = cirq.Simulator()

    # The two qubits to be characterized in this example.
    q_0 = cirq.GridQubit(0, 0)
    q_1 = cirq.GridQubit(0, 1)

    # Measure Rabi oscillation of q_0.
    rabi_results = cirq.experiments.rabi_oscillations(simulator, q_0, 4 * np.pi)
    rabi_results.plot()

    num_cfds = range(5, 20, 5)

    # Clifford-based randomized benchmarking of single-qubit gates on q_0.
    rb_result_1q = cirq.experiments.single_qubit_randomized_benchmarking(
        simulator, q_0, num_clifford_range=num_cfds, repetitions=100)
    rb_result_1q.plot()

    # Clifford-based randomized benchmarking of two-qubit gates on q_0 and q_1.
    rb_result_2q = cirq.experiments.two_qubit_randomized_benchmarking(
        simulator, q_0, q_1, num_clifford_range=num_cfds, repetitions=100)
    rb_result_2q.plot()

    # State-tomography of q_0 after application of an X/2 rotation.
    cir_1q = cirq.Circuit(cirq.X(q_0)**0.5)
    tomography_1q = cirq.experiments.single_qubit_state_tomography(simulator,
                                                                   q_0, cir_1q)
    tomography_1q.plot()

    # State-tomography of a Bell state between q_0 and q_1.
    cir_2q = cirq.Circuit(cirq.H(q_0), cirq.CNOT(q_0, q_1))
    tomography_2q = cirq.experiments.two_qubit_state_tomography(simulator,
                                                                q_0, q_1,
                                                                cir_2q)
    tomography_2q.plot()


if __name__ == '__main__':
    main()
