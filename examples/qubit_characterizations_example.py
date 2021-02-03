import numpy as np
import cirq


def main(minimum_cliffords=5, maximum_cliffords=20, cliffords_step=5):
    """
    Examples of how to run various qubit characterizations.  This example
    shows various methods on how to characterize a qubit, including a Rabi
    oscillation experiments, Clifford-based randomized benchmarking, and
    state tomography.

    The number of cliffords to use in a randomized benchmarking experiment
    can be varied.  For instance, setting minimum_cliffords=10,
    maximum_cliffords=30 and cliffords_step=5 will test depths of 10, 15, 20,
    and 25 (the maximum of the range is exclusive).

    Args:
        minimum_cliffords: the number of Clifford gates to start with in a
          randomized benchmarking study
        maximum_cliffords: the number of Clifford gates to scale up to in a
          randomized benchmarking study.  This is used as an exclusive limit.
        cliffords_step: the increment to step with from the minimum to maximum
          number of Clifford gates.
    """
    # The device to run the experiment.
    simulator = cirq.Simulator()

    # The two qubits to be characterized in this example.
    q_0 = cirq.GridQubit(0, 0)
    q_1 = cirq.GridQubit(0, 1)

    # Measure Rabi oscillation of q_0.
    rabi_results = cirq.experiments.rabi_oscillations(simulator, q_0, 4 * np.pi)
    rabi_results.plot()

    clifford_range = range(minimum_cliffords, maximum_cliffords, cliffords_step)

    # Clifford-based randomized benchmarking of single-qubit gates on q_0.
    rb_result_1q = cirq.experiments.single_qubit_randomized_benchmarking(
        simulator, q_0, num_clifford_range=clifford_range, repetitions=100
    )
    rb_result_1q.plot()

    # Clifford-based randomized benchmarking of two-qubit gates on q_0 and q_1.
    rb_result_2q = cirq.experiments.two_qubit_randomized_benchmarking(
        simulator, q_0, q_1, num_clifford_range=clifford_range, repetitions=100
    )
    rb_result_2q.plot()

    # State-tomography of q_0 after application of an X/2 rotation.
    cir_1q = cirq.Circuit(cirq.X(q_0) ** 0.5)
    tomography_1q = cirq.experiments.single_qubit_state_tomography(simulator, q_0, cir_1q)
    tomography_1q.plot()

    # State-tomography of a Bell state between q_0 and q_1.
    cir_2q = cirq.Circuit(cirq.H(q_0), cirq.CNOT(q_0, q_1))
    tomography_2q = cirq.experiments.two_qubit_state_tomography(simulator, q_0, q_1, cir_2q)
    tomography_2q.plot()


if __name__ == '__main__':
    main()
