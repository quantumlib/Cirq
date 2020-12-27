import numpy as np

import cirq


def assert_same_output_as_dense(circuit, qubit_order, initial_state=0):
    mps_simulator = cirq.MPSSimulator()
    ref_simulator = cirq.Simulator()

    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    np.testing.assert_almost_equal(actual.final_state.to_numpy(), expected.final_state_vector)
    assert len(actual.measurements) == 0


def test_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.CNOT(q0, q1))
    assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1])


def test_empty():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()

    for initial_state in range(4):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state)


def test_cnot():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))

    for initial_state in range(4):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state)


def test_cnot_flipped():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))

    for initial_state in range(4):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state)
