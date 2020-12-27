import numpy as np
import pytest

import cirq


def assert_same_output_as_dense(circuit, qubit_order, initial_state=0):
    mps_simulator = cirq.MPSSimulator()
    ref_simulator = cirq.Simulator()

    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    np.testing.assert_almost_equal(actual.final_state.to_numpy(), expected.final_state_vector)
    assert len(actual.measurements) == 0


def test_various_gates():
    gate_cls = [cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z]
    cross_gate_cls = [cirq.CNOT, cirq.SWAP]

    for q0_gate in gate_cls:
        for q1_gate in gate_cls:
            for cross_gate in cross_gate_cls:
                q0, q1 = cirq.LineQubit.range(2)
                circuit = cirq.Circuit(q0_gate(q0), q1_gate(q1), cross_gate(q0, q1))
                assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1])


def test_empty():
    q0 = cirq.NamedQid('q0', dimension=2)
    q1 = cirq.NamedQid('q1', dimension=3)
    q2 = cirq.NamedQid('q2', dimension=5)
    circuit = cirq.Circuit()

    for initial_state in range(2 * 3 * 5):
        assert_same_output_as_dense(
            circuit=circuit, qubit_order=[q0, q1, q2], initial_state=initial_state
        )


def test_cnot():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))

    for initial_state in range(4):
        assert_same_output_as_dense(
            circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state
        )


def test_cnot_flipped():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))

    for initial_state in range(4):
        assert_same_output_as_dense(
            circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state
        )


def test_jump_two():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CNOT(q0, q2))

    with pytest.raises(ValueError, match="Can only handle continguous qubits"):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2])


def test_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.H(q1), cirq.measure(q1))

    simulator = cirq.MPSSimulator()

    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['1'])[0] < 80
    assert sum(result.measurements['1'])[0] > 20
