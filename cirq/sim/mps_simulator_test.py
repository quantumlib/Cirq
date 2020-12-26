import numpy as np

import cirq


def assert_same_output_as_dense(circuit, qubit_order):
    mps_simulator = cirq.MPSSimulator()
    ref_simulator = cirq.Simulator()

    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order)
    np.testing.assert_almost_equal(actual.final_state.to_numpy(), expected.final_state_vector)
    assert len(actual.measurements) == 0


def test_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1])
