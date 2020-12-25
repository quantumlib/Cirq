import numpy as np

import cirq


def test_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.MPSSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(), np.array([0.5, 0.5, 0.5, 0.5]))
    assert len(result.measurements) == 0
