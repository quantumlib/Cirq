

import cirq


def test_pretty_state():
    simulator = cirq.google.XmonSimulator()
    c = cirq.Circuit()
    one_qubit = [Q1]
    two_qubits = [Q1, Q2]

    # Testing global pretty_state()
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
    result = simulator.simulate(c, qubit_order=one_qubit, initial_state=state)
    assert pretty_state(result.final_state) == "0.71|0⟩ + 0.71|1⟩"

    # Testing pretty_state() method in XmonStepResult
    circuit = basic_circuit()
    step = simulator.simulate_moment_steps(circuit)
    result = next(step)
    result.set_state(0)
    assert result.pretty_state() == "1.0|00⟩"

    # Testing pretty_state() method in XmonSimulateTrialResult
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex64)
    result = simulator.simulate(c, qubit_order=two_qubits, initial_state=state)
    assert result.pretty_state(decimals=1) == "0.7|00⟩ + 0.7|11⟩"

    state = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=np.complex64)
    result = simulator.simulate(c, qubit_order=two_qubits, initial_state=state)
    assert result.pretty_state(decimals=2) == "0.71|00⟩ + -0.71|11⟩"

    state = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.complex64)
    result = simulator.simulate(c, qubit_order=two_qubits, initial_state=state)
    assert result.pretty_state(decimals=2) == "0.71|01⟩ + 0.71|10⟩"

    state = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=np.complex64)
    result = simulator.simulate(
        cirq.Circuit(), qubit_order=two_qubits, initial_state=state)
    assert result.pretty_state(decimals=4) == "0.7071|01⟩ + -0.7071|10⟩"
