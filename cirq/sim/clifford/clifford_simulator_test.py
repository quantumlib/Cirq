import numpy as np
import pytest

import cirq


def test_simulate_no_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(),
                                   np.array([1, 0, 0, 0]))
    assert len(result.measurements) == 0


def test_run_hadamard():
    q0 = cirq.LineQubit(0)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


def test_run_GHZ():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


def test_run_correlations():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]


def test_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(),
                                   np.array([0.5, 0.5, 0.5, 0.5]))
    assert len(result.measurements) == 0


def test_simulate_initial_state():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit()
            if b0:
                circuit.append(cirq.X(q0))
            if b1:
                circuit.append(cirq.X(q1))
            circuit.append(cirq.measure(q0, q1))

            result = simulator.simulate(circuit, initial_state=1)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][1 - b1] = 1.0
            np.testing.assert_almost_equal(result.final_state.to_numpy(),
                                           np.reshape(expected_state, 4))


def test_simulate_qubit_order():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit()
            if b0:
                circuit.append(cirq.X(q0))
            if b1:
                circuit.append(cirq.X(q1))
            circuit.append(cirq.measure(q0, q1))

            result = simulator.simulate(circuit, qubit_order=[q1, q0])
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b1][b0] = 1.0
            np.testing.assert_almost_equal(result.final_state.to_numpy(),
                                           np.reshape(expected_state, 4))


def test_run_measure_multiple_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit()
            if b0:
                circuit.append(cirq.X(q0))
            if b1:
                circuit.append(cirq.X(q1))
            circuit.append(cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [[b0, b1]] * 3})


def test_simulate_moment_steps():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q0), cirq.H(q1))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step.state.to_numpy(),
                                           np.array([0.5] * 4))
        else:
            np.testing.assert_almost_equal(step.state.to_numpy(),
                                           np.array([1, 0, 0, 0]))


def test_simulate_moment_steps_sample():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, False]) or
                        np.array_equal(sample, [False, False]))
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, True]) or
                        np.array_equal(sample, [False, False]))


def test_simulate_moment_steps_intermediate_measurement():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0), cirq.H(q0))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 1:
            result = int(step.measurements['0'][0])
            expected = np.zeros(2)
            expected[result] = 1
            np.testing.assert_almost_equal(step.state.to_numpy(), expected)
        if i == 2:
            expected = np.array([np.sqrt(0.5), np.sqrt(0.5) * (-1)**result])
            np.testing.assert_almost_equal(step.state.to_numpy(), expected)


def test_clifford_trial_result_repr():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.CliffordState(qubit_map={q0: 0})
    assert (repr(
        cirq.CliffordTrialResult(params=cirq.ParamResolver({}),
                                 measurements={'m': np.array([[1]])},
                                 final_simulator_state=final_simulator_state))
            == "cirq.SimulationTrialResult(params=cirq.ParamResolver({}), "
            "measurements={'m': array([[1]])}, "
            "final_simulator_state=StabilizerStateChForm(num_qubits=1, "
            "initial_state=0))")


def test_clifford_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.CliffordState(qubit_map={q0: 0})
    assert (str(
        cirq.CliffordTrialResult(params=cirq.ParamResolver({}),
                                 measurements={'m': np.array([[1]])},
                                 final_simulator_state=final_simulator_state))
            == "measurements: m=1\n"
            "output state: |0⟩")


def test_clifford_step_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.CliffordState(qubit_map={q0: 0})

    assert (str(
        cirq.CliffordSimulatorStepResult(
            measurements={'m': np.array([[1]])},
            state=final_simulator_state)) == "m=1\n"
            "|0⟩")


def test_clifford_step_result_no_measurements_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.CliffordState(qubit_map={q0: 0})

    assert (str(
        cirq.CliffordSimulatorStepResult(measurements={},
                                         state=final_simulator_state)) == "|0⟩")


def test_clifford_state_str():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})

    assert (str(state) == "|00⟩")


def test_clifford_state_stabilizers():
    (q0, q1, q2) = (cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1, q2: 2})
    state.apply_unitary(cirq.H(q0))
    state.apply_unitary(cirq.H(q1))
    state.apply_unitary(cirq.S(q1))

    f = cirq.DensePauliString
    assert (state.stabilizers() == [f('XII'), f('IYI'), f('IIZ')])
    assert (state.destabilizers() == [f('ZII'), f('IZI'), f('IIX')])


def test_clifford_state_wave_function():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})

    np.testing.assert_equal(state.wave_function(),
                            [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j])


def test_clifford_tableau_str():
    (q0, q1, q2) = (cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1, q2: 2})
    state.apply_unitary(cirq.H(q0))
    state.apply_unitary(cirq.H(q1))
    state.apply_unitary(cirq.S(q1))

    assert (str(state.tableau) == "+ X I I \n+ I Y I \n+ I I Z ")


def test_clifford_tableau_repr():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    f = cirq.DensePauliString
    assert (repr(state.tableau) == "stabilizers: [{!r}, {!r}]".format(
        f("ZI"), f("IZ")))


def test_clifford_tableau_str_full():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    state.apply_unitary(cirq.H(q0))
    state.apply_unitary(cirq.S(q0))

    assert (state.tableau._str_full_() == "stable | destable\n"
            "-------+----------\n"
            "+ Y0   | + Z0  \n"
            "+   Z1 | +   X1\n")


def test_stabilizerStateChForm_H():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    with pytest.raises(ValueError, match="|y> is equal to |z>"):
        state.ch_form._H_decompose(0, 1, 1, 0)


def test_clifford_stabilizerStateChForm_repr():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    assert repr(state) == 'StabilizerStateChForm(num_qubits=2, initial_state=0)'


def test_clifford_circuit():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    circuit = cirq.Circuit()

    np.random.seed(0)

    for _ in range(100):
        x = np.random.randint(7)

        if x == 0:
            circuit.append(cirq.X(np.random.choice((q0, q1))))
        elif x == 1:
            circuit.append(cirq.Z(np.random.choice((q0, q1))))
        elif x == 2:
            circuit.append(cirq.Y(np.random.choice((q0, q1))))
        elif x == 3:
            circuit.append(cirq.S(np.random.choice((q0, q1))))
        elif x == 4:
            circuit.append(cirq.H(np.random.choice((q0, q1))))
        elif x == 5:
            circuit.append(cirq.CNOT(q0, q1))
        elif x == 6:
            circuit.append(cirq.CZ(q0, q1))

    clifford_simulator = cirq.CliffordSimulator()
    wave_function_simulator = cirq.Simulator()

    np.testing.assert_almost_equal(
        clifford_simulator.simulate(circuit).final_state.wave_function(),
        wave_function_simulator.simulate(circuit).final_state)


@pytest.mark.parametrize(
    "qubits",
    [cirq.LineQubit.range(2), cirq.LineQubit.range(4)])
def test_clifford_circuit_2(qubits):
    circuit = cirq.Circuit()

    np.random.seed(1)

    for _ in range(100):
        x = np.random.randint(7)

        if x == 0:
            circuit.append(cirq.X(np.random.choice(qubits)))  # coverage: ignore
        elif x == 1:
            circuit.append(cirq.Z(np.random.choice(qubits)))  # coverage: ignore
        elif x == 2:
            circuit.append(cirq.Y(np.random.choice(qubits)))  # coverage: ignore
        elif x == 3:
            circuit.append(cirq.S(np.random.choice(qubits)))  # coverage: ignore
        elif x == 4:
            circuit.append(cirq.H(np.random.choice(qubits)))  # coverage: ignore
        elif x == 5:
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))  # coverage: ignore
        elif x == 6:
            circuit.append(cirq.CZ(qubits[0], qubits[1]))  # coverage: ignore

    circuit.append(cirq.measure(qubits[0]))
    result = cirq.CliffordSimulator().run(circuit, repetitions=100)

    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


def test_non_clifford_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(cirq.T(q0))
    with pytest.raises(ValueError,
                       match="T cannot be run with Clifford simulator"):
        cirq.CliffordSimulator().simulate(circuit)
