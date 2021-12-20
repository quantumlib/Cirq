# pylint: disable=wrong-or-nonexistent-copyright-notice
import itertools
from unittest import mock

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing


def test_simulate_no_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(), np.array([1, 0, 0, 0]))
    assert len(result.measurements) == 0


def test_run_no_repetitions():
    q0 = cirq.LineQubit(0)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=0)
    assert sum(result.measurements['0']) == 0


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


def test_run_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.run_sweep(circuit, cirq.ParamResolver({}))


def test_simulate_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.simulate_sweep(circuit, cirq.ParamResolver({}))


def test_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state.to_numpy(), np.array([0.5, 0.5, 0.5, 0.5]))
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
            np.testing.assert_almost_equal(
                result.final_state.to_numpy(), np.reshape(expected_state, 4)
            )


def test_simulate_act_on_args():
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

            args = simulator._create_act_on_args(initial_state=1, qubits=(q0, q1))
            result = simulator.simulate(circuit, initial_state=args)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][1 - b1] = 1.0
            np.testing.assert_almost_equal(
                result.final_state.to_numpy(), np.reshape(expected_state, 4)
            )


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
            np.testing.assert_almost_equal(
                result.final_state.to_numpy(), np.reshape(expected_state, 4)
            )


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
            np.testing.assert_equal(result.measurements, {'0,1': [[b0, b1]] * 3})


def test_simulate_moment_steps():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q0), cirq.H(q1))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step.state.to_numpy(), np.array([0.5] * 4))
        else:
            np.testing.assert_almost_equal(step.state.to_numpy(), np.array([1, 0, 0, 0]))


def test_simulate_moment_steps_sample():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, False]) or np.array_equal(
                    sample, [False, False]
                )
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, True]) or np.array_equal(
                    sample, [False, False]
                )


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
            expected = np.array([np.sqrt(0.5), np.sqrt(0.5) * (-1) ** result])
            np.testing.assert_almost_equal(step.state.to_numpy(), expected)


def test_clifford_state_initial_state():
    q0 = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.CliffordState(qubit_map={q0: 0}, initial_state=2)
    state = cirq.CliffordState(qubit_map={q0: 0}, initial_state=1)
    np.testing.assert_allclose(state.state_vector(), [0, 1])


def test_clifford_trial_result_repr():
    q0 = cirq.LineQubit(0)
    final_step_result = mock.Mock(cirq.CliffordSimulatorStepResult)
    final_step_result._simulator_state.return_value = cirq.CliffordState(qubit_map={q0: 0})
    assert (
        repr(
            cirq.CliffordTrialResult(
                params=cirq.ParamResolver({}),
                measurements={'m': np.array([[1]])},
                final_step_result=final_step_result,
            )
        )
        == "cirq.SimulationTrialResult(params=cirq.ParamResolver({}), "
        "measurements={'m': array([[1]])}, "
        "final_simulator_state=StabilizerStateChForm(num_qubits=1))"
    )


def test_clifford_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_step_result = mock.Mock(cirq.CliffordSimulatorStepResult)
    final_step_result._simulator_state.return_value = cirq.CliffordState(qubit_map={q0: 0})
    assert (
        str(
            cirq.CliffordTrialResult(
                params=cirq.ParamResolver({}),
                measurements={'m': np.array([[1]])},
                final_step_result=final_step_result,
            )
        )
        == "measurements: m=1\n"
        "output state: |0⟩"
    )


def test_clifford_trial_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    final_step_result = mock.Mock(cirq.CliffordSimulatorStepResult)
    final_step_result._simulator_state.return_value = cirq.CliffordState(qubit_map={q0: 0})
    result = cirq.CliffordTrialResult(
        params=cirq.ParamResolver({}),
        measurements={'m': np.array([[1]])},
        final_step_result=final_step_result,
    )

    cirq.testing.assert_repr_pretty(result, "measurements: m=1\n" "output state: |0⟩")
    cirq.testing.assert_repr_pretty(result, "cirq.CliffordTrialResult(...)", cycle=True)


def test_clifford_step_result_str():
    q0 = cirq.LineQubit(0)
    result = next(
        cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.measure(q0, key='m')))
    )
    assert str(result) == "m=0\n" "|0⟩"


def test_clifford_step_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    result = next(
        cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.measure(q0, key='m')))
    )
    cirq.testing.assert_repr_pretty(result, "m=0\n" "|0⟩")
    cirq.testing.assert_repr_pretty(result, "cirq.CliffordSimulatorStateResult(...)", cycle=True)


def test_clifford_step_result_no_measurements_str():
    q0 = cirq.LineQubit(0)
    result = next(cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.I(q0))))
    assert str(result) == "|0⟩"


def test_clifford_state_str():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})

    assert str(state) == "|00⟩"


def test_clifford_state_state_vector():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})

    np.testing.assert_equal(state.state_vector(), [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])


def test_stabilizerStateChForm_H():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    with pytest.raises(ValueError, match="|y> is equal to |z>"):
        state.ch_form._H_decompose(0, 1, 1, 0)


def test_clifford_stabilizerStateChForm_repr():
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    assert repr(state) == 'StabilizerStateChForm(num_qubits=2)'


def test_clifford_circuit_SHSYSHS():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.Y(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.measure(q0),
    )

    clifford_simulator = cirq.CliffordSimulator()
    state_vector_simulator = cirq.Simulator()

    np.testing.assert_almost_equal(
        clifford_simulator.simulate(circuit).final_state.state_vector(),
        state_vector_simulator.simulate(circuit).final_state_vector,
    )


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
    state_vector_simulator = cirq.Simulator()

    np.testing.assert_almost_equal(
        clifford_simulator.simulate(circuit).final_state.state_vector(),
        state_vector_simulator.simulate(circuit).final_state_vector,
    )


@pytest.mark.parametrize("qubits", [cirq.LineQubit.range(2), cirq.LineQubit.range(4)])
def test_clifford_circuit_2(qubits):
    circuit = cirq.Circuit()

    np.random.seed(2)

    for _ in range(50):
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


def test_clifford_circuit_3():
    # This test tests the simulator on arbitrary 1-qubit Clifford gates.
    (q0, q1) = (cirq.LineQubit(0), cirq.LineQubit(1))
    circuit = cirq.Circuit()

    def random_clifford_gate():
        matrix = np.eye(2)
        for _ in range(10):
            matrix = matrix @ cirq.unitary(np.random.choice((cirq.H, cirq.S)))
        matrix *= np.exp(1j * np.random.uniform(0, 2 * np.pi))
        return cirq.MatrixGate(matrix)

    for _ in range(20):
        if np.random.randint(5) == 0:
            circuit.append(cirq.CNOT(q0, q1))
        else:
            circuit.append(random_clifford_gate()(np.random.choice((q0, q1))))

    clifford_simulator = cirq.CliffordSimulator()
    state_vector_simulator = cirq.Simulator()

    np.testing.assert_almost_equal(
        clifford_simulator.simulate(circuit).final_state.state_vector(),
        state_vector_simulator.simulate(circuit).final_state_vector,
        decimal=6,
    )


def test_non_clifford_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    circuit.append(cirq.T(q0))
    with pytest.raises(TypeError, match="support cirq.T"):
        cirq.CliffordSimulator().simulate(circuit)


def test_swap():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(a),
        cirq.SWAP(a, b),
        cirq.SWAP(a, b) ** 4,
        cirq.measure(a, key="a"),
        cirq.measure(b, key="b"),
    )
    r = cirq.CliffordSimulator().sample(circuit)
    assert not r["a"][0]
    assert r["b"][0]

    with pytest.raises(TypeError, match="CliffordSimulator doesn't support"):
        cirq.CliffordSimulator().simulate((cirq.Circuit(cirq.SWAP(a, b) ** 3.5)))


def test_sample_seed():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    simulator = cirq.CliffordSimulator(seed=1234)
    result = simulator.run(circuit, repetitions=20)
    measured = result.measurements['q']
    result_string = ''.join(map(lambda x: str(int(x[0])), measured))
    assert result_string == '11010001111100100000'


def test_is_supported_operation():
    class MultiQubitOp(cirq.Operation):
        """Multi-qubit operation with unitary.

        Used to verify that `is_supported_operation` does not attempt to
        allocate the unitary for multi-qubit operations.
        """

        @property
        def qubits(self):
            return cirq.LineQubit.range(100)

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _has_unitary_(self):
            return True

        def _unitary_(self):
            assert False

    q1, q2 = cirq.LineQubit.range(2)
    assert cirq.CliffordSimulator.is_supported_operation(cirq.X(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.H(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.CNOT(q1, q2))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.measure(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.global_phase_operation(1j))

    assert not cirq.CliffordSimulator.is_supported_operation(cirq.T(q1))
    assert not cirq.CliffordSimulator.is_supported_operation(MultiQubitOp())


def test_simulate_pauli_string():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit([cirq.PauliString({q: 'X'}), cirq.PauliString({q: 'Z'})])
    simulator = cirq.CliffordSimulator()

    result = simulator.simulate(circuit).final_state.state_vector()

    assert np.allclose(result, [0, -1])


def test_simulate_global_phase_operation():
    q1, q2 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.I(q1), cirq.I(q2), cirq.global_phase_operation(-1j)])
    simulator = cirq.CliffordSimulator()

    result = simulator.simulate(circuit).final_state.state_vector()

    assert np.allclose(result, [-1j, 0, 0, 0])


def test_json_roundtrip():
    (q0, q1, q2) = (cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1, q2: 2})

    # Apply some transformations.
    state.apply_unitary(cirq.X(q0))
    state.apply_unitary(cirq.H(q1))

    with pytest.raises(ValueError, match='T cannot be run with Clifford simulator.'):
        state.apply_unitary(cirq.T(q1))

    # Roundtrip serialize, then deserialize.
    state_roundtrip = cirq.CliffordState._from_json_dict_(**state._json_dict_())

    # Apply the same transformation on both the original object and the one that
    # went through the roundtrip.
    state.apply_unitary(cirq.S(q1))
    state_roundtrip.apply_unitary(cirq.S(q1))

    # And the CH form isn't changed either.
    assert np.allclose(state.ch_form.state_vector(), state_roundtrip.ch_form.state_vector())


def test_invalid_apply_measurement():
    q0 = cirq.LineQubit(0)
    state = cirq.CliffordState(qubit_map={q0: 0})
    measurements = {}
    with pytest.raises(TypeError, match='only supports cirq.MeasurementGate'):
        _ = state.apply_measurement(cirq.H(q0), measurements, np.random.RandomState())
    assert measurements == {}


def test_valid_apply_measurement():
    q0 = cirq.LineQubit(0)
    state = cirq.CliffordState(qubit_map={q0: 0}, initial_state=1)
    measurements = {}
    _ = state.apply_measurement(cirq.measure(q0), measurements, np.random.RandomState())
    assert measurements == {'0': [1]}


def test_reset():
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.X(q), cirq.reset(q), cirq.measure(q, key="out"))
    assert cirq.CliffordSimulator().sample(c)["out"][0] == 0
    c = cirq.Circuit(cirq.H(q), cirq.reset(q), cirq.measure(q, key="out"))
    assert cirq.CliffordSimulator().sample(c)["out"][0] == 0
    c = cirq.Circuit(cirq.reset(q), cirq.measure(q, key="out"))
    assert cirq.CliffordSimulator().sample(c)["out"][0] == 0


def test_state_copy():
    sim = cirq.CliffordSimulator()

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q))

    state_ch_forms = []
    for step in sim.simulate_moment_steps(circuit):
        state_ch_forms.append(step.state.ch_form)
    for x, y in itertools.combinations(state_ch_forms, 2):
        assert not np.shares_memory(x.v, y.v)
