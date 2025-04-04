# pylint: disable=wrong-or-nonexistent-copyright-notice
import itertools
import math

import numpy as np
import pytest
import sympy

import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value


def assert_same_output_as_dense(circuit, qubit_order, initial_state=0, grouping=None):
    mps_simulator = ccq.mps_simulator.MPSSimulator(grouping=grouping)
    ref_simulator = cirq.Simulator()

    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    np.testing.assert_allclose(
        actual.final_state.to_numpy(), expected.final_state_vector, atol=1e-4
    )
    assert len(actual.measurements) == 0


def test_various_gates_1d():
    gate_op_cls = [cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.T]
    cross_gate_op_cls = [cirq.CNOT, cirq.SWAP]

    q0, q1 = cirq.LineQubit.range(2)

    for q0_gate_op in gate_op_cls:
        for q1_gate_op in gate_op_cls:
            for cross_gate_op in cross_gate_op_cls:
                circuit = cirq.Circuit(q0_gate_op(q0), q1_gate_op(q1), cross_gate_op(q0, q1))
                for initial_state in range(2 * 2):
                    assert_same_output_as_dense(
                        circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state
                    )


def test_various_gates_1d_flip():
    q0, q1 = cirq.LineQubit.range(2)

    circuit = cirq.Circuit(cirq.H(q1), cirq.CNOT(q1, q0))

    assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1])
    assert_same_output_as_dense(circuit=circuit, qubit_order=[q1, q0])


def test_various_gates_2d():
    gate_op_cls = [cirq.I, cirq.H]
    cross_gate_op_cls = [cirq.CNOT, cirq.SWAP]

    q0, q1, q2, q3, q4, q5 = cirq.GridQubit.rect(3, 2)

    for q0_gate_op in gate_op_cls:
        for q1_gate_op in gate_op_cls:
            for q2_gate_op in gate_op_cls:
                for q3_gate_op in gate_op_cls:
                    for cross_gate_op1 in cross_gate_op_cls:
                        for cross_gate_op2 in cross_gate_op_cls:
                            circuit = cirq.Circuit(
                                q0_gate_op(q0),
                                q1_gate_op(q1),
                                cross_gate_op1(q0, q1),
                                q2_gate_op(q2),
                                q3_gate_op(q3),
                                cross_gate_op2(q3, q1),
                            )
                            assert_same_output_as_dense(
                                circuit=circuit, qubit_order=[q0, q1, q2, q3, q4, q5]
                            )


def test_grouping():
    q0, q1, q2 = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        cirq.X(q0) ** 0.1,
        cirq.Y(q1) ** 0.2,
        cirq.Z(q2) ** 0.3,
        cirq.CNOT(q0, q1),
        cirq.Y(q1) ** 0.4,
    )

    groupings = [
        None,
        {q0: 0, q1: 1, q2: 2},
        {q0: 0, q1: 0, q2: 1},
        {q0: 0, q1: 1, q2: 0},
        {q0: 1, q1: 0, q2: 0},
        {q0: 0, q1: 0, q2: 0},
    ]

    for grouping in groupings:
        for initial_state in range(2 * 2 * 2):
            assert_same_output_as_dense(
                circuit=circuit,
                qubit_order=[q0, q1, q2],
                initial_state=initial_state,
                grouping=grouping,
            )


def test_grouping_does_not_overlap():
    q0, q1 = cirq.LineQubit.range(2)
    mps_simulator = ccq.mps_simulator.MPSSimulator(grouping={q0: 0})

    with pytest.raises(ValueError, match="Grouping must cover exactly the qubits"):
        mps_simulator.simulate(cirq.Circuit(), qubit_order={q0: 0, q1: 1})


def test_same_partial_trace():
    qubit_order = cirq.LineQubit.range(2)
    q0, q1 = qubit_order

    mps_simulator = ccq.mps_simulator.MPSSimulator()

    for _ in range(50):
        for initial_state in range(4):
            circuit = cirq.testing.random_circuit(qubit_order, 3, 0.9)
            expected_density_matrix = cirq.final_density_matrix(
                circuit, qubit_order=qubit_order, initial_state=initial_state
            )
            expected_partial_trace = cirq.partial_trace(
                expected_density_matrix.reshape(2, 2, 2, 2), keep_indices=[0]
            )

            final_state = mps_simulator.simulate(
                circuit, qubit_order=qubit_order, initial_state=initial_state
            ).final_state
            actual_density_matrix = final_state.partial_trace([q0, q1])
            actual_partial_trace = final_state.partial_trace([q0])

            np.testing.assert_allclose(actual_density_matrix, expected_density_matrix, atol=1e-4)
            np.testing.assert_allclose(actual_partial_trace, expected_partial_trace, atol=1e-4)


def test_probs_dont_sum_up_to_one():
    q0 = cirq.NamedQid('q0', dimension=2)
    circuit = cirq.Circuit(cirq.measure(q0))

    simulator = ccq.mps_simulator.MPSSimulator(
        simulation_options=ccq.mps_simulator.MPSOptions(sum_prob_atol=-0.5)
    )

    with pytest.raises(ValueError, match="Sum of probabilities exceeds tolerance"):
        simulator.run(circuit, repetitions=1)


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


def test_simulation_state():
    q0, q1 = qubit_order = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q1, q0))
    mps_simulator = ccq.mps_simulator.MPSSimulator()
    ref_simulator = cirq.Simulator()
    for initial_state in range(4):
        args = mps_simulator._create_simulation_state(initial_state=initial_state, qubits=(q0, q1))
        actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=args)
        expected = ref_simulator.simulate(
            circuit, qubit_order=qubit_order, initial_state=initial_state
        )
        np.testing.assert_allclose(
            actual.final_state.to_numpy(), expected.final_state_vector, atol=1e-4
        )
        assert len(actual.measurements) == 0


def test_three_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCX(q0, q1, q2))

    with pytest.raises(ValueError, match="Can only handle 1 and 2 qubit operations"):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2])


def test_measurement_1qubit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.H(q1), cirq.measure(q1))

    simulator = ccq.mps_simulator.MPSSimulator()

    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['q(1)'])[0] < 80
    assert sum(result.measurements['q(1)'])[0] > 20


def test_reset():
    q = cirq.LineQubit(0)
    simulator = ccq.mps_simulator.MPSSimulator()
    c = cirq.Circuit(cirq.X(q), cirq.reset(q), cirq.measure(q))
    assert simulator.sample(c)['q(0)'][0] == 0
    c = cirq.Circuit(cirq.H(q), cirq.reset(q), cirq.measure(q))
    assert simulator.sample(c)['q(0)'][0] == 0
    c = cirq.Circuit(cirq.reset(q), cirq.measure(q))
    assert simulator.sample(c)['q(0)'][0] == 0


def test_measurement_2qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q2), cirq.measure(q0, q2))

    simulator = ccq.mps_simulator.MPSSimulator()

    repetitions = 1024
    measurement = simulator.run(circuit, repetitions=repetitions).measurements['q(0),q(2)']

    result_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    for i in range(repetitions):
        key = str(measurement[i, 0]) + str(measurement[i, 1])
        result_counts[key] += 1

    for result_count in result_counts.values():
        # Expected value is 1/4:
        assert result_count > repetitions * 0.15
        assert result_count < repetitions * 0.35


def test_measurement_str():
    q0 = cirq.NamedQid('q0', dimension=3)
    circuit = cirq.Circuit(cirq.measure(q0))

    simulator = ccq.mps_simulator.MPSSimulator()
    result = simulator.run(circuit, repetitions=7)

    assert str(result) == "q0 (d=3)=0000000"


def test_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = ccq.mps_simulator.MPSState(
        qubits=(q0,),
        prng=value.parse_random_state(0),
        simulation_options=ccq.mps_simulator.MPSOptions(),
    )
    result = ccq.mps_simulator.MPSTrialResult(
        params=cirq.ParamResolver({}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state,
    )
    assert 'output state: TensorNetwork' in str(result)


def test_trial_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    final_simulator_state = ccq.mps_simulator.MPSState(
        qubits=(q0,),
        prng=value.parse_random_state(0),
        simulation_options=ccq.mps_simulator.MPSOptions(),
    )
    result = ccq.mps_simulator.MPSTrialResult(
        params=cirq.ParamResolver({}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state,
    )
    cirq.testing.assert_repr_pretty_contains(result, 'output state: TensorNetwork')
    cirq.testing.assert_repr_pretty(result, "cirq.MPSTrialResult(...)", cycle=True)


def test_empty_step_result():
    q0 = cirq.LineQubit(0)
    sim = ccq.mps_simulator.MPSSimulator()
    step_result = next(sim.simulate_moment_steps(cirq.Circuit(cirq.measure(q0))))
    assert 'TensorNetwork' in str(step_result)


def test_step_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    sim = ccq.mps_simulator.MPSSimulator()
    step_result = next(sim.simulate_moment_steps(cirq.Circuit(cirq.measure(q0))))
    cirq.testing.assert_repr_pretty_contains(step_result, 'TensorNetwork')
    cirq.testing.assert_repr_pretty(step_result, "cirq.MPSSimulatorStepResult(...)", cycle=True)


def test_state_equal():
    q0, q1 = cirq.LineQubit.range(2)
    state0 = ccq.mps_simulator.MPSState(
        qubits=(q0,),
        prng=value.parse_random_state(0),
        simulation_options=ccq.mps_simulator.MPSOptions(cutoff=1e-3, sum_prob_atol=1e-3),
    )
    state1a = ccq.mps_simulator.MPSState(
        qubits=(q1,),
        prng=value.parse_random_state(0),
        simulation_options=ccq.mps_simulator.MPSOptions(cutoff=1e-3, sum_prob_atol=1e-3),
    )
    state1b = ccq.mps_simulator.MPSState(
        qubits=(q1,),
        prng=value.parse_random_state(0),
        simulation_options=ccq.mps_simulator.MPSOptions(cutoff=1729.0, sum_prob_atol=1e-3),
    )
    assert state0 == state0
    assert state0 != state1a
    assert state1a != state1b


def test_random_circuits_equal_more_rows():
    circuit = cirq.testing.random_circuit(
        qubits=cirq.GridQubit.rect(3, 2), n_moments=6, op_density=1.0
    )
    qubits = circuit.all_qubits()
    assert_same_output_as_dense(circuit, qubits)


def test_random_circuits_equal_more_cols():
    circuit = cirq.testing.random_circuit(
        qubits=cirq.GridQubit.rect(2, 3), n_moments=6, op_density=1.0
    )
    qubits = circuit.all_qubits()
    assert_same_output_as_dense(circuit, qubits)


def test_tensor_index_names():
    qubits = cirq.LineQubit.range(12)
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}
    state = ccq.mps_simulator.MPSState(qubits=qubit_map, prng=value.parse_random_state(0))

    assert state.i_str(0) == "i_00"
    assert state.i_str(11) == "i_11"
    assert state.mu_str(0, 3) == "mu_0_3"
    assert state.mu_str(3, 0) == "mu_0_3"


def test_simulate_moment_steps_sample():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

    simulator = ccq.mps_simulator.MPSSimulator()

    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(
                step._simulator_state().to_numpy(),
                np.asarray([1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2), 0.0]),
            )
            # There are two "Tensor()" copies in the string.
            assert len(str(step).split('Tensor(')) == 3
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, False]) or np.array_equal(
                    sample, [False, False]
                )
            np.testing.assert_almost_equal(
                step._simulator_state().to_numpy(),
                np.asarray([1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2), 0.0]),
            )
        else:
            np.testing.assert_almost_equal(
                step._simulator_state().to_numpy(),
                np.asarray([1.0 / math.sqrt(2), 0.0, 0.0, 1.0 / math.sqrt(2)]),
            )
            # There are two "Tensor()" copies in the string.
            assert len(str(step).split('Tensor(')) == 3
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, True]) or np.array_equal(
                    sample, [False, False]
                )


def test_sample_seed():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    simulator = ccq.mps_simulator.MPSSimulator(seed=1234)
    result = simulator.run(circuit, repetitions=20)
    measured = result.measurements['q']
    result_string = ''.join(map(lambda x: str(int(x[0])), measured))
    assert result_string == '01011001110111011011'


def test_run_no_repetitions():
    q0 = cirq.LineQubit(0)
    simulator = ccq.mps_simulator.MPSSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=0)
    assert len(result.measurements['q(0)']) == 0


def test_run_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = ccq.mps_simulator.MPSSimulator()
    circuit = cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.run_sweep(circuit, cirq.ParamResolver({}))


def test_deterministic_gate_noise():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q), cirq.measure(q))

    simulator1 = ccq.mps_simulator.MPSSimulator(noise=cirq.X)
    result1 = simulator1.run(circuit, repetitions=10)

    simulator2 = ccq.mps_simulator.MPSSimulator(noise=cirq.X)
    result2 = simulator2.run(circuit, repetitions=10)

    assert result1 == result2

    simulator3 = ccq.mps_simulator.MPSSimulator(noise=cirq.Z)
    result3 = simulator3.run(circuit, repetitions=10)

    assert result1 != result3


def test_nondeterministic_mixture_noise():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q), cirq.measure(q))

    simulator = ccq.mps_simulator.MPSSimulator(
        noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(0.5))
    )
    result1 = simulator.run(circuit, repetitions=50)
    result2 = simulator.run(circuit, repetitions=50)

    assert result1 != result2


def test_unsupported_noise_fails():
    with pytest.raises(ValueError, match='noise must be unitary or mixture but was'):
        ccq.mps_simulator.MPSSimulator(noise=cirq.amplitude_damp(0.5))


def test_state_copy():
    sim = ccq.mps_simulator.MPSSimulator()

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q))

    state_Ms = []
    for step in sim.simulate_moment_steps(circuit):
        state_Ms.append(step.state.M)
    for x, y in itertools.combinations(state_Ms, 2):
        assert len(x) == len(y)
        for i in range(len(x)):
            assert not np.shares_memory(x[i], y[i])


def test_simulation_state_initializer():
    expected_classical_data = cirq.ClassicalDataDictionaryStore(
        _records={cirq.MeasurementKey('test'): [(4,)]}
    )
    s = ccq.mps_simulator.MPSState(
        qubits=(cirq.LineQubit(0),),
        prng=np.random.RandomState(0),
        classical_data=expected_classical_data,
    )
    assert s.qubits == (cirq.LineQubit(0),)
    assert s.classical_data == expected_classical_data
    assert s.estimation_stats() == {
        'estimated_fidelity': 1.0,
        'memory_bytes': 16,
        'num_coefs_used': 2,
    }


def test_act_on_gate():
    args = ccq.mps_simulator.MPSState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState(0))

    cirq.act_on(cirq.X, args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(
        args.state_vector().reshape((2, 2, 2)),
        cirq.one_hot(index=(0, 1, 0), shape=(2, 2, 2), dtype=np.complex64),
    )
