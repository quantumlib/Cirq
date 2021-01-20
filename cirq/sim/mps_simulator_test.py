import itertools
import math

import numpy as np
import pytest
import sympy

import cirq


def assert_same_output_as_dense(circuit, qubit_order, initial_state=0):
    mps_simulator = cirq.MPSSimulator()
    ref_simulator = cirq.Simulator()

    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    np.testing.assert_almost_equal(actual.final_state.to_numpy(), expected.final_state_vector)
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

    circuit = cirq.Circuit(
        cirq.H(q1),
        cirq.CNOT(q1, q0),
    )

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


def test_three_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCX(q0, q1, q2))

    with pytest.raises(ValueError, match="Can only handle 1 and 2 qubit operations"):
        assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2])


def test_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.H(q1), cirq.measure(q1))

    simulator = cirq.MPSSimulator()

    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['1'])[0] < 80
    assert sum(result.measurements['1'])[0] > 20


def test_measurement_str():
    q0 = cirq.NamedQid('q0', dimension=3)
    circuit = cirq.Circuit(cirq.measure(q0))

    simulator = cirq.MPSSimulator()
    result = simulator.run(circuit, repetitions=7)

    assert str(result) == "q0 (d=3)=0000000"


def test_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.MPSState(qubit_map={q0: 0})
    assert (
        str(
            cirq.MPSTrialResult(
                params=cirq.ParamResolver({}),
                measurements={'m': np.array([[1]])},
                final_simulator_state=final_simulator_state,
            )
        )
        == "measurements: m=1\n"
        "output state: [array([[[1., 0.]]])]"
    )


def test_empty_step_result():
    q0 = cirq.LineQubit(0)
    state = cirq.MPSState(qubit_map={q0: 0})
    step_result = cirq.MPSSimulatorStepResult(state, measurements={'0': [1]})
    assert str(step_result) == "0=1\n[array([[[1., 0.]]])]"


def test_state_equal():
    q0, q1 = cirq.LineQubit.range(2)
    state0 = cirq.MPSState(qubit_map={q0: 0})
    state1 = cirq.MPSState(qubit_map={q1: 0})
    assert state0 == state0
    assert state0 != state1


def test_simulate_moment_steps_sample():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

    simulator = cirq.MPSSimulator()

    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(
                step._simulator_state().to_numpy(),
                np.asarray([1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2), 0.0]),
            )
            assert str(step) == "[array([[[0.70710678+0.j, 0.70710678+0.j]]]), array([[[1., 0.]]])]"
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
            assert (
                str(step)
                == """[array([[[0.84089642+0.j, 0.        +0.j],
        [0.        +0.j, 0.84089642+0.j]]]), array([[[0.84089642+0.j, 0.        +0.j]],

       [[0.        +0.j, 0.84089642+0.j]]])]"""
            )
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, True]) or np.array_equal(
                    sample, [False, False]
                )


def test_sample_seed():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    simulator = cirq.MPSSimulator(seed=1234)
    result = simulator.run(circuit, repetitions=20)
    measured = result.measurements['q']
    result_string = ''.join(map(lambda x: str(int(x[0])), measured))
    assert result_string == '01011001110111011011'


def test_run_no_repetitions():
    q0 = cirq.LineQubit(0)
    simulator = cirq.MPSSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=0)
    assert len(result.measurements['0']) == 0


def test_run_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.MPSSimulator()
    circuit = cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.run_sweep(circuit, cirq.ParamResolver({}))


def test_state_copy():
    sim = cirq.MPSSimulator()

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q))

    state_Ms = []
    for step in sim.simulate_moment_steps(circuit):
        state_Ms.append(step.state.M)
    for x, y in itertools.combinations(state_Ms, 2):
        assert len(x) == len(y)
        for i in range(len(x)):
            assert not np.shares_memory(x[i], y[i])
