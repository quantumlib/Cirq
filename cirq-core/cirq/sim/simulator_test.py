# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for simulator.py"""
import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock

import duet
import numpy as np
import pytest

import cirq
from cirq import study
from cirq.sim.simulator import (
    TStepResult,
    TSimulatorState,
    SimulatesAmplitudes,
    SimulatesExpectationValues,
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulationTrialResult,
    TActOnArgs,
)


class SimulatesIntermediateStateImpl(
    Generic[TStepResult, TSimulatorState, TActOnArgs],
    SimulatesIntermediateState[TStepResult, 'SimulationTrialResult', TSimulatorState, TActOnArgs],
    metaclass=abc.ABCMeta,
):
    """A SimulatesIntermediateState that uses the default SimulationTrialResult type."""

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: TStepResult,
    ) -> 'SimulationTrialResult':
        """This method creates a default trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_step_result: The final step result of the simulation.

        Returns:
            The SimulationTrialResult.
        """
        return SimulationTrialResult(
            params=params, measurements=measurements, final_step_result=final_step_result
        )


@mock.patch.multiple(cirq.SimulatesSamples, __abstractmethods__=set(), _run=mock.Mock())
def test_run_simulator_run():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': np.array([[1]])}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    circuit.__iter__ = mock.Mock(return_value=iter([]))
    param_resolver = mock.Mock(cirq.ParamResolver)
    param_resolver.param_dict = {}
    expected_result = cirq.Result(measurements=expected_measurements, params=param_resolver)
    assert expected_result == simulator.run(
        program=circuit, repetitions=10, param_resolver=param_resolver
    )
    simulator._run.assert_called_once_with(
        circuit=circuit, repetitions=10, param_resolver=param_resolver
    )


@mock.patch.multiple(cirq.SimulatesSamples, __abstractmethods__=set(), _run=mock.Mock())
def test_run_simulator_sweeps():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': np.array([[1]])}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    circuit.__iter__ = mock.Mock(return_value=iter([]))
    param_resolvers = [mock.Mock(cirq.ParamResolver), mock.Mock(cirq.ParamResolver)]
    for resolver in param_resolvers:
        resolver.param_dict = {}
    expected_results = [
        cirq.Result(measurements=expected_measurements, params=param_resolvers[0]),
        cirq.Result(measurements=expected_measurements, params=param_resolvers[1]),
    ]
    assert expected_results == simulator.run_sweep(
        program=circuit, repetitions=10, params=param_resolvers
    )
    simulator._run.assert_called_with(circuit=circuit, repetitions=10, param_resolver=mock.ANY)
    assert simulator._run.call_count == 2


@mock.patch.multiple(
    SimulatesIntermediateStateImpl, __abstractmethods__=set(), simulate_moment_steps=mock.Mock()
)
def test_intermediate_simulator():
    simulator = SimulatesIntermediateStateImpl()

    final_simulator_state = np.array([1, 0, 0, 0])

    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': [True, True]}
        yield result
        result = mock.Mock()
        result.measurements = {'b': [True, False]}
        result._simulator_state.return_value = final_simulator_state
        yield result

    simulator.simulate_moment_steps.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    param_resolver.param_dict = {}
    qubit_order = mock.Mock(cirq.QubitOrder)
    result = simulator.simulate(
        program=circuit, param_resolver=param_resolver, qubit_order=qubit_order, initial_state=2
    )
    np.testing.assert_equal(result.measurements['a'], [True, True])
    np.testing.assert_equal(result.measurements['b'], [True, False])
    assert set(result.measurements.keys()) == {'a', 'b'}
    assert result.params == param_resolver
    np.testing.assert_equal(result._final_simulator_state, final_simulator_state)


@mock.patch.multiple(
    SimulatesIntermediateStateImpl, __abstractmethods__=set(), simulate_moment_steps=mock.Mock()
)
def test_intermediate_sweeps():
    simulator = SimulatesIntermediateStateImpl()

    final_state = np.array([1, 0, 0, 0])

    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': np.array([True, True])}
        result._simulator_state.return_value = final_state
        yield result

    simulator.simulate_moment_steps.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [mock.Mock(cirq.ParamResolver), mock.Mock(cirq.ParamResolver)]
    for resolver in param_resolvers:
        resolver.param_dict = {}
    qubit_order = mock.Mock(cirq.QubitOrder)
    results = simulator.simulate_sweep(
        program=circuit, params=param_resolvers, qubit_order=qubit_order, initial_state=2
    )

    final_step_result = mock.Mock()
    final_step_result._simulator_state.return_value = final_state
    expected_results = [
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[0],
            final_step_result=final_step_result,
        ),
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[1],
            final_step_result=final_step_result,
        ),
    ]
    assert results == expected_results


class FakeStepResult(cirq.StepResult):
    def __init__(self, ones_qubits):
        self._ones_qubits = set(ones_qubits)

    def _simulator_state(self):
        pass

    def state_vector(self):
        pass

    def __setstate__(self, state):
        pass

    def sample(self, qubits, repetitions=1, seed=None):
        return np.array([[qubit in self._ones_qubits for qubit in qubits]] * repetitions)


def test_step_sample_measurement_ops():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1), cirq.measure(q2)]
    step_result = FakeStepResult([q1])

    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements, {'0,1': [[False, True]], '2': [[False]]})


def test_step_sample_measurement_ops_repetitions():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1), cirq.measure(q2)]
    step_result = FakeStepResult([q1])

    measurements = step_result.sample_measurement_ops(measurement_ops, repetitions=3)
    np.testing.assert_equal(measurements, {'0,1': [[False, True]] * 3, '2': [[False]] * 3})


def test_step_sample_measurement_ops_invert_mask():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [
        cirq.measure(q0, q1, invert_mask=(True,)),
        cirq.measure(q2, invert_mask=(False,)),
    ]
    step_result = FakeStepResult([q1])

    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements, {'0,1': [[True, True]], '2': [[False]]})


def test_step_sample_measurement_ops_no_measurements():
    step_result = FakeStepResult([])

    measurements = step_result.sample_measurement_ops([])
    assert measurements == {}


def test_step_sample_measurement_ops_not_measurement():
    q0 = cirq.LineQubit(0)
    step_result = FakeStepResult([q0])
    with pytest.raises(ValueError, match='MeasurementGate'):
        step_result.sample_measurement_ops([cirq.X(q0)])


def test_step_sample_measurement_ops_repeated_qubit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    step_result = FakeStepResult([q0])
    with pytest.raises(ValueError, match='MeasurementGate'):
        step_result.sample_measurement_ops(
            [cirq.measure(q0), cirq.measure(q1, q2), cirq.measure(q0)]
        )


def test_simulation_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    final_step_result = mock.Mock(cirq.StepResult)
    final_step_result._simulator_state.return_value = ()
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({}), measurements={}, final_step_result=final_step_result
        ),
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({}), measurements={}, final_step_result=final_step_result
        ),
    )
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_step_result=final_step_result,
        )
    )
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([1])},
            final_step_result=final_step_result,
        )
    )
    final_step_result._simulator_state.return_value = (0, 1)
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([1])},
            final_step_result=final_step_result,
        )
    )


def test_simulation_trial_result_repr():
    final_step_result = mock.Mock(cirq.StepResult)
    final_step_result._simulator_state.return_value = (0, 1)
    assert repr(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([1])},
            final_step_result=final_step_result,
        )
    ) == (
        "cirq.SimulationTrialResult("
        "params=cirq.ParamResolver({'s': 1}), "
        "measurements={'m': array([1])}, "
        "final_simulator_state=(0, 1))"
    )


def test_simulation_trial_result_str():
    final_step_result = mock.Mock(cirq.StepResult)
    final_step_result._simulator_state.return_value = (0, 1)
    assert (
        str(
            cirq.SimulationTrialResult(
                params=cirq.ParamResolver({'s': 1}),
                measurements={},
                final_step_result=final_step_result,
            )
        )
        == '(no measurements)'
    )

    assert (
        str(
            cirq.SimulationTrialResult(
                params=cirq.ParamResolver({'s': 1}),
                measurements={'m': np.array([1])},
                final_step_result=final_step_result,
            )
        )
        == 'm=1'
    )

    assert (
        str(
            cirq.SimulationTrialResult(
                params=cirq.ParamResolver({'s': 1}),
                measurements={'m': np.array([1, 2, 3])},
                final_step_result=final_step_result,
            )
        )
        == 'm=123'
    )

    assert (
        str(
            cirq.SimulationTrialResult(
                params=cirq.ParamResolver({'s': 1}),
                measurements={'m': np.array([9, 10, 11])},
                final_step_result=final_step_result,
            )
        )
        == 'm=9 10 11'
    )


def test_pretty_print():
    result = cirq.SimulationTrialResult(cirq.ParamResolver(), {}, np.array([1]))

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == '(no measurements)'

    # Test cycle handling
    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'SimulationTrialResult(...)'


@duet.sync
async def test_async_sample():
    m = {'mock': np.array([[0], [1]])}

    class MockSimulator(cirq.SimulatesSamples):
        def _run(self, circuit, param_resolver, repetitions):
            return m

    q = cirq.LineQubit(0)
    f = MockSimulator().run_async(cirq.Circuit(cirq.measure(q)), repetitions=10)
    result = await f
    np.testing.assert_equal(result.measurements, m)


def test_simulation_trial_result_qubit_map():
    q = cirq.LineQubit.range(2)
    result = cirq.Simulator().simulate(cirq.Circuit([cirq.CZ(q[0], q[1])]))
    assert result.qubit_map == {q[0]: 0, q[1]: 1}

    result = cirq.DensityMatrixSimulator().simulate(cirq.Circuit([cirq.CZ(q[0], q[1])]))
    assert result.qubit_map == {q[0]: 0, q[1]: 1}


def test_verify_unique_measurement_keys():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(
        [
            cirq.measure(q[0], key='a'),
            cirq.measure(q[1], key='a'),
            cirq.measure(q[0], key='b'),
            cirq.measure(q[1], key='b'),
        ]
    )
    with pytest.raises(ValueError, match='Measurement key a,b repeated'):
        _ = cirq.sample(circuit)


def test_simulate_with_invert_mask():
    class PlusGate(cirq.Gate):
        """A qudit gate that increments a qudit state mod its dimension."""

        def __init__(self, dimension, increment=1):
            self.dimension = dimension
            self.increment = increment % dimension

        def _qid_shape_(self):
            return (self.dimension,)

        def _unitary_(self):
            inc = (self.increment - 1) % self.dimension + 1
            u = np.empty((self.dimension, self.dimension))
            u[inc:] = np.eye(self.dimension)[:-inc]
            u[:inc] = np.eye(self.dimension)[-inc:]
            return u

    q0, q1, q2, q3, q4 = cirq.LineQid.for_qid_shape((2, 3, 3, 3, 4))
    c = cirq.Circuit(
        PlusGate(2, 1)(q0),
        PlusGate(3, 1)(q2),
        PlusGate(3, 2)(q3),
        PlusGate(4, 3)(q4),
        cirq.measure(q0, q1, q2, q3, q4, key='a', invert_mask=(True,) * 4),
    )
    assert np.all(cirq.Simulator().run(c).measurements['a'] == [[0, 1, 0, 2, 3]])


def test_monte_carlo_on_unknown_channel():
    class Reset11To00(cirq.Gate):
        def num_qubits(self) -> int:
            return 2

        def _kraus_(self):
            return [
                np.eye(4) - cirq.one_hot(index=(3, 3), shape=(4, 4), dtype=np.complex64),
                cirq.one_hot(index=(0, 3), shape=(4, 4), dtype=np.complex64),
            ]

    for k in range(4):
        out = cirq.Simulator().simulate(
            cirq.Circuit(Reset11To00().on(*cirq.LineQubit.range(2))),
            initial_state=k,
        )
        np.testing.assert_allclose(
            out.state_vector(), cirq.one_hot(index=k % 3, shape=4, dtype=np.complex64), atol=1e-8
        )


def test_iter_definitions():
    final_step_result = mock.Mock(cirq.StepResult)
    final_step_result._simulator_state.return_value = []
    dummy_trial_result = SimulationTrialResult(
        params={}, measurements={}, final_step_result=final_step_result
    )

    class FakeNonIterSimulatorImpl(
        SimulatesAmplitudes,
        SimulatesExpectationValues,
        SimulatesFinalState,
    ):
        """A class which defines the non-Iterator simulator API methods.

        After v0.12, simulators are expected to implement the *_iter methods.
        """

        def compute_amplitudes_sweep(
            self,
            program: 'cirq.AbstractCircuit',
            bitstrings: Sequence[int],
            params: study.Sweepable,
            qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        ) -> Sequence[Sequence[complex]]:
            return [[1.0]]

        def simulate_expectation_values_sweep(
            self,
            program: 'cirq.AbstractCircuit',
            observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
            params: 'study.Sweepable',
            qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
            initial_state: Any = None,
            permit_terminal_measurements: bool = False,
        ) -> List[List[float]]:
            return [[1.0]]

        def simulate_sweep(
            self,
            program: 'cirq.AbstractCircuit',
            params: study.Sweepable,
            qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
            initial_state: Any = None,
        ) -> List[SimulationTrialResult]:
            return [dummy_trial_result]

    non_iter_sim = FakeNonIterSimulatorImpl()
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0))
    bitstrings = [0b0]
    params = {}
    assert non_iter_sim.compute_amplitudes_sweep(circuit, bitstrings, params) == [[1.0]]
    amp_iter = non_iter_sim.compute_amplitudes_sweep_iter(circuit, bitstrings, params)
    assert next(amp_iter) == [1.0]

    obs = cirq.X(q0)
    assert non_iter_sim.simulate_expectation_values_sweep(circuit, obs, params) == [[1.0]]
    ev_iter = non_iter_sim.simulate_expectation_values_sweep_iter(circuit, obs, params)
    assert next(ev_iter) == [1.0]

    assert non_iter_sim.simulate_sweep(circuit, params) == [dummy_trial_result]
    state_iter = non_iter_sim.simulate_sweep_iter(circuit, params)
    assert next(state_iter) == dummy_trial_result


def test_missing_iter_definitions():
    class FakeMissingIterSimulatorImpl(
        SimulatesAmplitudes,
        SimulatesExpectationValues,
        SimulatesFinalState,
    ):
        """A class which fails to define simulator methods."""

    missing_iter_sim = FakeMissingIterSimulatorImpl()
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0))
    bitstrings = [0b0]
    params = {}
    with pytest.raises(RecursionError):
        missing_iter_sim.compute_amplitudes_sweep(circuit, bitstrings, params)
    with pytest.raises(RecursionError):
        amp_iter = missing_iter_sim.compute_amplitudes_sweep_iter(circuit, bitstrings, params)
        next(amp_iter)

    obs = cirq.X(q0)
    with pytest.raises(RecursionError):
        missing_iter_sim.simulate_expectation_values_sweep(circuit, obs, params)
    with pytest.raises(RecursionError):
        ev_iter = missing_iter_sim.simulate_expectation_values_sweep_iter(circuit, obs, params)
        next(ev_iter)

    with pytest.raises(RecursionError):
        missing_iter_sim.simulate_sweep(circuit, params)
    with pytest.raises(RecursionError):
        state_iter = missing_iter_sim.simulate_sweep_iter(circuit, params)
        next(state_iter)


def test_trial_result_initializer():
    with pytest.raises(ValueError, match='Exactly one of'):
        _ = SimulationTrialResult(cirq.ParamResolver(), {}, None, None)
    with pytest.raises(ValueError, match='Exactly one of'):
        _ = SimulationTrialResult(cirq.ParamResolver(), {}, object(), mock.Mock(TStepResult))
