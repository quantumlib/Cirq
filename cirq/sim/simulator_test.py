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

from unittest import mock
import numpy as np
import pytest

import cirq



@mock.patch.multiple(cirq.SimulatesSamples,
                     __abstractmethods__=set(),
                     _run=mock.Mock())
def test_run_simulator_run():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': np.array([[1]])}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    expected_result = cirq.TrialResult(repetitions=10,
                                       measurements=expected_measurements,
                                       params=param_resolver)
    assert expected_result == simulator.run(program=circuit,
                                            repetitions=10,
                                            param_resolver=param_resolver)
    simulator._run.assert_called_once_with(circuit=circuit,
                                           repetitions=10,
                                           param_resolver=param_resolver)


@mock.patch.multiple(cirq.SimulatesSamples,
                     __abstractmethods__=set(),
                     _run=mock.Mock())
def test_run_simulator_sweeps():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': np.array([[1]])}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [mock.Mock(cirq.ParamResolver),
                       mock.Mock(cirq.ParamResolver)]
    expected_results = [cirq.TrialResult(repetitions=10,
                                         measurements=expected_measurements,
                                         params=param_resolvers[0]),
                        cirq.TrialResult(repetitions=10,
                                         measurements=expected_measurements,
                                         params=param_resolvers[1])]
    assert expected_results == simulator.run_sweep(program=circuit,
                                                   repetitions=10,
                                                   params=param_resolvers)
    simulator._run.assert_called_with(circuit=circuit,
                                      repetitions=10,
                                      param_resolver=mock.ANY)
    assert simulator._run.call_count == 2


@mock.patch.multiple(cirq.SimulatesIntermediateState,
                     __abstractmethods__=set(),
                     _simulator_iterator=mock.Mock())
def test_intermediate_simulator():
    simulator = cirq.SimulatesIntermediateState()

    final_simulator_state = np.array([1, 0, 0, 0])
    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': [True, True]}
        yield result
        result = mock.Mock()
        result.measurements = {'b': [True, False]}
        result.simulator_state.return_value = final_simulator_state
        yield result

    simulator._simulator_iterator.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    qubit_order = mock.Mock(cirq.QubitOrder)
    result = simulator.simulate(program=circuit,
                                param_resolver=param_resolver,
                                qubit_order=qubit_order,
                                initial_state=2)
    np.testing.assert_equal(result.measurements['a'], [True, True])
    np.testing.assert_equal(result.measurements['b'], [True, False])
    assert set(result.measurements.keys()) == {'a', 'b'}
    assert result.params == param_resolver
    np.testing.assert_equal(result.final_simulator_state, final_simulator_state)


@mock.patch.multiple(cirq.SimulatesIntermediateState,
                     __abstractmethods__=set(),
                     _simulator_iterator=mock.Mock())
def test_intermediate_sweeps():
    simulator = cirq.SimulatesIntermediateState()

    final_state = np.array([1, 0, 0, 0])
    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': np.array([True, True])}
        result.simulator_state.return_value = final_state
        yield result

    simulator._simulator_iterator.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [mock.Mock(cirq.ParamResolver),
                       mock.Mock(cirq.ParamResolver)]
    qubit_order = mock.Mock(cirq.QubitOrder)
    results = simulator.simulate_sweep(program=circuit,
                                       params=param_resolvers,
                                       qubit_order=qubit_order,
                                       initial_state=2)
    expected_results = [
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[0],
            final_simulator_state=final_state),
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[1],
            final_simulator_state=final_state)
    ]
    assert results == expected_results


class FakeStepResult(cirq.StepResult):

    def __init__(self, ones_qubits):
        self._ones_qubits = set(ones_qubits)

    def simulator_state(self):
        pass

    def state_vector(self):
        pass

    def __setstate__(self, state):
        pass

    def sample(self, qubits, repetitions):
        return [[qubit in self._ones_qubits for qubit in qubits]] * repetitions


def test_step_sample_measurement_ops():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1), cirq.measure(q2)]
    step_result = FakeStepResult([q1])

    measurements = step_result.sample_measurement_ops(measurement_ops)
    np.testing.assert_equal(measurements,
                            {'0,1': [[False, True]], '2': [[False]]})


def test_step_sample_measurement_ops_repetitions():
    q0, q1, q2 = cirq.LineQubit.range(3)
    measurement_ops = [cirq.measure(q0, q1), cirq.measure(q2)]
    step_result = FakeStepResult([q1])

    measurements = step_result.sample_measurement_ops(measurement_ops,
                                                      repetitions=3)
    np.testing.assert_equal(measurements,
                            {'0,1': [[False, True]] * 3, '2': [[False]] * 3})


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
                [cirq.measure(q0), cirq.measure(q1, q2), cirq.measure(q0)])


def test_simulation_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.SimulationTrialResult(params=cirq.ParamResolver({}),
                                   measurements={},
                                   final_simulator_state=()),
        cirq.SimulationTrialResult(params=cirq.ParamResolver({}),
                                   measurements={},
                                   final_simulator_state=()))
    eq.add_equality_group(
        cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}),
                                   measurements={},
                                   final_simulator_state=()))
    eq.add_equality_group(
        cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}),
                                   measurements={'m': np.array([[1]])},
                                   final_simulator_state=()))
    eq.add_equality_group(
        cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}),
                                   measurements={'m': np.array([[1]])},
                                   final_simulator_state=(0, 1)))



def test_simulation_trial_result_repr():
    assert repr(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}),
                                           measurements={'m': np.array([[1]])},
                                           final_simulator_state=(0, 1))) == (
               "cirq.SimulationTrialResult("
               "params=cirq.ParamResolver({'s': 1}), "
               "measurements={'m': array([[1]])}, "
               "final_simulator_state=(0, 1))")


def test_simulation_trial_result_str():
    assert str(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_simulator_state=(0, 1))) == '(no measurements)'

    assert str(cirq.SimulationTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=(0, 1))) == 'm=1'


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
