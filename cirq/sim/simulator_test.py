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
from typing import List, Dict

import numpy as np

import cirq
from cirq.testing.mock import mock


@mock.patch.multiple(cirq.RunSimulator, _run=mock.Mock())
def test_run_simulator_run():
    simulator = cirq.RunSimulator()
    expected_measurements = {'a': [[1]]}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    extensions = mock.Mock(cirq.Extensions)
    expected_result = cirq.TrialResult(repetitions=10,
                                       measurements=expected_measurements,
                                       params=param_resolver)
    assert expected_result == simulator.run(circuit=circuit,
                                            repetitions=10,
                                            param_resolver=param_resolver,
                                            extensions=extensions)
    simulator._run.assert_called_once_with(circuit=circuit,
                                           repetitions=10,
                                           param_resolver=param_resolver,
                                           extensions=extensions)


@mock.patch.multiple(cirq.RunSimulator, _run=mock.Mock())
def test_run_simulator_sweeps():
    simulator = cirq.RunSimulator()
    expected_measurements = {'a': [[1]]}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [mock.Mock(cirq.ParamResolver),
                       mock.Mock(cirq.ParamResolver)]
    extensions = mock.Mock(cirq.Extensions)
    expected_results = [cirq.TrialResult(repetitions=10,
                                         measurements=expected_measurements,
                                         params=param_resolvers[0]),
                        cirq.TrialResult(repetitions=10,
                                         measurements=expected_measurements,
                                         params=param_resolvers[1])]
    assert expected_results == simulator.run_sweep(program=circuit,
                                                repetitions=10,
                                                params=param_resolvers,
                                                extensions=extensions)
    simulator._run.assert_called_with(circuit=circuit,
                                           repetitions=10,
                                           param_resolver=mock.ANY,
                                           extensions=extensions)
    assert simulator._run.call_count == 2


@mock.patch.multiple(cirq.WaveFunctionSimulator,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator():
    simulator = cirq.WaveFunctionSimulator()

    final_state = np.array([1, 0, 0, 0])
    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': [True, True]}
        yield result
        result = mock.Mock()
        result.measurements = {'b': [True, False]}
        result.state.return_value = final_state
        yield result

    simulator._simulator_iterator.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    qubit_order = mock.Mock(cirq.QubitOrder)
    extensions = mock.Mock(cirq.Extensions)
    result = simulator.simulate(circuit=circuit,
                                param_resolver=param_resolver,
                                qubit_order=qubit_order,
                                initial_state=2,
                                extensions=extensions)
    np.testing.assert_equal(result.measurements['a'], [True, True])
    np.testing.assert_equal(result.measurements['b'], [True, False])
    assert set(result.measurements.keys()) == {'a', 'b'}
    assert result.params == param_resolver
    np.testing.assert_equal(result.final_state, final_state)


@mock.patch.multiple(cirq.WaveFunctionSimulator,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator_no_steps():
    simulator = cirq.WaveFunctionSimulator()

    initial_state = np.array([1, 0, 0, 0], dtype=np.complex64)

    simulator._simulator_iterator.return_value = iter([])
    circuit = cirq.testing.random_circuit(2, 20, 0.99)
    param_resolver = mock.Mock(cirq.ParamResolver)
    qubit_order = circuit.all_qubits()
    extensions = mock.Mock(cirq.Extensions)
    result = simulator.simulate(circuit=circuit,
                                param_resolver=param_resolver,
                                qubit_order=list(qubit_order),
                                initial_state=initial_state,
                                extensions=extensions)
    assert len(result.measurements) == 0
    assert result.params == param_resolver
    np.testing.assert_equal(result.final_state, initial_state)


@mock.patch.multiple(cirq.WaveFunctionSimulator,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator_sweeps():
    simulator = cirq.WaveFunctionSimulator()

    final_state = np.array([1, 0, 0, 0])
    def steps(*args, **kwargs):
        result = mock.Mock()
        result.measurements = {'a': np.array([True, True])}
        result.state.return_value = final_state
        yield result

    simulator._simulator_iterator.side_effect = steps
    circuit = mock.Mock(cirq.Circuit)
    param_resolvers = [mock.Mock(cirq.ParamResolver),
                       mock.Mock(cirq.ParamResolver)]
    qubit_order = mock.Mock(cirq.QubitOrder)
    extensions = mock.Mock(cirq.Extensions)
    results = simulator.simulate_sweep(program=circuit,
                                       params=param_resolvers,
                                       qubit_order=qubit_order,
                                       initial_state=2,
                                       extensions=extensions)
    expected_results = [
        cirq.SimulateTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[0],
            final_state=final_state),
        cirq.SimulateTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[1],
            final_state=final_state)
    ]
    assert results == expected_results


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_simulator_simulate_trial_result_repr():
    v = cirq.SimulateTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))

    assert repr(v) == ("SimulateTrialResult("
                       "params=cirq.ParamResolver({'a': 2}), "
                       "measurements={'m': array([1, 2])}, "
                       "final_state=array([0, 1, 0, 0]))")


def test_simulator_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.SimulateTrialResult(
            params=cirq.ParamResolver({'a': 2}),
            measurements={'m': np.array([1, 2])},
            final_state=np.array([0, 1, 0, 0])))
    eq.add_equality_group(
        cirq.SimulateTrialResult(
            params=cirq.ParamResolver({'a': 2}),
            measurements={'m': np.array([1, 2])},
            final_state=np.array([0, 0, 1, 0])))
    eq.add_equality_group(
        cirq.SimulateTrialResult(
            params=cirq.ParamResolver({'a': 3}),
            measurements={'m': np.array([1, 2])},
            final_state=np.array([0, 0, 1, 0])))


def test_simulator_trial_pretty_state():
    result = cirq.SimulateTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))
    assert result.pretty_state() == '|01⟩'


class BasicStepResult(cirq.StepResult):

    def __init__(self, qubit_map: Dict,
        measurements: Dict[str, List[bool]]) -> None:
        super().__init__(qubit_map, measurements)

    @property
    def state(self) -> np.ndarray:
        return np.array([0, 1, 0, 0])


def test_step_result_pretty_state():
    step_result = BasicStepResult({}, {})
    assert step_result.pretty_state() == '|01⟩'

