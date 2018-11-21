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
import pytest

import cirq
from cirq.testing.mock import mock


@mock.patch.multiple(cirq.SimulatesSamples, _run=mock.Mock())
def test_run_simulator_run():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': [[1]]}
    simulator._run.return_value = expected_measurements
    circuit = mock.Mock(cirq.Circuit)
    param_resolver = mock.Mock(cirq.ParamResolver)
    expected_result = cirq.TrialResult(repetitions=10,
                                       measurements=expected_measurements,
                                       params=param_resolver)
    assert expected_result == simulator.run(circuit=circuit,
                                            repetitions=10,
                                            param_resolver=param_resolver)
    simulator._run.assert_called_once_with(circuit=circuit,
                                           repetitions=10,
                                           param_resolver=param_resolver)


@mock.patch.multiple(cirq.SimulatesSamples, _run=mock.Mock())
def test_run_simulator_sweeps():
    simulator = cirq.SimulatesSamples()
    expected_measurements = {'a': [[1]]}
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


@mock.patch.multiple(cirq.SimulatesIntermediateWaveFunction,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator():
    simulator = cirq.SimulatesIntermediateWaveFunction()

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
    result = simulator.simulate(circuit=circuit,
                                param_resolver=param_resolver,
                                qubit_order=qubit_order,
                                initial_state=2)
    np.testing.assert_equal(result.measurements['a'], [True, True])
    np.testing.assert_equal(result.measurements['b'], [True, False])
    assert set(result.measurements.keys()) == {'a', 'b'}
    assert result.params == param_resolver
    np.testing.assert_equal(result.final_state, final_state)


@mock.patch.multiple(cirq.SimulatesIntermediateWaveFunction,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator_no_steps():
    simulator = cirq.SimulatesIntermediateWaveFunction()

    initial_state = np.array([1, 0, 0, 0], dtype=np.complex64)

    simulator._simulator_iterator.return_value = iter([])
    circuit = cirq.testing.random_circuit(2, 20, 0.99)
    param_resolver = mock.Mock(cirq.ParamResolver)
    qubit_order = circuit.all_qubits()
    result = simulator.simulate(circuit=circuit,
                                param_resolver=param_resolver,
                                qubit_order=list(qubit_order),
                                initial_state=initial_state)
    assert len(result.measurements) == 0
    assert result.params == param_resolver
    np.testing.assert_equal(result.final_state, initial_state)


@mock.patch.multiple(cirq.SimulatesIntermediateWaveFunction,
                     _simulator_iterator=mock.Mock())
def test_wave_simulator_sweeps():
    simulator = cirq.SimulatesIntermediateWaveFunction()

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
    results = simulator.simulate_sweep(program=circuit,
                                       params=param_resolvers,
                                       qubit_order=qubit_order,
                                       initial_state=2)
    expected_results = [
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[0],
            final_state=final_state),
        cirq.SimulationTrialResult(
            measurements={'a': np.array([True, True])},
            params=param_resolvers[1],
            final_state=final_state)
    ]
    assert results == expected_results


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_simulator_simulate_trial_result_repr():
    v = cirq.SimulationTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))

    assert repr(v) == ("SimulationTrialResult("
                       "params=cirq.ParamResolver({'a': 2}), "
                       "measurements={'m': array([1, 2])}, "
                       "final_state=array([0, 1, 0, 0]))")


def test_simulator_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'a': 2}),
            measurements={'m': np.array([1, 0])},
            final_state=np.array([0, 1, 0, 0])))
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'a': 2}),
            measurements={'m': np.array([1, 0])},
            final_state=np.array([0, 0, 1, 0])))
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'a': 3}),
            measurements={'m': np.array([1, 0])},
            final_state=np.array([0, 0, 1, 0])))
    eq.add_equality_group(
        cirq.SimulationTrialResult(
            params=cirq.ParamResolver({'a': 3}),
            measurements={'m': np.array([0, 1])},
            final_state=np.array([0, 0, 1, 0])))


def test_simulator_trial_pretty_state():
    result = cirq.SimulationTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))
    assert result.dirac_notation() == '|01⟩'


def test_simulator_trial_density_matrix():
    result = cirq.SimulationTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))
    rho = np.array([[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho,
        result.density_matrix())


def test_simulator_trial_bloch_vector():
    result = cirq.SimulationTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_state=np.array([0, 1, 0, 0]))
    bloch = np.array([0,0,-1])
    np.testing.assert_array_almost_equal(bloch,
        result.bloch_vector(1))


def test_step_result_pretty_state():
    class BasicStepResult(cirq.StepResult):

        def __init__(self, qubit_map: Dict,
                measurements: Dict[str, List[bool]]) -> None:
            super().__init__(qubit_map, measurements)

        def state(self) -> np.ndarray:
            return np.array([0, 1, 0, 0])

    step_result = BasicStepResult({}, {})
    assert step_result.dirac_notation() == '|01⟩'


def test_step_result_density_matrix():
    class BasicStepResult(cirq.StepResult):

        def __init__(self, qubit_map: Dict,
                measurements: Dict[str, List[bool]]) -> None:
            super().__init__(qubit_map, measurements)

        def state(self) -> np.ndarray:
            return np.array([0, 1, 0, 0])

    step_result = BasicStepResult({}, {})
    rho = np.array([[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho,
        step_result.density_matrix())


def test_step_result_bloch_vector():
    class BasicStepResult(cirq.StepResult):

        def __init__(self, qubit_map: Dict,
                measurements: Dict[str, List[bool]]) -> None:
            super().__init__(qubit_map, measurements)

        def state(self) -> np.ndarray:
            return np.array([0, 1, 0, 0])

    step_result = BasicStepResult({}, {})
    bloch = np.array([0,0,-1])
    np.testing.assert_array_almost_equal(bloch,
        step_result.bloch_vector(1))


class FakeStepResult(cirq.StepResult):

    def __init__(self, ones_qubits):
        self._ones_qubits = set(ones_qubits)

    def state(self):
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


class MultiHTestGate(cirq.Gate):
    def _decompose_(self, qubits):
        return cirq.H.on_each(qubits)


def test_simulates_composite():
    c = cirq.Circuit.from_ops(MultiHTestGate().on(*cirq.LineQubit.range(2)))
    expected = np.array([0.5] * 4)
    np.testing.assert_allclose(c.apply_unitary_effect_to_state(),
                               expected)
    np.testing.assert_allclose(cirq.Simulator().simulate(c).final_state,
                               expected)


def test_simulate_measurement_inversions():
    q = cirq.NamedQubit('q')

    c = cirq.Circuit.from_ops(cirq.MeasurementGate(key='q',
                                                   invert_mask=(True,)).on(q))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([True])}

    c = cirq.Circuit.from_ops(cirq.MeasurementGate(key='q',
                                                   invert_mask=(False,)).on(q))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([False])}
