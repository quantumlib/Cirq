# Copyright 2019 The Cirq Developers
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

import numpy as np

import cirq
import cirq.testing


def test_state_vector_trial_result_repr():
    final_simulator_state = cirq.StateVectorSimulatorState(
        qubit_map={cirq.NamedQubit('a'): 0}, state_vector=np.array([0, 1]))
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state)
    assert repr(trial_result) == (
        "cirq.StateVectorTrialResult("
        "params=cirq.ParamResolver({'s': 1}), "
        "measurements={'m': array([[1]])}, "
        "final_simulator_state=cirq.StateVectorSimulatorState("
        "state_vector=np.array([0, 1]), "
        "qubit_map={cirq.NamedQubit('a'): 0}))")


def test_state_vector_simulator_state_repr():
    final_simulator_state = cirq.StateVectorSimulatorState(
        qubit_map={cirq.NamedQubit('a'): 0}, state_vector=np.array([0, 1]))
    cirq.testing.assert_equivalent_repr(final_simulator_state)


def test_state_vector_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=cirq.StateVectorSimulatorState(
                np.array([]), {})),
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=cirq.StateVectorSimulatorState(
                np.array([]), {})))
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_simulator_state=cirq.StateVectorSimulatorState(
                np.array([]), {})))
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=cirq.StateVectorSimulatorState(
                np.array([]), {})))
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=cirq.StateVectorSimulatorState(
                np.array([1]), {})))


def test_state_vector_trial_result_state_mixin():
    qubits = cirq.LineQubit.range(2)
    qubit_map = {qubits[i]: i for i in range(2)}
    result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_simulator_state=cirq.StateVectorSimulatorState(
            qubit_map=qubit_map, state_vector=np.array([0, 1, 0, 0])))
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, result.density_matrix_of(qubits))
    bloch = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch,
                                         result.bloch_vector_of(qubits[1]))
    assert result.dirac_notation() == '|01⟩'


def test_state_vector_trial_result_qid_shape():
    final_simulator_state = cirq.StateVectorSimulatorState(
        qubit_map={cirq.NamedQubit('a'): 0}, state_vector=np.array([0, 1]))
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state)
    assert cirq.qid_shape(final_simulator_state) == (2,)
    assert cirq.qid_shape(trial_result) == (2,)

    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    final_simulator_state = cirq.StateVectorSimulatorState(
        qubit_map={
            q0: 1,
            q1: 0
        }, state_vector=np.array([0, 0, 0, 0, 1, 0]))
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[2, 0]])},
        final_simulator_state=final_simulator_state)
    assert cirq.qid_shape(final_simulator_state) == (3, 2)
    assert cirq.qid_shape(trial_result) == (3, 2)


def test_str_big():
    qs = cirq.LineQubit.range(20)
    result = cirq.StateVectorTrialResult(
        cirq.ParamResolver(), {},
        cirq.StateVectorSimulatorState(np.array([1] * 2**10),
                                       {q: q.x for q in qs}))
    assert str(result).startswith('measurements: (no measurements)\n'
                                  'output vector: [1 1 1 ..')


def test_pretty_print():
    result = cirq.StateVectorTrialResult(
        cirq.ParamResolver(), {},
        cirq.StateVectorSimulatorState(np.array([1]), {}))

    # Test Jupyter console output from
    class FakePrinter:

        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == 'measurements: (no measurements)\noutput vector: |⟩'

    # Test cycle handling
    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'StateVectorTrialResult(...)'


def test_deprecated():
    with cirq.testing.assert_logs('WaveFunctionTrialResult',
                                  'StateVectorTrialResult', 'deprecated'):
        _ = cirq.sim.WaveFunctionTrialResult(
            cirq.ParamResolver(), {},
            cirq.StateVectorSimulatorState(np.array([1]), {}))

    with cirq.testing.assert_logs('final_state', 'final_state_vector',
                                  'deprecated'):
        _ = cirq.sim.StateVectorTrialResult(
            cirq.ParamResolver(), {},
            cirq.StateVectorSimulatorState(np.array([1]), {})).final_state

    with cirq.testing.assert_logs('WaveFunctionSimulatorState',
                                  'StateVectorSimulatorState', 'deprecated'):
        _ = cirq.sim.WaveFunctionSimulatorState(np.array([1]), {})

    class TestStepResult(cirq.sim.WaveFunctionStepResult):

        def _simulator_state(self):
            pass

        def _simulator_state(self):
            pass

        def sample(self, qubits, repetitions, seed):
            pass

    with cirq.testing.assert_logs('WaveFunctionStepResult',
                                  'StateVectorStepResult', 'deprecated'):
        _ = TestStepResult()

    class TestSimulatesClass(cirq.sim.SimulatesIntermediateWaveFunction):

        def _simulator_iterator(self, circuit, param_resolver, qubit_order,
                                initial_state):
            pass

    with cirq.testing.assert_logs('SimulatesIntermediateWaveFunction',
                                  'SimulatesIntermediateStateVector',
                                  'deprecated'):
        _ = TestSimulatesClass()
