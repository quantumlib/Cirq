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

from unittest import mock

import numpy as np

import cirq
import cirq.testing


def test_state_vector_trial_result_repr():
    q0 = cirq.NamedQubit('a')
    args = cirq.ActOnStateVectorArgs(
        target_tensor=np.array([0, 1], dtype=np.int32),
        available_buffer=np.array([0, 1], dtype=np.int32),
        prng=np.random.RandomState(0),
        log_of_measurement_results={},
        qubits=[q0],
    )
    final_step_result = cirq.SparseSimulatorStep(args, cirq.Simulator())
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]], dtype=np.int32)},
        final_step_result=final_step_result,
    )
    expected_repr = (
        "cirq.StateVectorTrialResult("
        "params=cirq.ParamResolver({'s': 1}), "
        "measurements={'m': np.array([[1]], dtype=np.int32)}, "
        "final_step_result=cirq.SparseSimulatorStep("
        "sim_state=cirq.ActOnStateVectorArgs("
        "target_tensor=np.array([0, 1], dtype=np.int32), "
        "available_buffer=np.array([0, 1], dtype=np.int32), "
        "qubits=(cirq.NamedQubit('a'),), "
        "log_of_measurement_results={}), "
        "dtype=np.complex64))"
    )
    assert repr(trial_result) == expected_repr
    assert eval(expected_repr) == trial_result


def test_state_vector_simulator_state_repr():
    final_simulator_state = cirq.StateVectorSimulatorState(
        qubit_map={cirq.NamedQubit('a'): 0}, state_vector=np.array([0, 1])
    )
    cirq.testing.assert_equivalent_repr(final_simulator_state)


def test_state_vector_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    final_step_result = mock.Mock(cirq.StateVectorStepResult)
    final_step_result._qubit_mapping = {}
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        np.array([]), {}
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_step_result=final_step_result,
        ),
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_step_result=final_step_result,
        ),
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_step_result=final_step_result,
        )
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_step_result=final_step_result,
        )
    )
    final_step_result = mock.Mock(cirq.StateVectorStepResult)
    final_step_result._qubit_mapping = {}
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        np.array([1]), {}
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_step_result=final_step_result,
        )
    )


def test_state_vector_trial_result_state_mixin():
    qubits = cirq.LineQubit.range(2)
    qubit_map = {qubits[i]: i for i in range(2)}
    final_step_result = mock.Mock(cirq.StateVectorStepResult)
    final_step_result._qubit_mapping = qubit_map
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        qubit_map=qubit_map, state_vector=np.array([0, 1, 0, 0])
    )
    result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_step_result=final_step_result,
    )
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, result.density_matrix_of(qubits))
    bloch = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, result.bloch_vector_of(qubits[1]))
    assert result.dirac_notation() == '|01⟩'


def test_state_vector_trial_result_qid_shape():
    qubit_map = {cirq.NamedQubit('a'): 0}
    final_step_result = mock.Mock(cirq.StateVectorStepResult)
    final_step_result._qubit_mapping = qubit_map
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        qubit_map=qubit_map, state_vector=np.array([0, 1])
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_step_result=final_step_result,
    )
    assert cirq.qid_shape(final_step_result._simulator_state()) == (2,)
    assert cirq.qid_shape(trial_result) == (2,)

    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    qubit_map = {q0: 1, q1: 0}
    final_step_result._qubit_mapping = qubit_map
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        qubit_map=qubit_map, state_vector=np.array([0, 0, 0, 0, 1, 0])
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[2, 0]])},
        final_step_result=final_step_result,
    )
    assert cirq.qid_shape(final_step_result._simulator_state()) == (3, 2)
    assert cirq.qid_shape(trial_result) == (3, 2)


def test_state_vector_trial_state_vector_is_copy():
    final_state_vector = np.array([0, 1])
    qubit_map = {cirq.NamedQubit('a'): 0}
    final_step_result = mock.Mock(cirq.StateVectorStepResult)
    final_step_result._qubit_mapping = qubit_map
    final_step_result._simulator_state.return_value = cirq.StateVectorSimulatorState(
        qubit_map=qubit_map, state_vector=final_state_vector
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({}), measurements={}, final_step_result=final_step_result
    )
    assert final_step_result._simulator_state().state_vector is final_state_vector
    assert trial_result.state_vector() is not final_state_vector


def test_str_big():
    qs = cirq.LineQubit.range(20)
    args = cirq.ActOnStateVectorArgs(
        target_tensor=np.array([1] * 2 ** 10),
        available_buffer=np.array([1] * 2 ** 10),
        prng=np.random.RandomState(0),
        log_of_measurement_results={},
        qubits=qs,
    )
    final_step_result = cirq.SparseSimulatorStep(args, cirq.Simulator())
    result = cirq.StateVectorTrialResult(
        cirq.ParamResolver(),
        {},
        final_step_result,
    )
    assert 'output vector: [1 1 1 ..' in str(result)


def test_pretty_print():
    args = cirq.ActOnStateVectorArgs(
        target_tensor=np.array([1]),
        available_buffer=np.array([1]),
        prng=np.random.RandomState(0),
        log_of_measurement_results={},
        qubits=[],
    )
    final_step_result = cirq.SparseSimulatorStep(args, cirq.Simulator())
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_step_result)

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == 'measurements: (no measurements)\n\nphase:\noutput vector: |⟩'

    # Test cycle handling
    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'StateVectorTrialResult(...)'
