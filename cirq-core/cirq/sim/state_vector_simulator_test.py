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
    q0 = cirq.NamedQubit('a')
    final_simulator_state = cirq.StateVectorSimulationState(
        available_buffer=np.array([0, 1], dtype=np.complex64),
        prng=np.random.RandomState(0),
        qubits=[q0],
        initial_state=np.array([0, 1], dtype=np.complex64),
        dtype=np.complex64,
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]], dtype=np.int32)},
        final_simulator_state=final_simulator_state,
    )
    expected_repr = (
        "cirq.StateVectorTrialResult("
        "params=cirq.ParamResolver({'s': 1}), "
        "measurements={'m': np.array([[1]], dtype=np.int32)}, "
        "final_simulator_state=cirq.StateVectorSimulationState("
        "initial_state=np.array([0j, (1+0j)], dtype=np.complex64), "
        "qubits=(cirq.NamedQubit('a'),), "
        "classical_data=cirq.ClassicalDataDictionaryStore()))"
    )
    assert repr(trial_result) == expected_repr
    assert eval(expected_repr) == trial_result


def test_state_vector_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    final_simulator_state = cirq.StateVectorSimulationState(initial_state=np.array([]))
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state,
        ),
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state,
        ),
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_simulator_state=final_simulator_state,
        )
    )
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=final_simulator_state,
        )
    )
    final_simulator_state = cirq.StateVectorSimulationState(initial_state=np.array([1]))
    eq.add_equality_group(
        cirq.StateVectorTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=final_simulator_state,
        )
    )


def test_state_vector_trial_result_state_mixin():
    qubits = cirq.LineQubit.range(2)
    final_simulator_state = cirq.StateVectorSimulationState(
        qubits=qubits, initial_state=np.array([0, 1, 0, 0])
    )
    result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'m': np.array([1, 2])},
        final_simulator_state=final_simulator_state,
    )
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, result.density_matrix_of(qubits))
    bloch = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, result.bloch_vector_of(qubits[1]))
    assert result.dirac_notation() == '|01⟩'


def test_state_vector_trial_result_qid_shape():
    final_simulator_state = cirq.StateVectorSimulationState(
        qubits=[cirq.NamedQubit('a')], initial_state=np.array([0, 1])
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state,
    )
    assert cirq.qid_shape(trial_result) == (2,)

    final_simulator_state = cirq.StateVectorSimulationState(
        qubits=cirq.LineQid.for_qid_shape((3, 2)), initial_state=np.array([0, 0, 0, 0, 1, 0])
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[2, 0]])},
        final_simulator_state=final_simulator_state,
    )
    assert cirq.qid_shape(trial_result) == (3, 2)


def test_state_vector_trial_state_vector_is_copy():
    final_state_vector = np.array([0, 1], dtype=np.complex64)
    qubit_map = {cirq.NamedQubit('a'): 0}
    final_simulator_state = cirq.StateVectorSimulationState(
        qubits=list(qubit_map), initial_state=final_state_vector
    )
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state
    )
    assert trial_result.state_vector(copy=True) is not final_simulator_state.target_tensor


def test_state_vector_trial_result_no_qubits():
    initial_state_vector = np.array([1], dtype=np.complex64)
    initial_state = initial_state_vector.reshape((2,) * 0)  # reshape as tensor for 0 qubits
    final_simulator_state = cirq.StateVectorSimulationState(qubits=[], initial_state=initial_state)
    trial_result = cirq.StateVectorTrialResult(
        params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state
    )
    state_vector = trial_result.state_vector()
    assert state_vector.shape == (1,)
    assert np.array_equal(state_vector, initial_state_vector)


def test_str_big():
    qs = cirq.LineQubit.range(10)
    final_simulator_state = cirq.StateVectorSimulationState(
        prng=np.random.RandomState(0),
        qubits=qs,
        initial_state=np.array([1] * 2**10, dtype=np.complex64) * 0.03125,
        dtype=np.complex64,
    )
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_simulator_state)
    assert 'output vector: [0.03125+0.j 0.03125+0.j 0.03125+0.j ..' in str(result)


def test_pretty_print():
    final_simulator_state = cirq.StateVectorSimulationState(
        available_buffer=np.array([1]),
        prng=np.random.RandomState(0),
        qubits=[],
        initial_state=np.array([1], dtype=np.complex64),
        dtype=np.complex64,
    )
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_simulator_state)

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
