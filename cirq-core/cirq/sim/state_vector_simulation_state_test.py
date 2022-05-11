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

from unittest import mock

import numpy as np
import pytest

import cirq


def test_default_parameter():
    dtype = np.complex64
    tensor = cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64)
    qubits = cirq.LineQubit.range(3)
    args = cirq.StateVectorSimulationState(qubits=qubits, initial_state=tensor, dtype=dtype)
    qid_shape = cirq.protocols.qid_shape(qubits)
    tensor = np.reshape(tensor, qid_shape)
    np.testing.assert_almost_equal(args.target_tensor, tensor)
    assert args.available_buffer.shape == tensor.shape
    assert args.available_buffer.dtype == tensor.dtype


def test_infer_target_tensor():
    dtype = np.complex64
    args = cirq.StateVectorSimulationState(
        qubits=cirq.LineQubit.range(2),
        initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype),
        dtype=dtype,
    )
    np.testing.assert_almost_equal(
        args.target_tensor,
        np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    )

    args = cirq.StateVectorSimulationState(
        qubits=cirq.LineQubit.range(2), initial_state=0, dtype=dtype
    )
    np.testing.assert_almost_equal(
        args.target_tensor,
        np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    )


def test_shallow_copy_buffers():
    args = cirq.StateVectorSimulationState(qubits=cirq.LineQubit.range(1), initial_state=0)
    copy = args.copy(deep_copy_buffers=False)
    assert copy.available_buffer is args.available_buffer


def test_decomposed_fallback():
    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty((2, 2, 2), dtype=np.complex64),
        qubits=cirq.LineQubit.range(3),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64),
        dtype=np.complex64,
    )

    cirq.act_on(Composite(), args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(
        args.target_tensor, cirq.one_hot(index=(0, 1, 0), shape=(2, 2, 2), dtype=np.complex64)
    )


def test_cannot_act():
    class NoDetails:
        pass

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty((2, 2, 2), dtype=np.complex64),
        qubits=cirq.LineQubit.range(3),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64),
        dtype=np.complex64,
    )

    with pytest.raises(TypeError, match="Can't simulate operations"):
        cirq.act_on(NoDetails(), args, qubits=())


def test_act_using_probabilistic_single_qubit_channel():
    class ProbabilisticSorX(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _kraus_(self):
            return [cirq.unitary(cirq.S) * np.sqrt(1 / 3), cirq.unitary(cirq.X) * np.sqrt(2 / 3)]

    initial_state = cirq.testing.random_superposition(dim=16).reshape((2,) * 4)
    mock_prng = mock.Mock()

    mock_prng.random.return_value = 1 / 3 + 1e-6
    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty_like(initial_state),
        qubits=cirq.LineQubit.range(4),
        prng=mock_prng,
        initial_state=np.copy(initial_state),
        dtype=initial_state.dtype,
    )
    cirq.act_on(ProbabilisticSorX(), args, [cirq.LineQubit(2)])
    np.testing.assert_allclose(
        args.target_tensor.reshape(16),
        cirq.final_state_vector(
            cirq.X(cirq.LineQubit(2)) ** -1,
            initial_state=initial_state,
            qubit_order=cirq.LineQubit.range(4),
        ),
        atol=1e-8,
    )

    mock_prng.random.return_value = 1 / 3 - 1e-6
    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty_like(initial_state),
        qubits=cirq.LineQubit.range(4),
        prng=mock_prng,
        initial_state=np.copy(initial_state),
        dtype=initial_state.dtype,
    )
    cirq.act_on(ProbabilisticSorX(), args, [cirq.LineQubit(2)])
    np.testing.assert_allclose(
        args.target_tensor.reshape(16),
        cirq.final_state_vector(
            cirq.S(cirq.LineQubit(2)),
            initial_state=initial_state,
            qubit_order=cirq.LineQubit.range(4),
        ),
        atol=1e-8,
    )


def test_act_using_adaptive_two_qubit_channel():
    class Decay11(cirq.Gate):
        def num_qubits(self) -> int:
            return 2

        def _kraus_(self):
            bottom_right = cirq.one_hot(index=(3, 3), shape=(4, 4), dtype=np.complex64)
            top_right = cirq.one_hot(index=(0, 3), shape=(4, 4), dtype=np.complex64)
            return [
                np.eye(4) * np.sqrt(3 / 4),
                (np.eye(4) - bottom_right) * np.sqrt(1 / 4),
                top_right * np.sqrt(1 / 4),
            ]

    mock_prng = mock.Mock()

    def get_result(state: np.ndarray, sample: float):
        mock_prng.random.return_value = sample
        args = cirq.StateVectorSimulationState(
            available_buffer=np.empty_like(state),
            qubits=cirq.LineQubit.range(4),
            prng=mock_prng,
            initial_state=np.copy(state),
            dtype=state.dtype,
        )
        cirq.act_on(Decay11(), args, [cirq.LineQubit(1), cirq.LineQubit(3)])
        return args.target_tensor

    def assert_not_affected(state: np.ndarray, sample: float):
        np.testing.assert_allclose(get_result(state, sample), state, atol=1e-8)

    all_zeroes = cirq.one_hot(index=(0, 0, 0, 0), shape=(2,) * 4, dtype=np.complex128)
    all_ones = cirq.one_hot(index=(1, 1, 1, 1), shape=(2,) * 4, dtype=np.complex128)
    decayed_all_ones = cirq.one_hot(index=(1, 0, 1, 0), shape=(2,) * 4, dtype=np.complex128)

    # Decays the 11 state to 00.
    np.testing.assert_allclose(get_result(all_ones, 3 / 4 - 1e-8), all_ones)
    np.testing.assert_allclose(get_result(all_ones, 3 / 4 + 1e-8), decayed_all_ones)

    # Decoheres the 11 subspace from other subspaces as sample rises.
    superpose = all_ones * np.sqrt(1 / 2) + all_zeroes * np.sqrt(1 / 2)
    np.testing.assert_allclose(get_result(superpose, 3 / 4 - 1e-8), superpose)
    np.testing.assert_allclose(get_result(superpose, 3 / 4 + 1e-8), all_zeroes)
    np.testing.assert_allclose(get_result(superpose, 7 / 8 - 1e-8), all_zeroes)
    np.testing.assert_allclose(get_result(superpose, 7 / 8 + 1e-8), decayed_all_ones)

    # Always acts like identity when sample < p=3/4.
    for _ in range(10):
        assert_not_affected(
            cirq.testing.random_superposition(dim=16).reshape((2,) * 4), sample=3 / 4 - 1e-8
        )

    # Acts like identity on superpositions of first three states.
    for _ in range(10):
        mock_prng.random.return_value = 3 / 4 + 1e-6
        projected_state = cirq.testing.random_superposition(dim=16).reshape((2,) * 4)
        projected_state[cirq.slice_for_qubits_equal_to([1, 3], 3)] = 0
        projected_state /= np.linalg.norm(projected_state)
        assert abs(np.linalg.norm(projected_state) - 1) < 1e-8
        assert_not_affected(projected_state, sample=3 / 4 + 1e-8)


def test_probability_comes_up_short_results_in_fallback():
    class Short(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _kraus_(self):
            return [cirq.unitary(cirq.X) * np.sqrt(0.999), np.eye(2) * 0]

    mock_prng = mock.Mock()
    mock_prng.random.return_value = 0.9999

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(2, dtype=np.complex64),
        qubits=cirq.LineQubit.range(1),
        prng=mock_prng,
        initial_state=np.array([1, 0], dtype=np.complex64),
        dtype=np.complex64,
    )

    cirq.act_on(Short(), args, cirq.LineQubit.range(1))

    np.testing.assert_allclose(args.target_tensor, np.array([0, 1]))


def test_random_channel_has_random_behavior():
    q = cirq.LineQubit(0)
    s = cirq.Simulator().sample(
        cirq.Circuit(cirq.X(q), cirq.amplitude_damp(0.4).on(q), cirq.measure(q, key='out')),
        repetitions=100,
    )
    v = s['out'].value_counts()
    assert v[0] > 1
    assert v[1] > 1


def test_measured_channel():
    # This behaves like an X-basis measurement.
    kc = cirq.KrausChannel(
        kraus_ops=(np.array([[1, 1], [1, 1]]) * 0.5, np.array([[1, -1], [-1, 1]]) * 0.5), key='m'
    )
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), kc.on(q0))
    sim = cirq.Simulator(seed=0)
    results = sim.run(circuit, repetitions=100)
    assert results.histogram(key='m') == {0: 100}


def test_measured_mixture():
    # This behaves like an X-basis measurement.
    mm = cirq.MixedUnitaryChannel(
        mixture=((0.5, np.array([[1, 0], [0, 1]])), (0.5, np.array([[0, 1], [1, 0]]))), key='flip'
    )
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(mm.on(q0), cirq.measure(q0, key='m'))
    sim = cirq.Simulator(seed=0)
    results = sim.run(circuit, repetitions=100)
    assert results.histogram(key='flip') == results.histogram(key='m')


def test_with_qubits():
    original = cirq.StateVectorSimulationState(
        qubits=cirq.LineQubit.range(2), initial_state=1, dtype=np.complex64
    )
    extened = original.with_qubits(cirq.LineQubit.range(2, 4))
    np.testing.assert_almost_equal(
        extened.target_tensor,
        cirq.state_vector_kronecker_product(
            np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64),
            np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64),
        ),
    )


def test_qid_shape_error():
    with pytest.raises(ValueError, match="qid_shape must be provided"):
        cirq.sim.state_vector_simulation_state._BufferedStateVector.create(initial_state=0)


def test_deprecated_methods():
    args = cirq.StateVectorSimulationState(qubits=[cirq.LineQubit(0)])
    with cirq.testing.assert_deprecated('unintentionally made public', deadline='v0.16'):
        args.subspace_index([0], 0)
    with cirq.testing.assert_deprecated('unintentionally made public', deadline='v0.16'):
        args.swap_target_tensor_for(np.array([]))
