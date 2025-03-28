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
from typing import Sequence

import numpy as np
import pytest

import cirq


def test_single_qubit_readout_result_repr():
    result = cirq.experiments.SingleQubitReadoutCalibrationResult(
        zero_state_errors={cirq.LineQubit(0): 0.1},
        one_state_errors={cirq.LineQubit(0): 0.2},
        repetitions=1000,
        timestamp=0.3,
    )
    cirq.testing.assert_equivalent_repr(result)


class NoisySingleQubitReadoutSampler(cirq.Sampler):
    def __init__(self, p0: float, p1: float, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        """Sampler that flips some bits upon readout.

        Args:
            p0: Probability of flipping a 0 to a 1.
            p1: Probability of flipping a 1 to a 0.
            seed: A seed for the pseudorandom number generator.
        """
        self.p0 = p0
        self.p1 = p1
        self.prng = cirq.value.parse_random_state(seed)
        self.simulator = cirq.Simulator(seed=self.prng, split_untangled_states=False)

    def run_sweep(
        self, program: 'cirq.AbstractCircuit', params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence[cirq.Result]:
        results = self.simulator.run_sweep(program, params, repetitions)
        for result in results:
            for bits in result.measurements.values():
                rand_num = self.prng.uniform(size=bits.shape)
                should_flip = np.logical_or(
                    np.logical_and(bits == 0, rand_num < self.p0),
                    np.logical_and(bits == 1, rand_num < self.p1),
                )
                bits[should_flip] ^= 1
        return results


def test_estimate_single_qubit_readout_errors_no_noise():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.Simulator()
    repetitions = 1000
    result = cirq.estimate_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions
    )
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_single_qubit_readout_errors_with_noise():
    qubits = cirq.LineQubit.range(5)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    repetitions = 1000
    result = cirq.estimate_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions
    )
    for error in result.zero_state_errors.values():
        assert 0.08 < error < 0.12
    for error in result.one_state_errors.values():
        assert 0.18 < error < 0.22
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_parallel_readout_errors_no_noise():
    qubits = [cirq.GridQubit(i, 0) for i in range(10)]
    sampler = cirq.Simulator()
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions
    )
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)
    _ = result.plot_integrated_histogram()
    _, _ = result.plot_heatmap()


def test_estimate_parallel_readout_errors_all_zeros():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.ZerosSampler()
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions
    )
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 1 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_parallel_readout_errors_bad_bit_string():
    qubits = cirq.LineQubit.range(4)
    with pytest.raises(ValueError, match='but was None'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(
            cirq.ZerosSampler(),
            qubits=qubits,
            repetitions=1000,
            trials=35,
            trials_per_batch=10,
            bit_strings=[[1] * 4],
        )
    with pytest.raises(ValueError, match='0 or 1'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(
            cirq.ZerosSampler(),
            qubits=qubits,
            repetitions=1000,
            trials=2,
            bit_strings=np.array([[12, 47, 2, -4], [0.1, 7, 0, 0]]),
        )


def test_estimate_parallel_readout_errors_zero_reps():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='non-zero repetition'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(
            cirq.ZerosSampler(), qubits=qubits, repetitions=0, trials=35, trials_per_batch=10
        )


def test_estimate_parallel_readout_errors_zero_trials():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='non-zero trials'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(
            cirq.ZerosSampler(), qubits=qubits, repetitions=1000, trials=0, trials_per_batch=10
        )


def test_estimate_parallel_readout_errors_zero_batch():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='non-zero trials_per_batch'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(
            cirq.ZerosSampler(), qubits=qubits, repetitions=1000, trials=10, trials_per_batch=0
        )


def test_estimate_parallel_readout_errors_batching():
    qubits = cirq.LineQubit.range(5)
    sampler = cirq.ZerosSampler()
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions, trials=35, trials_per_batch=10
    )
    assert result.zero_state_errors == {q: 0.0 for q in qubits}
    assert result.one_state_errors == {q: 1.0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_parallel_readout_errors_with_noise():
    qubits = cirq.LineQubit.range(5)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions, trials=40
    )
    for error in result.one_state_errors.values():
        assert 0.17 < error < 0.23
    for error in result.zero_state_errors.values():
        assert 0.07 < error < 0.13
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_parallel_readout_errors_missing_qubits():
    qubits = cirq.LineQubit.range(4)

    result = cirq.estimate_parallel_single_qubit_readout_errors(
        cirq.ZerosSampler(),
        qubits=qubits,
        repetitions=2000,
        trials=1,
        bit_strings=np.array([[0] * 4]),
    )
    assert result.zero_state_errors == {q: 0 for q in qubits}
    # Trial did not include a one-state
    assert all(np.isnan(result.one_state_errors[q]) for q in qubits)
    assert result.repetitions == 2000
    assert isinstance(result.timestamp, float)
