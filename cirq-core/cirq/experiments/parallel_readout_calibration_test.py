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
from typing import List
import pytest

import numpy as np

import cirq


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
        self.simulator = cirq.Simulator(seed=self.prng)

    def run_sweep(
        self,
        program: 'cirq.Circuit',
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> List[cirq.Result]:
        results = self.simulator.run_sweep(program, params, repetitions)
        for result in results:
            for bits in result.measurements.values():
                with np.nditer(bits, op_flags=['readwrite']) as it:
                    for x in it:
                        if x == 0 and self.prng.uniform() < self.p0:
                            x[...] = 1
                        elif self.prng.uniform() < self.p1:
                            x[...] = 0
        return results


def test_estimate_parallel_readout_errors_no_noise():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.Simulator()
    repetitions = 1000
    result = cirq.estimate_parallel_readout_errors(sampler, qubits=qubits, repetitions=repetitions)
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_single_qubit_readout_errors_all_zeros():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.ZerosSampler()
    repetitions = 1000
    result = cirq.estimate_parallel_readout_errors(sampler, qubits=qubits, repetitions=repetitions)
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 1 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_single_qubit_readout_errors_bad_bit_string():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='providing bit_string'):
        _ = cirq.estimate_parallel_readout_errors(
            cirq.ZerosSampler(),
            qubits=qubits,
            repetitions=1000,
            trials=35,
            trials_per_batch=10,
            bit_strings=[1, 1, 1, 1],
        )


def test_estimate_single_qubit_readout_errors_batching():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.ZerosSampler()
    repetitions = 1000
    result = cirq.estimate_parallel_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions, trials=35, trials_per_batch=10
    )
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 1 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)


def test_estimate_single_qubit_readout_errors_with_noise():
    qubits = cirq.LineQubit.range(5)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    repetitions = 1000
    result = cirq.estimate_parallel_readout_errors(
        sampler, qubits=qubits, repetitions=repetitions, trials=40
    )
    for error in result.one_state_errors.values():
        assert 0.18 < error < 0.22
    for error in result.zero_state_errors.values():
        assert 0.08 < error < 0.12
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)
