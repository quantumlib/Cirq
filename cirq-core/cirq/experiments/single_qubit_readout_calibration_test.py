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
        self.simulator = cirq.Simulator(seed=self.prng, split_untangled_states=False)

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


def test_single_qubit_readout_calibration_result_repr():
    result = cirq.experiments.SingleQubitReadoutCalibrationResult(
        zero_state_errors={cirq.LineQubit(0): 0.1},
        one_state_errors={cirq.LineQubit(0): 0.2},
        repetitions=1000,
        timestamp=0.3,
    )
    cirq.testing.assert_equivalent_repr(result)
