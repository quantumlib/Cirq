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

import collections
import random
from typing import Optional

import numpy as np

from cirq import circuits, study, ops, value
from cirq.work import collector


class PauliStringSampleCollector(collector.SampleCollector[ops.PauliString,
                                                           float]):
    """Estimates the energy of a linear combination of Pauli observables."""

    def __init__(self,
                 circuit: circuits.Circuit,
                 terms: value.LinearDict[ops.PauliString],
                 samples_per_job: int = 1000):
        """
        Args:
            circuit: Produces the state to be tested.
            terms: The pauli product observables to measure. Their sampled
                expectations will be scaled by their coefficients and their
                dictionary weights, and then added up to produce the final
                result.
            samples_per_job: How many samples to request at a time.
        """
        self._circuit = circuit
        self._samples_per_job = 1000
        self._terms = value.LinearDict({
            abs(k): v * k.coefficient for k, v in terms.items()
        })
        self._zeros = collections.defaultdict(lambda: 0)
        self._ones = collections.defaultdict(lambda: 0)

    def next_job(self) -> Optional[collector.CircuitSampleJob[ops.PauliString]]:
        pauli = random.choice(list(self._terms.keys()))
        return collector.CircuitSampleJob(
            circuit=_circuit_plus_pauli_string_measurements(self._circuit,
                                                            pauli),
            repetitions=self._samples_per_job,
            key=pauli)

    def on_job_result(self, job: collector.CircuitSampleJob, result: study.TrialResult):
        parities = result.histogram(
            key='out',
            fold_func=lambda bits: np.sum(bits) % 2)
        self._zeros[job.key] += parities[0]
        self._ones[job.key] += parities[1]

    def result(self) -> float:
        energy = 0
        for term, coef in self._terms.items():
            a = self._zeros[term]
            b = self._ones[term]
            if a + b:
                energy += coef * (a - b) / (a + b)
        return energy


def _circuit_plus_pauli_string_measurements(circuit: circuits.Circuit,
                                            pauli_string: ops.PauliString
                                            ) -> circuits.Circuit:
    """A circuit measuring the given observable at the end of the given circuit.
    """
    assert pauli_string
    n = len(pauli_string.keys())
    measure_layer = ops.Moment([
        ops.measure(*sorted(pauli_string.keys()), key='out')
    ])

    circuit = circuit.copy()
    fix_map = {
        ops.X: ops.Y**0.5,
        ops.Y: ops.X**-0.5,
    }
    transform_layer = ops.Moment([
        fix_map[p].on(q)
        for q, p in pauli_string.items()
        if p in fix_map
    ])
    if transform_layer:
        circuit.append(transform_layer)
    circuit.append(measure_layer)
    return circuit
