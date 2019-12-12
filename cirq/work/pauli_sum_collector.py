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
from typing import cast, Dict, Optional, Union, TYPE_CHECKING

import numpy as np

from cirq import ops
from cirq.work import collector

if TYPE_CHECKING:
    import cirq


class PauliSumCollector(collector.Collector):
    """Estimates the energy of a linear combination of Pauli observables."""

    def __init__(self,
                 circuit: 'cirq.Circuit',
                 observable: 'cirq.PauliSumLike',
                 *,
                 samples_per_term: int,
                 max_samples_per_job: int = 1000000):
        """
        Args:
            circuit: Produces the state to be tested.
            observable: The pauli product observables to measure. Their sampled
                expectations will be scaled by their coefficients and their
                dictionary weights, and then added up to produce the final
                result.
            samples_per_term: The number of samples to collect for each
                PauliString term in order to estimate its expectation.
            max_samples_per_job: How many samples to request at a time.
        """
        observable = ops.PauliSum.wrap(observable)

        self._circuit = circuit
        self._samples_per_job = max_samples_per_job
        self._pauli_coef_terms = [
            (p / p.coefficient, p.coefficient) for p in observable if p
        ]

        self._identity_offset = 0
        for p in observable:
            if not p:
                self._identity_offset += p.coefficient

        self._zeros: Dict[ops.PauliString, int] = collections.defaultdict(
            lambda: 0)
        self._ones: Dict[ops.PauliString, int] = collections.defaultdict(lambda:
                                                                         0)
        self._samples_per_term = samples_per_term
        self._total_samples_requested = 0

    def next_job(self) -> Optional['cirq.CircuitSampleJob']:
        i = self._total_samples_requested // self._samples_per_term
        if i >= len(self._pauli_coef_terms):
            return None
        pauli, _ = self._pauli_coef_terms[i]
        remaining = self._samples_per_term * (i +
                                              1) - self._total_samples_requested
        amount_to_request = min(remaining, self._samples_per_job)
        self._total_samples_requested += amount_to_request
        return collector.CircuitSampleJob(
            circuit=_circuit_plus_pauli_string_measurements(
                self._circuit, pauli),
            repetitions=amount_to_request,
            tag=pauli)

    def on_job_result(self, job: 'cirq.CircuitSampleJob',
                      result: 'cirq.TrialResult'):
        job_id = cast(ops.PauliString, job.tag)
        parities = result.histogram(key='out',
                                    fold_func=lambda bits: np.sum(bits) % 2)
        self._zeros[job_id] += parities[0]
        self._ones[job_id] += parities[1]

    def estimated_energy(self) -> Union[float, complex]:
        """Sums up the sampled expectations, weighted by their coefficients."""
        energy = 0j
        for pauli_string, coef in self._pauli_coef_terms:
            a = self._zeros[pauli_string]
            b = self._ones[pauli_string]
            if a + b:
                energy += coef * (a - b) / (a + b)
        energy = complex(energy)
        if energy.imag == 0:
            energy = energy.real
        energy += self._identity_offset
        return energy


def _circuit_plus_pauli_string_measurements(circuit: 'cirq.Circuit',
                                            pauli_string: 'cirq.PauliString'
                                           ) -> 'cirq.Circuit':
    """A circuit measuring the given observable at the end of the given circuit.
    """
    assert pauli_string
    circuit = circuit.copy()
    circuit.append(ops.Moment(pauli_string.to_z_basis_ops()))
    circuit.append(
        ops.Moment([ops.measure(*sorted(pauli_string.keys()), key='out')]))
    return circuit
