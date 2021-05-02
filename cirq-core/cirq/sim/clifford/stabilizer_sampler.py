# Copyright 2020 The Cirq Developers
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

from typing import Dict, List

import numpy as np

import cirq
from cirq import circuits, protocols, value
from cirq.qis.clifford_tableau import CliffordTableau
from cirq.sim.clifford.act_on_clifford_tableau_args import ActOnCliffordTableauArgs
from cirq.work import sampler


class StabilizerSampler(sampler.Sampler):
    """An efficient sampler for stabilizer circuits."""

    def __init__(self, *, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        """
        Args:
            seed: The random seed or generator to use when sampling.
        """
        self.init = True
        self._prng = value.parse_random_state(seed)

    def run_sweep(
        self,
        program: 'cirq.Circuit',
        params: 'cirq.Sweepable',
        repetitions: int = 1,
    ) -> List['cirq.Result']:
        results: List[cirq.Result] = []
        for param_resolver in cirq.to_resolvers(params):
            resolved_circuit = cirq.resolve_parameters(program, param_resolver)
            measurements = self._run(
                resolved_circuit,
                repetitions=repetitions,
            )
            results.append(cirq.Result(params=param_resolver, measurements=measurements))
        return results

    def _run(self, circuit: circuits.Circuit, repetitions: int) -> Dict[str, np.ndarray]:

        measurements: Dict[str, List[int]] = {
            key: [] for key in protocols.measurement_keys(circuit)
        }
        axes_map = {q: i for i, q in enumerate(circuit.all_qubits())}

        for _ in range(repetitions):
            state = ActOnCliffordTableauArgs(
                CliffordTableau(num_qubits=len(axes_map)),
                axes=(),
                prng=self._prng,
                log_of_measurement_results={},
            )
            for op in circuit.all_operations():
                state.axes = tuple(axes_map[q] for q in op.qubits)
                protocols.act_on(op, state)

            for k, v in state.log_of_measurement_results.items():
                measurements[k].append(v)

        return {k: np.array(v) for k, v in measurements.items()}
