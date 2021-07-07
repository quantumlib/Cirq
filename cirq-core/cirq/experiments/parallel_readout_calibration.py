# Copyright 2021 The Cirq Developers
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

from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

import dataclasses
import time
import random

import numpy as np
import sympy

from cirq import circuits, ops, study

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass(frozen=True)
class ParallelReadoutResult:
    """Result of estimating single qubit readout error.

    Attributes:
        zero_state_errors: A dictionary from qubit to probability of measuring
            a 1 when the qubit is initialized to |0⟩.
        one_state_errors: A dictionary from qubit to probability of measuring
            a 0 when the qubit is initialized to |1⟩.
        repetitions: The number of repetitions that were used to estimate the
            probabilities.
        timestamp: The time the data was taken, in seconds since the epoch.
    """

    zero_state_errors: Dict['cirq.Qid', float]
    one_state_errors: Dict['cirq.Qid', float]
    repetitions: int
    timestamp: float

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'zero_state_errors': list(self.zero_state_errors.items()),
            'one_state_errors': list(self.one_state_errors.items()),
            'repetitions': self.repetitions,
            'timestamp': self.timestamp,
        }

    @classmethod
    def _from_json_dict_(
        cls, zero_state_errors, one_state_errors, repetitions, timestamp, **kwargs
    ):
        return cls(
            zero_state_errors=dict(zero_state_errors),
            one_state_errors=dict(one_state_errors),
            repetitions=repetitions,
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            'cirq.experiments.ParallelReadoutResult('
            f'zero_state_errors={self.zero_state_errors!r}, '
            f'one_state_errors={self.one_state_errors!r}, '
            f'repetitions={self.repetitions!r}, '
            f'timestamp={self.timestamp!r})'
        )


def estimate_parallel_readout_errors(
    sampler: 'cirq.Sampler',
    *,
    qubits: Iterable['cirq.Qid'],
    trials: int = 20,
    repetitions: int = 1000,
    trials_per_batch: Optional[int] = None,
) -> ParallelReadoutResult:

    """Estimate single-qubit readout error.

    For each trial, prepare a bitstring of random |0> and |1> states for
    each state.  Measure each qubit.  Capture the errors per qubit of
    zero and one state over each triel.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform for
            each trial.
        trials: The number of bitstrings to prepare.
        trials_per_batch:  If provided, split the experiment into batches
            with this number of trials in each batch.

    Returns:
        A ParallelReadoutResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    qubits = list(qubits)
    num_qubits = len(qubits)

    sweeps = {}

    trial_bits = [random.getrandbits(trials) for _ in qubits]
    all_circuits = []
    all_sweeps = []
    if trials_per_batch is not None:
        num_batchs = trials // trials_per_batch
        if trials % trials_per_batch > 1:
            num_batchs += 1
    else:
        num_batchs = 1
        trials_per_batch = trials

    for batch in range(num_batchs):
        circuit = circuits.Circuit()
        single_sweeps = []
        for idx, q in enumerate(qubits):
            sym_val = f'bit_{idx}'
            current_trial = trial_bits[idx]
            trial_range = range(batch * trials_per_batch, (batch + 1) * trials_per_batch)
            circuit.append(ops.X(q) ** sympy.Symbol(sym_val))
            single_sweeps.append(
                study.Points(
                    key=sym_val, points=[(current_trial >> bit) & 1 for bit in trial_range]
                )
            )

        circuit.append(ops.measure_each(*qubits, key_func=repr))
        total_sweeps = study.Zip(*single_sweeps)
        all_circuits.append(circuit)
        all_sweeps.append(total_sweeps)

    results = sampler.run_batch(all_circuits, all_sweeps, repetitions=repetitions)
    timestamp = time.time()

    zero_state_trials = {q: [] for q in qubits}
    one_state_trials = {q: [] for q in qubits}
    for batch_idx, batch_result in enumerate(results):
        for trial_idx, trial_result in enumerate(batch_result):
            for idx, q in enumerate(qubits):
                had_x_gate = (trial_bits[idx] >> trial_idx) & 1
                if had_x_gate:
                    one_state_trials[q].append(1 - np.mean(trial_result.measurements[repr(q)]))
                else:
                    zero_state_trials[q].append(np.mean(trial_result.measurements[repr(q)]))

    zero_state_errors = {q: np.mean(zero_state_trials[q]) for q in qubits}
    one_state_errors = {q: np.mean(one_state_trials[q]) for q in qubits}

    return ParallelReadoutResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp,
    )
