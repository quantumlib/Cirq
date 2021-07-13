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

from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import time
import random

import numpy as np
import sympy

from cirq import circuits, ops, study
from cirq.experiments.readout_experiment_result import ReadoutExperimentResult

if TYPE_CHECKING:
    import cirq


def estimate_parallel_readout_errors(
    sampler: 'cirq.Sampler',
    *,
    qubits: Iterable['cirq.Qid'],
    trials: int = 20,
    repetitions: int = 1000,
    trials_per_batch: Optional[int] = None,
    bit_strings: Optional[List[int]] = None,
) -> ReadoutExperimentResult:

    """Estimate readout error for qubits simultaneously.

    For each trial, prepare a bitstring of random |0〉and |1〉states for
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
        bit_strings: A list of ints that specifies the bit strings for each
            qubit.  There should be exactly one int per qubit.  Bits
            represent whether the qubit is |0〉or |1〉when initialized.
            Trials are done from least significant bit to most significant bit.
            If not provided, the function will generate random bit strings
            for you.

    Returns:
        A ReadoutExperimentResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.  Note that,
        if there did not exist a trial where a given qubit was set to |0〉,
        the zero-state error will be set to `nan` (not a number).  Likewise
        for qubits with no |1〉trial and one-state error.
    """
    qubits = list(qubits)

    if trials <= 0:
        raise ValueError("Must provide non-zero trials for readout calibration.")
    if repetitions <= 0:
        raise ValueError("Must provide non-zero repetition for readout calibration.")
    if bit_strings is None:
        bit_strings = [random.getrandbits(trials) for _ in qubits]
    if len(bit_strings) != len(qubits):
        raise ValueError(
            f'If providing bit_strings, # of bit strings ({len(bit_strings)}) '
            f'must equal # of qubits ({len(qubits)})'
        )

    all_circuits = []
    all_sweeps: List[study.Sweepable] = []
    if trials_per_batch is not None:
        if trials_per_batch <= 0:
            raise ValueError("Must provide non-zero trials_per_batch for readout calibration.")
        num_batchs = trials // trials_per_batch
        if trials % trials_per_batch > 0:
            num_batchs += 1
    else:
        num_batchs = 1
        trials_per_batch = trials
    for batch in range(num_batchs):
        circuit = circuits.Circuit()
        single_sweeps = []
        for idx, q in enumerate(qubits):
            sym_val = f'bit_{idx}'
            current_trial = bit_strings[idx]
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

    zero_state_trials: Dict[cirq.Qid, List[float]] = {q: [] for q in qubits}
    one_state_trials: Dict[cirq.Qid, List[float]] = {q: [] for q in qubits}
    for batch_result in results:
        for trial_idx, trial_result in enumerate(batch_result):
            for idx, q in enumerate(qubits):
                had_x_gate = (bit_strings[idx] >> trial_idx) & 1
                if had_x_gate:
                    one_state_trials[q].append(1 - np.mean(trial_result.measurements[repr(q)]))
                else:
                    zero_state_trials[q].append(np.mean(trial_result.measurements[repr(q)]))

    zero_state_errors = {q: np.mean(zero_state_trials[q]) for q in qubits}
    one_state_errors = {q: np.mean(one_state_trials[q]) for q in qubits}

    return ReadoutExperimentResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp,
    )
