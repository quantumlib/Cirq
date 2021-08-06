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
"""Module for supporting single qubit readout experiments using
either correlated or uncorrelated readout statistics.
"""
import dataclasses
import random
import time
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import sympy
import numpy as np
from cirq import circuits, ops, study

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class SingleQubitReadoutCalibrationResult:
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
            'cirq.experiments.SingleQubitReadoutCalibrationResult('
            f'zero_state_errors={self.zero_state_errors!r}, '
            f'one_state_errors={self.one_state_errors!r}, '
            f'repetitions={self.repetitions!r}, '
            f'timestamp={self.timestamp!r})'
        )


def estimate_single_qubit_readout_errors(
    sampler: 'cirq.Sampler', *, qubits: Iterable['cirq.Qid'], repetitions: int = 1000
) -> SingleQubitReadoutCalibrationResult:
    """Estimate single-qubit readout error.

    For each qubit, prepare the |0⟩ state and measure. Calculate how often a 1
    is measured. Also, prepare the |1⟩ state and calculate how often a 0 is
    measured. The state preparations and measurements are done in parallel,
    i.e., for the first experiment, we actually prepare every qubit in the |0⟩
    state and measure them simultaneously.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    return estimate_correlated_single_qubit_readout_errors(
        sampler=sampler,
        qubits=qubits,
        repetitions=repetitions,
        trials=2,
        bit_strings=[1 for q in qubits],
    )


def estimate_correlated_single_qubit_readout_errors(
    sampler: 'cirq.Sampler',
    *,
    qubits: Iterable['cirq.Qid'],
    trials: int = 20,
    repetitions: int = 1000,
    trials_per_batch: Optional[int] = None,
    bit_strings: Optional[List[int]] = None,
) -> SingleQubitReadoutCalibrationResult:
    """Estimate single qubit readout error using parallel operations.

    For each trial, prepare and then measure a random computational basis
    bitstring on qubits using gates in parallel.
    Returns a SingleQubitReadoutCalibrationResult which can be used to
    compute readout errors for each qubit.

    Args:
        sampler: The `cirq.Sampler` used to run the circuits.
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
        A SingleQubitReadoutCalibrationResult storing the readout error
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
    if trials_per_batch is None:
        trials_per_batch = trials
    if trials_per_batch <= 0:
        raise ValueError("Must provide non-zero trials_per_batch for readout calibration.")

    num_batches = (trials + trials_per_batch - 1) // trials_per_batch
    all_circuits = []
    all_sweeps: List[study.Sweepable] = []

    for batch in range(num_batches):
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

    zero_state_trials: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    one_state_trials: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    zero_state_totals: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    one_state_totals: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    for batch_result in results:
        for trial_idx, trial_result in enumerate(batch_result):
            for idx, q in enumerate(qubits):
                had_x_gate = (bit_strings[idx] >> trial_idx) & 1
                if had_x_gate:
                    one_state_trials[q] += repetitions - np.sum(trial_result.measurements[repr(q)])
                    one_state_totals[q] += repetitions
                else:
                    zero_state_trials[q] += np.sum(trial_result.measurements[repr(q)])
                    zero_state_totals[q] += repetitions

    zero_state_errors = {
        q: zero_state_trials[q] / zero_state_totals[q] if zero_state_totals[q] > 0 else np.nan
        for q in qubits
    }
    one_state_errors = {
        q: one_state_trials[q] / one_state_totals[q] if one_state_totals[q] > 0 else np.nan
        for q in qubits
    }

    return SingleQubitReadoutCalibrationResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp,
    )
