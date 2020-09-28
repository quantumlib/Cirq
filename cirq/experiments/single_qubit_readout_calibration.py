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

from typing import Any, Dict, Iterable, TYPE_CHECKING

import dataclasses
import time

import numpy as np

from cirq import circuits, ops

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass(frozen=True)
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
            'timestamp': self.timestamp
        }

    @classmethod
    def _from_json_dict_(cls, zero_state_errors, one_state_errors, repetitions,
                         timestamp, **kwargs):
        return cls(zero_state_errors=dict(zero_state_errors),
                   one_state_errors=dict(one_state_errors),
                   repetitions=repetitions,
                   timestamp=timestamp)

    def __repr__(self) -> str:
        return ('cirq.experiments.SingleQubitReadoutCalibrationResult('
                f'zero_state_errors={self.zero_state_errors!r}, '
                f'one_state_errors={self.one_state_errors!r}, '
                f'repetitions={self.repetitions!r}, '
                f'timestamp={self.timestamp!r})')


def estimate_single_qubit_readout_errors(
        sampler: 'cirq.Sampler',
        *,
        qubits: Iterable['cirq.Qid'],
        repetitions: int = 1000) -> SingleQubitReadoutCalibrationResult:
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
    qubits = list(qubits)

    zeros_circuit = circuits.Circuit(ops.measure_each(*qubits, key_func=repr))
    ones_circuit = circuits.Circuit(ops.X.on_each(*qubits),
                                    ops.measure_each(*qubits, key_func=repr))

    zeros_result = sampler.run(zeros_circuit, repetitions=repetitions)
    ones_result = sampler.run(ones_circuit, repetitions=repetitions)
    timestamp = time.time()

    zero_state_errors = {
        q: np.mean(zeros_result.measurements[repr(q)]) for q in qubits
    }
    one_state_errors = {
        q: 1 - np.mean(ones_result.measurements[repr(q)]) for q in qubits
    }

    return SingleQubitReadoutCalibrationResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp)
