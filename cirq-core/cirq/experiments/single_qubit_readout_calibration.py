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

from typing import Iterable, TYPE_CHECKING

from cirq._compat import deprecated
from cirq.experiments.parallel_readout_calibration import estimate_parallel_readout_errors
from cirq.experiments.readout_experiment_result import ReadoutExperimentResult


if TYPE_CHECKING:
    import cirq


@deprecated(
    deadline="v0.13",
    fix="use cirq.experiments.ReadoutExperimentResult instead",
    name="cirq.experiments.SingleQubitReadoutCalibrationResult",
)
class SingleQubitReadoutCalibrationResult(ReadoutExperimentResult):
    """Result of estimating single qubit readout error.
    Deprecated: Use ReadoutExperimentResult instead.

    Attributes:
        zero_state_errors: A dictionary from qubit to probability of measuring
            a 1 when the qubit is initialized to |0⟩.
        one_state_errors: A dictionary from qubit to probability of measuring
            a 0 when the qubit is initialized to |1⟩.
        repetitions: The number of repetitions that were used to estimate the
            probabilities.
        timestamp: The time the data was taken, in seconds since the epoch.
    """


def estimate_single_qubit_readout_errors(
    sampler: 'cirq.Sampler', *, qubits: Iterable['cirq.Qid'], repetitions: int = 1000
) -> ReadoutExperimentResult:
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
        A ReadoutExperimentResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    return estimate_parallel_readout_errors(
        sampler=sampler,
        qubits=qubits,
        repetitions=repetitions,
        trials=2,
        bit_strings=[1 for q in qubits],
    )
