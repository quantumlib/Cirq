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
"""TODO: Add module docstring."""

from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np

from cirq.circuits import Circuit

def run_shuffled_with_readout_benchmarking(
    circuits: list[Circuit],
    # sampler: cirq.Sampler,
    # circuit_repetitions: int | list[int],
    # num_random_bitstrings: int,
    # readout_repetitions: int
    ) -> Iterable['cirq.Qid']:
    # tuple[np.ndarray, dict[cirq.Qid: tuple[float, float]]]:

    """Run the circuits in a shuffled order with readout error benchmarking."""

    # Extract qubits from input circuits
    qubits = set()
    for circuit in circuits:
        qubits.update(circuit.all_qubits())
    qubits = sorted(qubits)

    # Generate the readout calibration circuits

    # Shuffle the circuits

    # Run the shuffled circuits 

    # Unshuffled measurements 

    return qubits

    

             