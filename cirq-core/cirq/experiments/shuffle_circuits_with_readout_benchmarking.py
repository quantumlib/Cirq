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

from cirq import ops, study
from cirq.circuits import Circuit

def run_shuffled_with_readout_benchmarking(
    circuits: list[Circuit],
    # sampler: cirq.Sampler,
    # circuit_repetitions: int | list[int],
    num_random_bitstrings: int,
    # readout_repetitions: int
    ) -> list[Circuit]:
    # tuple[np.ndarray, dict[cirq.Qid: tuple[float, float]]]:

    """Run the circuits in a shuffled order with readout error benchmarking."""

    # Extract qubits from input circuits
    qubits = set()
    for circuit in circuits:
        qubits.update(circuit.all_qubits())
    qubits = sorted(qubits)

    # Generate the readout calibration circuits
    rng = np.random.default_rng()
    x_or_I = lambda bit: ops.X if bit == 1 else ops.I

    random_bitstrings = [rng.integers(0, 2, size=(1, len(qubits)))
                for n in range(1, num_random_bitstrings+1)]
    
    readout_calibration_circuits = []
    for bitstrs_n in random_bitstrings:
        readout_calibration_circuits += [
            Circuit(
                [x_or_I(bit)(qubit) for bit, qubit in zip(bitstr, qubits)]
                + [ops.M(qubits, key="m")]
            )
            for bitstr in bitstrs_n
    ]

    # Shuffle the circuits

    # Run the shuffled circuits 

    # Unshuffled measurements 

    return readout_calibration_circuits

    

             