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
"""Tools for running circuits in a shuffled order with readout error benchmarking."""

from typing import Iterable, Optional

import numpy as np

from cirq import ops, circuits, work

def run_shuffled_with_readout_benchmarking(
    input_circuits: list[circuits.Circuit],
    sampler: work.Sampler,
    circuit_repetitions: int | list[int],
    num_random_bitstrings: int = 100,
    readout_repetitions: int = 1000,
    ) -> tuple[np.ndarray, dict["cirq.Qid": tuple[float, float]]]:
    # tuple[np.ndarray, dict[cirq.Qid: tuple[float, float]]]:

    """Run the circuits in a shuffled order with readout error benchmarking.
    
    Args:
        circuits: The circuits to run.
        sampler: The sampler to use.
        circuit_repetitions: The repetitions for `circuits`.
        num_random_bitstrings: The number of random bitstrings for measuring readout.
        readout_repetitions: The number of repetitions for each readout bitstring.

    Returns:
        The unshuffled measurements and a dictionary from qubits to the corresponding readout error rates (e0 and e1, where e0 is the 0->1 readout error rate and e1 is the 1->0 readout error rate).

    """

    # Check input_circuits is not empty
    if not input_circuits:
        raise ValueError("Input circuits must not be empty.")
    # Check input_circuits type is cirq.circuits
    if not all(isinstance(circuit, circuits.Circuit) for circuit in input_circuits):
        raise ValueError("Input circuits must be of type cirq.Circuit." )
    # Check input_circuits have measurements
    for circuit in input_circuits:
        if not any(isinstance(op.gate, ops.MeasurementGate) for op in circuit.all_operations()):
            raise ValueError("Input circuits must have measurements.")
    
    # Check circuit_repetitions
    if type(circuit_repetitions) == int:
        if (circuit_repetitions <= 0):
            raise ValueError("Must provide non-zero circuit_repetitions.")
        circuit_repetitions = [circuit_repetitions] * len(input_circuits) 
    if len(circuit_repetitions) != len(input_circuits):
        raise ValueError("Number of circuit_repetitions must match the number of input circuits.")
    
    # Check num_random_bitstrings
    if num_random_bitstrings <= 0:
        raise ValueError("Must provide non-zero num_random_bitstrings.")
       
    # Check readout_repetitions is bigger than 0
    if readout_repetitions <= 0:
        raise ValueError("Must provide non-zero readout_repetitions for readout calibration.")
      
    # Extract qubits from input circuits
    qubits = set()
    for circuit in input_circuits:
        qubits.update(circuit.all_qubits())
    qubits = sorted(qubits)

    # Generate the readout calibration circuits
    rng = np.random.default_rng()
    x_or_I = lambda bit: ops.X if bit == 1 else ops.I

    random_bitstrings = np.random.randint(0, 2, size=(num_random_bitstrings, len(qubits)), dtype=np.int32)
    
    readout_calibration_circuits = []
    for bitstr in random_bitstrings:
        readout_calibration_circuits.append(
            circuits.Circuit(
                [x_or_I(bit)(qubit) for bit, qubit in zip(bitstr, qubits)]
                + [ops.M(qubits, key="m")]
            )
        )

    # Shuffle the circuits
    all_circuits = input_circuits + readout_calibration_circuits
    all_repetitions = circuit_repetitions + [readout_repetitions] * len(readout_calibration_circuits)

    shuf_order = np.arange(len(all_circuits))
    rng.shuffle(shuf_order)
    unshuf_order = np.zeros_like(shuf_order)
    unshuf_order[shuf_order] = np.arange(len(all_circuits))
    shuffled_circuits = [all_circuits[_] for _ in shuf_order]
    all_repetitions = [all_repetitions[_] for _ in shuf_order]

    # Run the shuffled circuits and measure
    results = sampler.run_batch(shuffled_circuits, repetitions=all_repetitions)
    shuffled_measurements = [res[0].measurements["m"] for res in results]
    unshuffled_measurements = [shuffled_measurements[_] for _ in unshuf_order]

    unshuffled_readout_measurements = unshuffled_measurements[len(input_circuits):]

    # Analyze results
    zero_state_trials = np.zeros((1, len(qubits)), dtype=np.int64)
    one_state_trials = np.zeros((1, len(qubits)), dtype=np.int64)
    zero_state_totals = np.zeros((1, len(qubits)), dtype=np.int64)
    one_state_totals = np.zeros((1, len(qubits)), dtype=np.int64)
    trial_idx = 0
    for trial_result in unshuffled_readout_measurements:
        trial_result = trial_result.astype(np.int64)  # Cast to int64
        sample_counts = np.einsum('ij->j', trial_result)
        
        zero_state_trials += sample_counts * (1 - random_bitstrings[trial_idx])
        zero_state_totals += readout_repetitions * (1 - random_bitstrings[trial_idx])
        one_state_trials += (readout_repetitions - sample_counts) * random_bitstrings[trial_idx]
        one_state_totals += readout_repetitions * random_bitstrings[trial_idx]
        
        trial_idx += 1

    readout_error_rates = {
    q: (
        zero_state_trials[0][qubit_idx] / zero_state_totals[0][qubit_idx] if zero_state_totals[0][qubit_idx] > 0
            else np.nan, one_state_trials[0][qubit_idx] / one_state_totals[0][qubit_idx]
            if one_state_totals[0][qubit_idx] > 0
            else np.nan
        )
    for qubit_idx, q in enumerate(qubits)
    }

    return unshuffled_measurements, readout_error_rates