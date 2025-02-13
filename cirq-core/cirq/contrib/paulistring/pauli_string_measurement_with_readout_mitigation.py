# Copyright 2025 The Cirq Developers
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
""" Tools for measuring expectation values of Pauli strings with readout error mitigation. """

from typing import List, Tuple, Union

import numpy as np

from cirq import ops, circuits, work
from cirq.contrib.shuffle_circuits import run_shuffled_with_readout_benchmarking
from cirq.experiments import SingleQubitReadoutCalibrationResult
from cirq.experiments.readout_confusion_matrix import TensoredConfusionMatrices
from cirq.study import ResultDict


def _pauli_string_to_basis_change_ops(
    pauli_string: ops.PauliString, qid_list: list[ops.Qid]
) -> List[ops.Operation]:
    """Creates a moment to change to the eigenbasis of the given Pauli string.

    This function constructs a list of ops.Operation that performs basis changes
    necessary to measure the given pauli_string in the computational basis.

    Args:
        pauli_string: The Pauli string to diagonalize.
        qid_list: An ordered list of the qubits in the circuit.

    Returns:
        A list of Operations that, when applied before measurement in the
        computational basis, effectively measures in the eigenbasis of
        pauli_strings.
    """
    # ADD A TEST: GridQubit
    operations = []
    for qubit in pauli_string:
        pauli_op = pauli_string[qubit]
        # Use x,y properties of Qid instead of assuming LineQubit
        qubit_index = qid_list.index(qubit)
        if pauli_op == ops.X:
            # Rotate to X basis: Ry(-pi/2)
            operations.append(ops.ry(-np.pi / 2)(qid_list[qubit_index]))
        elif pauli_op == ops.Y:
            # Rotate to Y basis: Rx(pi/2)
            operations.append(ops.rx(np.pi / 2)(qid_list[qubit_index]))
        # No operation needed for Pauli Z or I (identity)
    return operations


def _build_one_qubit_confusion_matrix(e0: float, e1: float) -> np.ndarray:
    """Builds a 2x2 confusion matrix for a single qubit.

    Args:
        e0: the 0->1 readout error rate.
        e1: the 1->0 readout error rate.

    Returns:
        A 2x2 NumPy array representing the confusion matrix.
    """
    return np.array([[1 - e0, e1], [e0, 1 - e1]])


def _build_many_one_qubits_confusion_matrix(
    qubits_to_error: SingleQubitReadoutCalibrationResult,
) -> list[np.ndarray]:
    """Builds a list of confusion matrices from calibration results.

    This function iterates through the calibration results for each qubit and
    constructs a list of single-qubit confusion matrices.

    Args:
        qubits_to_error: An object containing calibration results for
            single-qubit readout errors, including zero-state and one-state errors
            for each qubit.

    Returns:
        A list of NumPy arrays, where each array is a 2x2 confusion matrix
        for a qubit. The order of matrices corresponds to the order of qubits
        in the calibration results (alphabetical order by qubit name).
        Returns an empty list if no calibration results are provided.
    """
    cms = []
    if not qubits_to_error:
        return cms

    for qubit in sorted(qubits_to_error.zero_state_errors.keys()):
        e0 = qubits_to_error.zero_state_errors[qubit]
        e1 = qubits_to_error.one_state_errors[qubit]
        cms.append(_build_one_qubit_confusion_matrix(e0, e1))
    return cms

def _process_pauli_measurement_results(
    qubits: List[ops.Qid],
    pauli_strings: List[ops.PauliString],
    circuit_results: List[ResultDict],
    confusion_matrices: List[np.ndarray],
    pauli_repetitions: int,
    timestamp: float,
) -> Tuple[List[Tuple[ops.PauliString, float, float]], List[Tuple[ops.PauliString, float, float]]]:
    """Calculates both error-mitigated expectation values and unmitigated expectation values 
    from measurement results.

    This function takes the results from shuffled readout benchmarking and:
    1. Builds confusion matrices from the calibration data.
    2. Constructs a tensored confusion matrix for error mitigation, and one without.
    3. Mitigates readout errors for each Pauli string measurement.
    4. Calculates and returns both error-mitigated and unmitigated expectation values.

    Args:
        qubits: Qubits to build confusion matrices for. In a sorted order.
        pauli_strings: The list of PauliStrings that were measured.
        circuit_results: A list of ResultDict obtained
            from running the Pauli measurement circuits.
        confusion_matrices: A list of confusion matrices from calibration results. 
        pauli_repetitions: The number of repetitions used for Pauli string measurements.
        timestamp: The timestamp of the calibration results.

    Returns:
        A tuple containing:
            - A list of tuples, where each tuple contains:
                - The PauliString that was measured.
                - The error-mitigated expectation value (float) of the Pauli string.
                - The standard deviation of the error-mitigated expectation value.
            - A list of tuples, where each tuple contains:
                - The PauliString that was measured.
                - The unmitigated expectation value (float) of the Pauli string.
                - The standard deviation of the unmitigated expectation value.
    """

    tensored_cm = TensoredConfusionMatrices(
        confusion_matrices,
        [[q] for q in qubits],
        repetitions=pauli_repetitions,
        timestamp=timestamp,
    )

    exp_vals_with_mitigation = []
    exp_vals_without_mitigation = []

    for pauli_index, circuit_result in enumerate(circuit_results):
        measurement_results = circuit_result.measurements["m"]

        z_mitigated_values, d_z = tensored_cm.readout_mitigation_pauli_uncorrelated(
            qubits, measurement_results
        )

        p1 = np.mean(np.sum(measurement_results, axis=1) % 2)
        z_unmitigated_values = 1 - 2 * np.mean(p1)
        dz_unmit = 2 * np.sqrt(p1 * (1 - p1) / pauli_repetitions)

        pauli_string = pauli_strings[pauli_index]
        exp_vals_with_mitigation.append((pauli_string, z_mitigated_values, d_z))
        exp_vals_without_mitigation.append((pauli_string, z_unmitigated_values, dz_unmit))

    return exp_vals_with_mitigation, exp_vals_without_mitigation

def measure_pauli_strings(circuits_to_pauli: List[Tuple[circuits.Circuit, list[ops.PauliString]]],
                          sampler: work.Sampler, rng_or_seed: Union[np.random.Generator, int],
                          pauli_repetitions: int, readout_repetitions: int,
                          num_random_bitstrings: int
                          ) -> List[Tuple[circuits.Circuit,Tuple[
                              List[Tuple[ops.PauliString, float, float]],
                              List[Tuple[ops.PauliString, float, float]]
          ]],
]:
    """Measures expectation values of Pauli strings on given circuits with/without 
    readout error mitigation.

    This function takes a list of circuits and corresponding List[Pauli string] to measure.
    For each circuit-List[Pauli string] pair, it:
    1.  Constructs circuits to measure the Pauli string expectation value by
        adding basis change moments and measurement operations.
    2.  Runs shuffled readout benchmarking on these circuits to calibrate readout errors.
    3.  Mitigates readout errors using the calibrated confusion matrices.
    4.  Calculates and returns both error-mitigated and unmitigatedexpectation values for
    each Pauli string.

    Args:
        circuits_to_pauli: A list of tuples, where each tuple contains:
            - A Circuit to prepare the state for measurement.
            - A list of PauliString objects to measure on the state
              prepared by the circuit.
        sampler: The sampler to use.
        rng_or_seed: A random number generator or seed for the shuffled benchmarking.
        pauli_repetitions: The number of repetitions for each circuit when measuring
            Pauli strings.
        readout_repetitions: The number of repetitions for readout calibration
            in the shuffled benchmarking.
        num_random_bitstrings: The number of random bitstrings to use in shuffled
            benchmarking.

    Returns:
        A list of tuples, where each tuple corresponds to an input circuit and contains:
            - The original input Circuit.
            - A tuple containing:
                - A list of tuples, where each tuple contains:
                    - The `cirq.PauliString` that was measured.
                    - The error-mitigated expectation value (float) of the Pauli string.
                    - The standard deviation of the expectation value.
                - A list of tuples, where each tuple contains:
                    - The `cirq.PauliString` that was measured.
                    - The unmitigated expectation value (float) of the Pauli string.
                    - The standard deviation of the expectation value.
    """

    # Extract unique qubit tuples from all circuits
    input_circuits = [circuit for circuit, _ in circuits_to_pauli]

    unique_qubit_tuples = set()
    for circuit in input_circuits:
        unique_qubit_tuples.add(tuple(sorted(set(circuit.all_qubits()))))
    qubits_list = sorted(unique_qubit_tuples)

    pauli_measurement_circuits: List[circuits.Circuit] = []

    # Build the basis-change circuits for each Pauli string
    pauli_measurement_circuits = []
    for input_circuit, pauli_strings in circuits_to_pauli:
        qid_list = list(set(input_circuit.all_qubits()))
        basis_change_circuits = []
        for pauli_string in pauli_strings:
            basis_change_circuit = input_circuit + _pauli_string_to_basis_change_ops(
                pauli_string, qid_list) + ops.measure(*qid_list, key="m")
            basis_change_circuits.append(basis_change_circuit)
        pauli_measurement_circuits.extend(basis_change_circuits)

    # Run shuffled benchmarking for readout calibration
    circuits_results, calibration_results = run_shuffled_with_readout_benchmarking(
        input_circuits=pauli_measurement_circuits,
        sampler=sampler,
        circuit_repetitions=pauli_repetitions,
        rng_or_seed=rng_or_seed,
        qubits=qubits_list,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    results = []
    circuit_result_index = 0
    for input_circuit, pauli_strings in circuits_to_pauli:
        qubits_in_circuit = tuple(sorted(set(input_circuit.all_qubits())))

        confusion_matrices = _build_many_one_qubits_confusion_matrix(
            calibration_results[qubits_in_circuit])
        mitigated_res, unmitigated_res = _process_pauli_measurement_results(
            qubits_in_circuit, pauli_strings,
            circuits_results[circuit_result_index: circuit_result_index + len(pauli_strings)],
            confusion_matrices, pauli_repetitions,
            calibration_results[qubits_in_circuit].timestamp)
        results.append((input_circuit, (mitigated_res, unmitigated_res)))

        circuit_result_index += len(pauli_strings)
    return results
