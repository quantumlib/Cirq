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
"""Tools for measuring expectation values of Pauli strings with readout error mitigation. """
import time
from typing import List, Union, Dict, Optional, Tuple
import attrs

import numpy as np

from cirq import ops, circuits, work
from cirq.contrib.shuffle_circuits import run_shuffled_with_readout_benchmarking
from cirq.experiments import SingleQubitReadoutCalibrationResult
from cirq.experiments.readout_confusion_matrix import TensoredConfusionMatrices
from cirq.study import ResultDict


@attrs.define
class PauliStringMeasurementResult:
    """Result of measuring a Pauli string.

    Attributes:
        pauli_string: The Pauli string that is measured.
        mitigated_expectation: The error-mitigated expectation value of the Pauli string.
        mitigated_stddev: The standard deviation of the error-mitigated expectation value.
        unmitigated_expectation: The unmitigated expectation value of the Pauli string.
        unmitigated_stddev: The standard deviation of the unmitigated expectation value.
        calibration_result: The calibration result for single-qubit readout errors.
    """

    pauli_string: ops.PauliString
    mitigated_expectation: float
    mitigated_stddev: float
    unmitigated_expectation: float
    unmitigated_stddev: float
    calibration_result: Optional[SingleQubitReadoutCalibrationResult]


@attrs.define
class CircuitToPauliStringsMeasurementResult:
    """Result of measuring Pauli strings on a circuit.

    Attributes:
        circuit: The circuit that is measured.
        results: A list of PauliStringMeasurementResult objects.
    """
    
    circuit: circuits.FrozenCircuit
    results: List[PauliStringMeasurementResult]


def _pauli_string_to_basis_change_ops(
    pauli_string: ops.PauliString, qid_list: list[ops.Qid]
) -> List[ops.Operation]:
    """Creates operations to change to the eigenbasis of the given Pauli string.

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
    operations = []
    for qubit in qid_list:  # Iterate over ALL qubits in the circuit
        if qubit in pauli_string:
            pauli_op = pauli_string[qubit]
            if pauli_op == ops.X:
                operations.append(ops.ry(-np.pi / 2)(qubit))
            elif pauli_op == ops.Y:
                operations.append(ops.rx(np.pi / 2)(qubit))
            # If pauli_op is Z or I, no operation needed
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
    """
    cms: list[np.ndarray] = []
    if not qubits_to_error:
        return cms

    for qubit in sorted(qubits_to_error.zero_state_errors.keys()):
        e0 = qubits_to_error.zero_state_errors[qubit]
        e1 = qubits_to_error.one_state_errors[qubit]
        cms.append(_build_one_qubit_confusion_matrix(e0, e1))
    return cms


def _build_many_one_qubits_empty_confusion_matrix(qubits_length: int):
    """Builds a list of empty confusion matrices"""
    cms: list[np.ndarray] = []
    for _ in range(qubits_length):
        cms.append(_build_one_qubit_confusion_matrix(0, 0))
    return cms


def _process_pauli_measurement_results(
    qubits: List[ops.Qid],
    pauli_strings: List[ops.PauliString],
    circuit_results: List[ResultDict],
    calibration_results: Dict[Tuple[ops.Qid, ...], SingleQubitReadoutCalibrationResult],
    pauli_repetitions: int,
    timestamp: float,
    disable_readout_mitigation: bool = False,
) -> List[PauliStringMeasurementResult]:
    """Calculates both error-mitigated expectation values and unmitigated expectation values
    from measurement results.

    This function takes the results from shuffled readout benchmarking and:
    1. Constructs a tensored confusion matrix for error mitigation.
    2. Mitigates readout errors for each Pauli string measurement.
    3. Calculates and returns both error-mitigated and unmitigated expectation values.

    Args:
        qubits: Qubits to build confusion matrices for. In a sorted order.
        pauli_strings: The list of PauliStrings that are measured.
        circuit_results: A list of ResultDict obtained
            from running the Pauli measurement circuits.
        confusion_matrices: A list of confusion matrices from calibration results.
        pauli_repetitions: The number of repetitions used for Pauli string measurements.
        timestamp: The timestamp of the calibration results.
        disable_readout_mitigation: If set to True, returns no error-mitigated error
            expectation values.

    Returns:
        A list of PauliStringMeasurementResult objects, where each object contains:
            - The Pauli string that was measured.
            - The mitigated expectation value of the Pauli string.
            - The standard deviation of the mitigated expectation value.
            - The unmitigated expectation value of the Pauli string.
            - The standard deviation of the unmitigated expectation value.
    """

    pauli_measurement_results: List[PauliStringMeasurementResult] = []

    for pauli_index, circuit_result in enumerate(circuit_results):
        measurement_results = circuit_result.measurements["m"]

        pauli_string = pauli_strings[pauli_index]
        qubits_sorted = sorted(pauli_string.qubits)
        qubit_indices = [qubits.index(q) for q in qubits_sorted]

        confusion_matrices = (
            _build_many_one_qubits_confusion_matrix(calibration_results[tuple(qubits_sorted)])
            if disable_readout_mitigation is False
            else _build_many_one_qubits_empty_confusion_matrix(len(qubits_sorted))
        )
        tensored_cm = TensoredConfusionMatrices(
            confusion_matrices,
            [[q] for q in qubits_sorted],
            repetitions=pauli_repetitions,
            timestamp=timestamp,
        )

        #  Create a mask for the relevant qubits in the measurement results
        relevant_bits = measurement_results[:, qubit_indices]

        # Calculate the mitigated expectation.
        mitigated_values, d_m = tensored_cm.readout_mitigation_pauli_uncorrelated(
            qubits_sorted, relevant_bits
        )

        # Calculate the unmitigated expectation.
        parity = np.sum(relevant_bits, axis=1) % 2
        unmitigated_values = 1 - 2 * np.mean(parity)
        d_unmit = 2 * np.sqrt(np.mean(parity) * (1 - np.mean(parity)) / pauli_repetitions)

        pauli_measurement_results.append(
            PauliStringMeasurementResult(
                pauli_string=pauli_strings[pauli_index],
                mitigated_expectation=mitigated_values,
                mitigated_stddev=d_m,
                unmitigated_expectation=unmitigated_values,
                unmitigated_stddev=d_unmit,
                calibration_result=(
                    calibration_results[tuple(qubits_sorted)]
                    if disable_readout_mitigation is False
                    else None
                ),
            )
        )

    return pauli_measurement_results


def measure_pauli_strings(
    circuits_to_pauli: Dict[circuits.FrozenCircuit, list[ops.PauliString]],
    sampler: work.Sampler,
    rng_or_seed: Union[np.random.Generator, int],
    pauli_repetitions: int,
    readout_repetitions: int,
    num_random_bitstrings: int,
) -> List[CircuitToPauliStringsMeasurementResult]:
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
        circuits_to_pauli: A dictionary mapping circuits to a list of Pauli strings
            to measure.
        sampler: The sampler to use.
        rng_or_seed: A random number generator or seed for the shuffled benchmarking.
        pauli_repetitions: The number of repetitions for each circuit when measuring
            Pauli strings.
        readout_repetitions: The number of repetitions for readout calibration
            in the shuffled benchmarking.
        num_random_bitstrings: The number of random bitstrings to use in shuffled
            benchmarking.

    Returns:
        A list of CircuitToPauliStringsMeasurementResult objects, where each object contains:
            - The circuit that was measured.
            - A list of PauliStringMeasurementResult objects.
            - The calibration result for single-qubit readout errors.
    """

    # Extract unique qubit tuples from input pauli strings
    unique_qubit_tuples = set()
    for pauli_strings in circuits_to_pauli.values():
        for pauli_string in pauli_strings:
            unique_qubit_tuples.add(tuple(sorted(set(pauli_string.qubits))))
    # qubits_list is a list of qubit tuples
    qubits_list = sorted(unique_qubit_tuples)

    # Build the basis-change circuits for each Pauli string
    pauli_measurement_circuits = list[ops.PauliString]()
    for input_circuit, pauli_strings in circuits_to_pauli.items():
        qid_list = list(sorted(input_circuit.all_qubits()))
        basis_change_circuits = []
        input_circuit_unfrozen = input_circuit.unfreeze()
        for pauli_string in pauli_strings:
            basis_change_circuit = (
                input_circuit_unfrozen
                + _pauli_string_to_basis_change_ops(pauli_string, qid_list)
                + ops.measure(*qid_list, key="m")
            )
            basis_change_circuits.append(basis_change_circuit)
        pauli_measurement_circuits.extend(basis_change_circuits)

    # Run shuffled benchmarking for readout calibration
    circuits_results, calibration_results = run_shuffled_with_readout_benchmarking(
        input_circuits=pauli_measurement_circuits,
        sampler=sampler,
        circuit_repetitions=pauli_repetitions,
        rng_or_seed=rng_or_seed,
        qubits=[list(qubits) for qubits in qubits_list],
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    # Process the results to calculate expectation values
    results: List[CircuitToPauliStringsMeasurementResult] = []
    circuit_result_index = 0
    for input_circuit, pauli_strings in circuits_to_pauli.items():
        qubits_in_circuit = tuple(sorted(input_circuit.all_qubits()))

        disable_readout_mitigation = False if num_random_bitstrings != 0 else True
        pauli_measurement_results = _process_pauli_measurement_results(
            list(qubits_in_circuit),
            pauli_strings,
            circuits_results[circuit_result_index : circuit_result_index + len(pauli_strings)],
            calibration_results,
            pauli_repetitions,
            time.time(),
            disable_readout_mitigation,
        )
        results.append(
            CircuitToPauliStringsMeasurementResult(
                circuit=input_circuit, results=pauli_measurement_results
            )
        )

        circuit_result_index += len(pauli_strings)
    return results
