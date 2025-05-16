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

"""Tools for measuring expectation values of Pauli strings with readout error mitigation."""

from __future__ import annotations

import itertools
import time
from typing import cast, Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np

from cirq import circuits, ops, work
from cirq.contrib.shuffle_circuits import run_shuffled_with_readout_benchmarking
from cirq.experiments.readout_confusion_matrix import TensoredConfusionMatrices

if TYPE_CHECKING:
    from cirq.experiments import SingleQubitReadoutCalibrationResult
    from cirq.study import ResultDict


@attrs.frozen
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
    calibration_result: Optional[SingleQubitReadoutCalibrationResult] = None


@attrs.frozen
class CircuitToPauliStringsMeasurementResult:
    """Result of measuring Pauli strings on a circuit.

    Attributes:
        circuit: The circuit that is measured.
        results: A list of PauliStringMeasurementResult objects.
    """

    circuit: circuits.FrozenCircuit
    results: List[PauliStringMeasurementResult]


def _commute_or_identity(
    op1: Union[ops.Pauli, ops.IdentityGate], op2: Union[ops.Pauli, ops.IdentityGate]
) -> bool:
    if op1 == ops.I or op2 == ops.I:
        return True
    return op1 == op2


def _are_two_pauli_strings_qubit_wise_commuting(
    pauli_str1: ops.PauliString,
    pauli_str2: ops.PauliString,
    all_qubits: Union[list[ops.Qid], FrozenSet[ops.Qid]],
) -> bool:
    for qubit in all_qubits:
        op1 = pauli_str1.get(qubit, default=ops.I)
        op2 = pauli_str2.get(qubit, default=ops.I)

        if not _commute_or_identity(op1, op2):
            return False
    return True


def _validate_group_paulis_qwc(
    pauli_strs: list[ops.PauliString], all_qubits: Union[list[ops.Qid], FrozenSet[ops.Qid]]
):
    """Checks if a group of Pauli strings are Qubit-Wise Commuting.

    Args:
        pauli_strings: A list of cirq.PauliString objects.
        all_qubits: A list of all qubits to consider for the QWC check.
                    The check is performed for each qubit in this list.

    Returns:
        True if the group is QWC, False otherwise.
    """
    if len(pauli_strs) <= 1:
        return True
    for p1, p2 in itertools.combinations(pauli_strs, 2):
        if not _are_two_pauli_strings_qubit_wise_commuting(p1, p2, all_qubits):
            return False
    return True


def _validate_single_pauli_string(pauli_str: ops.PauliString):
    if not isinstance(pauli_str, ops.PauliString):
        raise TypeError(
            f"All elements in the Pauli string lists must be cirq.PauliString "
            f"instances, got {type(pauli_str)}."
        )

    if all(q == ops.I for q in pauli_str) or not pauli_str:
        raise ValueError(
            "Empty Pauli strings or Pauli strings consisting "
            "only of Pauli I are not allowed. Please provide "
            "valid input Pauli strings."
        )
    if pauli_str.coefficient.imag != 0:
        raise ValueError(
            "Cannot compute expectation value of a non-Hermitian PauliString. "
            "Coefficient must be real."
        )


def _validate_input(
    circuits_to_pauli: Union[
        Dict[circuits.FrozenCircuit, list[ops.PauliString]],
        Dict[circuits.FrozenCircuit, list[list[ops.PauliString]]],
    ],
    pauli_repetitions: int,
    readout_repetitions: int,
    num_random_bitstrings: int,
    rng_or_seed: Union[np.random.Generator, int],
):
    if not circuits_to_pauli:
        raise ValueError("Input circuits must not be empty.")

    for circuit in circuits_to_pauli.keys():
        if not isinstance(circuit, circuits.FrozenCircuit):
            raise TypeError("All keys in 'circuits_to_pauli' must be FrozenCircuit instances.")

    first_value: Union[list[ops.PauliString], list[list[ops.PauliString]]] = next(
        iter(circuits_to_pauli.values())  # type: ignore
    )
    for circuit, pauli_strs_list in circuits_to_pauli.items():
        if isinstance(pauli_strs_list, Sequence) and isinstance(first_value[0], Sequence):
            for pauli_strs in pauli_strs_list:
                if not pauli_strs:
                    raise ValueError("Empty group of Pauli strings is not allowed")
                if not (
                    isinstance(pauli_strs, Sequence) and isinstance(pauli_strs[0], ops.PauliString)
                ):
                    raise TypeError(
                        f"Inconsistent type in list for circuit {circuit}. "
                        f"Expected all elements to be sequences of ops.PauliString, "
                        f"but found {type(pauli_strs)}."
                    )
                if not _validate_group_paulis_qwc(pauli_strs, circuit.all_qubits()):
                    raise ValueError(
                        f"Pauli group containing {pauli_strs} is invalid: "
                        f"The group of Pauli strings are not "
                        f"Qubit-Wise Commuting with each other."
                    )
                for pauli_str in pauli_strs:
                    _validate_single_pauli_string(pauli_str)
        elif isinstance(pauli_strs_list, Sequence) and isinstance(first_value[0], ops.PauliString):
            for pauli_str in pauli_strs_list:  # type: ignore
                _validate_single_pauli_string(pauli_str)
        else:
            raise TypeError(
                f"Expected all elements to be either a sequence of PauliStrings"
                f" or sequences of ops.PauliStrings. "
                f"Got {type(pauli_strs_list)} instead."
            )

    # Check rng is a numpy random generator
    if not isinstance(rng_or_seed, np.random.Generator) and not isinstance(rng_or_seed, int):
        raise ValueError("Must provide a numpy random generator or a seed")

    # Check pauli_repetitions is bigger than 0
    if pauli_repetitions <= 0:
        raise ValueError("Must provide non-zero pauli_repetitions.")

    # Check num_random_bitstrings is bigger than or equal to 0
    if num_random_bitstrings < 0:
        raise ValueError("Must provide zero or more num_random_bitstrings.")

    # Check readout_repetitions is bigger than 0
    if readout_repetitions <= 0:
        raise ValueError("Must provide non-zero readout_repetitions for readout calibration.")


def _normalize_input_paulis(
    circuits_to_pauli: Union[
        Dict[circuits.FrozenCircuit, list[ops.PauliString]],
        Dict[circuits.FrozenCircuit, list[list[ops.PauliString]]],
    ],
) -> Dict[circuits.FrozenCircuit, list[list[ops.PauliString]]]:
    first_value = next(iter(circuits_to_pauli.values()))
    if (
        first_value
        and isinstance(first_value, list)
        and isinstance(first_value[0], ops.PauliString)
    ):
        input_dict = cast(Dict[circuits.FrozenCircuit, List[ops.PauliString]], circuits_to_pauli)
        normalized_circuits_to_pauli: Dict[circuits.FrozenCircuit, list[list[ops.PauliString]]] = {}
        for circuit, paulis in input_dict.items():
            normalized_circuits_to_pauli[circuit] = [[ps] for ps in paulis]
        return normalized_circuits_to_pauli
    return cast(Dict[circuits.FrozenCircuit, List[List[ops.PauliString]]], circuits_to_pauli)


def _pauli_strings_to_basis_change_ops(
    pauli_strings: list[ops.PauliString], qid_list: list[ops.Qid]
):
    operations = []
    for qubit in qid_list:
        for pauli_str in pauli_strings:
            pauli_op = pauli_str.get(qubit, default=ops.I)
            if pauli_op == ops.X:
                operations.append(ops.ry(-np.pi / 2)(qubit))  # =cirq.H
                break
            elif pauli_op == ops.Y:
                operations.append(ops.rx(np.pi / 2)(qubit))
                break
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

    for qubit in sorted(qubits_to_error.zero_state_errors.keys()):
        e0 = qubits_to_error.zero_state_errors[qubit]
        e1 = qubits_to_error.one_state_errors[qubit]
        cms.append(_build_one_qubit_confusion_matrix(e0, e1))
    return cms


def _build_many_one_qubits_empty_confusion_matrix(qubits_length: int) -> list[np.ndarray]:
    """Builds a list of empty confusion matrices"""
    return [_build_one_qubit_confusion_matrix(0, 0) for _ in range(qubits_length)]


def _process_pauli_measurement_results(
    qubits: list[ops.Qid],
    pauli_string_groups: list[list[ops.PauliString]],
    circuit_results: list[ResultDict],
    calibration_results: Dict[Tuple[ops.Qid, ...], SingleQubitReadoutCalibrationResult],
    pauli_repetitions: int,
    timestamp: float,
    disable_readout_mitigation: bool = False,
) -> list[PauliStringMeasurementResult]:
    """Calculates both error-mitigated expectation values and unmitigated expectation values
    from measurement results.

    This function takes the results from shuffled readout benchmarking and:
    1. Constructs a tensored confusion matrix for error mitigation.
    2. Mitigates readout errors for each Pauli string measurement.
    3. Calculates and returns both error-mitigated and unmitigated expectation values.

    Args:
        qubits: Qubits to build confusion matrices for. In a sorted order.
        pauli_strings: The lists of QWC Pauli string groups that are measured.
        circuit_results: A list of ResultDict obtained
            from running the Pauli measurement circuits.
        confusion_matrices: A list of confusion matrices from calibration results.
        pauli_repetitions: The number of repetitions used for Pauli string measurements.
        timestamp: The timestamp of the calibration results.
        disable_readout_mitigation: If set to True, returns no error-mitigated error
            expectation values.

    Returns:
        A list of PauliStringMeasurementResult.
    """

    pauli_measurement_results: List[PauliStringMeasurementResult] = []

    for pauli_group_index, circuit_result in enumerate(circuit_results):
        measurement_results = circuit_result.measurements["m"]
        pauli_strs = pauli_string_groups[pauli_group_index]

        for pauli_str in pauli_strs:
            qubits_sorted = sorted(pauli_str.qubits)
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
            raw_mitigated_values, raw_d_m = tensored_cm.readout_mitigation_pauli_uncorrelated(
                qubits_sorted, relevant_bits
            )
            mitigated_values_with_coefficient = raw_mitigated_values * pauli_str.coefficient.real
            d_m_with_coefficient = raw_d_m * abs(pauli_str.coefficient.real)

            # Calculate the unmitigated expectation.
            parity = np.sum(relevant_bits, axis=1) % 2
            raw_unmitigated_values = 1 - 2 * np.mean(parity)
            raw_d_unmit = 2 * np.sqrt(np.mean(parity) * (1 - np.mean(parity)) / pauli_repetitions)
            unmitigated_value_with_coefficient = raw_unmitigated_values * pauli_str.coefficient.real
            d_unmit_with_coefficient = raw_d_unmit * abs(pauli_str.coefficient.real)

            pauli_measurement_results.append(
                PauliStringMeasurementResult(
                    pauli_string=pauli_str,
                    mitigated_expectation=mitigated_values_with_coefficient,
                    mitigated_stddev=d_m_with_coefficient,
                    unmitigated_expectation=unmitigated_value_with_coefficient,
                    unmitigated_stddev=d_unmit_with_coefficient,
                    calibration_result=(
                        calibration_results[tuple(qubits_sorted)]
                        if disable_readout_mitigation is False
                        else None
                    ),
                )
            )

    return pauli_measurement_results


def measure_pauli_strings(
    circuits_to_pauli: Union[
        Dict[circuits.FrozenCircuit, List[ops.PauliString]],
        Dict[circuits.FrozenCircuit, List[List[ops.PauliString]]],
    ],
    sampler: work.Sampler,
    pauli_repetitions: int,
    readout_repetitions: int,
    num_random_bitstrings: int,
    rng_or_seed: Union[np.random.Generator, int],
) -> List[CircuitToPauliStringsMeasurementResult]:
    """Measures expectation values of Pauli strings on given circuits with/without
    readout error mitigation.

    This function takes a dictionary mapping circuits to lists of QWC Pauli string groups.
    For each circuit and its associated list of QWC pauli string group, it:
    1.  Constructs circuits to measure the Pauli string expectation value by
        adding basis change moments and measurement operations.
    2.  Runs shuffled readout benchmarking on these circuits to calibrate readout errors.
    3.  Mitigates readout errors using the calibrated confusion matrices.
    4.  Calculates and returns both error-mitigated and unmitigated expectation values for
    each Pauli string.

    Args:
        circuits_to_pauli: A dictionary mapping circuits to either:
            - A list of QWC groups (List[List[ops.PauliString]]). Each QWC group
              is a list of PauliStrings that are mutually Qubit-Wise Commuting.
              Pauli strings within the same group will be calculated using the
              same measurement results.
            - A list of PauliStrings (List[ops.PauliString]). In this case, each
              PauliString is treated as its own measurement group.
        sampler: The sampler to use.
        pauli_repetitions: The number of repetitions for each circuit when measuring
            Pauli strings.
        readout_repetitions: The number of repetitions for readout calibration
            in the shuffled benchmarking.
        num_random_bitstrings: The number of random bitstrings to use in readout
            benchmarking.
        rng_or_seed: A random number generator or seed for the readout benchmarking.

    Returns:
        A list of CircuitToPauliStringsMeasurementResult objects, where each object contains:
            - The circuit that was measured.
            - A list of PauliStringMeasurementResult objects.
            - The calibration result for single-qubit readout errors.
    """

    _validate_input(
        circuits_to_pauli,
        pauli_repetitions,
        readout_repetitions,
        num_random_bitstrings,
        rng_or_seed,
    )

    normalized_circuits_to_pauli = _normalize_input_paulis(circuits_to_pauli)

    # Extract unique qubit tuples from input pauli strings
    unique_qubit_tuples = set()
    for pauli_string_groups in normalized_circuits_to_pauli.values():
        for pauli_strings in pauli_string_groups:
            for pauli_string in pauli_strings:
                unique_qubit_tuples.add(tuple(sorted(pauli_string.qubits)))
    # qubits_list is a list of qubit tuples
    qubits_list = sorted(unique_qubit_tuples)

    # Build the basis-change circuits for each Pauli string group
    pauli_measurement_circuits = list[circuits.Circuit]()
    for input_circuit, pauli_string_groups in normalized_circuits_to_pauli.items():
        qid_list = list(sorted(input_circuit.all_qubits()))
        basis_change_circuits = []
        input_circuit_unfrozen = input_circuit.unfreeze()
        for pauli_strings in pauli_string_groups:
            basis_change_circuit = (
                input_circuit_unfrozen
                + _pauli_strings_to_basis_change_ops(pauli_strings, qid_list)
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
    for input_circuit, pauli_string_groups in normalized_circuits_to_pauli.items():

        qubits_in_circuit = tuple(sorted(input_circuit.all_qubits()))

        disable_readout_mitigation = False if num_random_bitstrings != 0 else True

        pauli_measurement_results = _process_pauli_measurement_results(
            list(qubits_in_circuit),
            pauli_string_groups,
            circuits_results[
                circuit_result_index : circuit_result_index + len(pauli_string_groups)
            ],
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

        circuit_result_index += len(pauli_string_groups)
    return results
