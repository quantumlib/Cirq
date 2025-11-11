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
from typing import cast, Sequence, TYPE_CHECKING

import attrs
import numpy as np
import sympy

import cirq.contrib.shuffle_circuits.shuffle_circuits_with_readout_benchmarking as sc_readout
from cirq import circuits, ops, study, work
from cirq.experiments.readout_confusion_matrix import TensoredConfusionMatrices
from cirq.study import ResultDict

if TYPE_CHECKING:
    from cirq.experiments.single_qubit_readout_calibration import (
        SingleQubitReadoutCalibrationResult,
    )


@attrs.frozen
class CircuitToPauliStringsParameters:
    """
    Parameters for measuring Pauli strings on a circuit.
    Attributes:
        circuit: The circuit to measure.
        pauli_strings:
            - A list of QWC groups (list[list[ops.PauliString]]). Each QWC group
              is a list of PauliStrings that are mutually Qubit-Wise Commuting.
              Pauli strings within the same group will be calculated using the
              same measurement results.
        postselection_symmetries: A tuple mapping Pauli strings or Pauli sums to
                                  expected values for postselection symmetries.
                                  Measured bitstrings which do not have the indicated
                                  values of the symmetry operators are postselected out.
    """

    circuit: circuits.FrozenCircuit
    pauli_strings: list[list[ops.PauliString]]
    postselection_symmetries: list[tuple[ops.PauliString | ops.PauliSum, int]]


@attrs.frozen
class PostFilteringSymmetryCalibrationResult:
    """Result of post-selection symmetry calibration.
    Attributes:
        raw_bitstrings: The raw bitstrings obtained from the measurement.
        filtered_bitstrings: The bitstrings after applying post-selection symmetries.
    """

    raw_bitstrings: np.ndarray
    filtered_bitstrings: np.ndarray


@attrs.frozen
class PauliStringMeasurementResult:
    """Result of measuring a Pauli string.

    Attributes:
        pauli_string: The Pauli string that is measured.
        mitigated_expectation: The error-mitigated expectation value of the Pauli string.
        mitigated_stddev: The standard deviation of the error-mitigated expectation value.
        unmitigated_expectation: The unmitigated expectation value of the Pauli string.
        unmitigated_stddev: The standard deviation of the unmitigated expectation value.
        calibration_result: The calibration result for readout errors. It can be either
           a SingleQubitReadoutCalibrationResult (in the case of mitigating with confusion
           matrices) or a PostFilteringSymmetryCalibrationResult (in the case of mitigating
           with post-selection symmetries).

    """

    pauli_string: ops.PauliString
    mitigated_expectation: float
    mitigated_stddev: float
    unmitigated_expectation: float
    unmitigated_stddev: float
    calibration_result: (
        SingleQubitReadoutCalibrationResult | PostFilteringSymmetryCalibrationResult | None
    ) = None


@attrs.frozen
class CircuitToPauliStringsMeasurementResult:
    """Result of measuring Pauli strings on a circuit.

    Attributes:
        circuit: The circuit that is measured.
        results: A list of PauliStringMeasurementResult objects.
    """

    circuit: circuits.FrozenCircuit
    results: list[PauliStringMeasurementResult]


def _commute_or_identity(
    op1: ops.Pauli | ops.IdentityGate, op2: ops.Pauli | ops.IdentityGate
) -> bool:
    if op1 == ops.I or op2 == ops.I:
        return True
    return op1 == op2


def _are_two_pauli_strings_qubit_wise_commuting(
    pauli_str1: ops.PauliString,
    pauli_str2: ops.PauliString,
    all_qubits: list[ops.Qid] | frozenset[ops.Qid],
) -> bool:
    for qubit in all_qubits:
        op1 = pauli_str1.get(qubit, default=ops.I)
        op2 = pauli_str2.get(qubit, default=ops.I)

        if not _commute_or_identity(op1, op2):
            return False
    return True


def _are_pauli_sum_and_pauli_string_qubit_wise_commuting(
    pauli_sum: ops.PauliSum,
    pauli_str: ops.PauliString,
    all_qubits: list[ops.Qid] | frozenset[ops.Qid],
) -> bool:
    """Checks if a Pauli sum and a Pauli string are Qubit-Wise Commuting."""
    for pauli_sum_term in pauli_sum:
        for qubit in all_qubits:
            op1 = pauli_sum_term.get(qubit, default=ops.I)
            op2 = pauli_str.get(qubit, default=ops.I)

            if not _commute_or_identity(op1, op2):
                return False
    return True


def _are_symmetry_and_pauli_string_qubit_wise_commuting(
    symmetry: ops.PauliString | ops.PauliSum,
    pauli_str: ops.PauliString,
    all_qubits: list[ops.Qid] | frozenset[ops.Qid],
) -> bool:
    """
    Checks if a symmetry (Pauli string or Pauli sum) and a Pauli string are Qubit-Wise Commuting.
    This is neccessary because the code's post-selection method relies on measuring both the symmetry
    and the Pauli string at the same time, using a single experimental shot.
    """
    if isinstance(symmetry, ops.PauliSum):
        return _are_pauli_sum_and_pauli_string_qubit_wise_commuting(symmetry, pauli_str, all_qubits)
    elif isinstance(symmetry, ops.PauliString):
        return _are_two_pauli_strings_qubit_wise_commuting(symmetry, pauli_str, all_qubits)
    else:
        return False


def _validate_group_paulis_qwc(
    pauli_strs: list[ops.PauliString], all_qubits: list[ops.Qid] | frozenset[ops.Qid]
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


def _validate_circuit_to_pauli_strings_parameters(
    circuits_to_pauli: list[CircuitToPauliStringsParameters],
):
    """Validates the input parameters for measuring Pauli strings.


    Args:
        circuits_to_pauli: A list of CircuitToPauliStringsParameters objects.


    Raises:
        ValueError: If any of the input parameters are invalid.
        TypeError: If the types of the input parameters are incorrect.
    """
    for i, params in enumerate(circuits_to_pauli):
        # 1. Validate Circuit
        if not params.circuit:
            raise ValueError(f"Item {i}: Circuit must not be empty.")
        if not isinstance(params.circuit, circuits.FrozenCircuit):
            raise TypeError(
                f"Item {i}: Expected circuit to be FrozenCircuit, got {type(params.circuit)}."
            )

        # 2. Validate Pauli strings
        if not params.pauli_strings:
            raise ValueError(f"Item {i}: Pauli strings list must not be empty.")

        for j, pauli_group in enumerate(params.pauli_strings):
            if not pauli_group:
                raise ValueError(
                    f"Item {i}, group {j}: Empty group of Pauli strings is not allowed."
                )
            if not isinstance(pauli_group[0], ops.PauliString):
                raise TypeError(
                    f"Item {i}, group {j}: Expected elements to be ops.PauliString, "
                    f"but found {type(pauli_group[0])}."
                )
            if not _validate_group_paulis_qwc(pauli_group, params.circuit.all_qubits()):
                raise ValueError(
                    f"Item {i}, group {j}: Pauli group {pauli_group} is not "
                    "Qubit-Wise Commuting."
                )
            for pauli_str in pauli_group:
                _validate_single_pauli_string(pauli_str)

        # 3. Validate postselection symmetries
        for sym, _ in params.postselection_symmetries:
            if isinstance(sym, ops.PauliSum):
                terms = list(sym)
                if not _validate_group_paulis_qwc(terms, params.circuit.all_qubits()):
                    raise ValueError(
                        f"Pauli sum {sym} for circuit {params.circuit} is invalid: "
                        "Terms are not Qubit-Wise Commuting."
                    )
                for term in terms:
                    _validate_single_pauli_string(term)
            elif isinstance(sym, ops.PauliString):
                _validate_single_pauli_string(sym)
            else:
                raise TypeError(
                    f"Postselection symmetry keys must be cirq.PauliString or cirq.PauliSum, "
                    f"got {type(sym)}."
                )

        # Check if input symmetries are commuting with all Pauli strings in the circuit
        qubits_in_circuit = list(sorted(params.circuit.all_qubits()))

        if not all(
            _are_symmetry_and_pauli_string_qubit_wise_commuting(sym, pauli_str, qubits_in_circuit)
            for pauli_strs in params.pauli_strings
            for pauli_str in pauli_strs
            for sym, _ in params.postselection_symmetries
        ):
            raise ValueError(
                f"Postselection symmetries of {params.circuit} are not commuting with all Pauli strings."
            )


def _validate_input(
    circuits_to_pauli: (
        dict[circuits.FrozenCircuit, list[ops.PauliString]]
        | dict[circuits.FrozenCircuit, list[list[ops.PauliString]]]
        | list[CircuitToPauliStringsParameters]
    ),
    pauli_repetitions: int,
    readout_repetitions: int,
    num_random_bitstrings: int,
    rng_or_seed: np.random.Generator | int,
):
    if not circuits_to_pauli:
        raise ValueError("Input circuits_to_pauli parameter must not be empty.")

    normalized_circuits_to_pauli = _validate_and_normalize_unformatted_input(circuits_to_pauli)

    _validate_circuit_to_pauli_strings_parameters(normalized_circuits_to_pauli)

    # Check rng is a numpy random generator
    if not isinstance(rng_or_seed, np.random.Generator) and not isinstance(rng_or_seed, int):
        raise ValueError("Must provide a numpy random generator or a seed")

    # Check pauli_repetitions is bigger than 0
    if pauli_repetitions <= 0:
        raise ValueError("Must provide positive pauli_repetitions.")

    # Check num_random_bitstrings is bigger than or equal to 0
    if num_random_bitstrings < 0:
        raise ValueError("Must provide zero or more num_random_bitstrings.")

    # Check readout_repetitions is bigger than 0
    if readout_repetitions <= 0:
        raise ValueError("Must provide positive readout_repetitions for readout calibration.")

    return normalized_circuits_to_pauli


def _validate_and_normalize_unformatted_input(
    circuits_input: (
        dict[circuits.FrozenCircuit, list[ops.PauliString]]
        | dict[circuits.FrozenCircuit, list[list[ops.PauliString]]]
        | list[CircuitToPauliStringsParameters]
    ),
) -> list[CircuitToPauliStringsParameters]:
    """Converts any valid input format into a standardized list of parameters
    where pauli_strings is always list[list[PauliString]]."""

    param_list: list[CircuitToPauliStringsParameters] = []

    # 1. Standardize to list[CircuitToPauliStringsParameters]
    if isinstance(circuits_input, dict):
        for circuit, paulis in circuits_input.items():
            # Normalize flat lists to nested lists
            normalized_paulis = paulis
            if paulis and isinstance(paulis, list) and isinstance(paulis[0], ops.PauliString):
                # Convert [PS, PS] -> [[PS], [PS]]
                normalized_paulis = [[cast(ops.PauliString, ps)] for ps in paulis]

            param_list.append(
                CircuitToPauliStringsParameters(
                    circuit=circuit,
                    pauli_strings=cast(list[list[ops.PauliString]], normalized_paulis),
                    postselection_symmetries=[],
                )
            )
    elif isinstance(circuits_input, list):
        param_list = circuits_input
    else:
        raise TypeError("Input must be a dict or a list of CircuitToPauliStringsParameters.")

    for params in param_list:
        if params.pauli_strings and isinstance(params.pauli_strings[0], ops.PauliString):
            raise TypeError(
                "CircuitToPauliStringsParameters was initialized with a flat list "
                "of PauliStrings. It must be initialized with a list of lists (groups)."
            )

    return param_list


def _extract_readout_qubits(pauli_strings: list[ops.PauliString]) -> list[ops.Qid]:
    """Extracts unique qubits from a list of QWC Pauli strings."""
    return sorted(set(q for ps in pauli_strings for q in ps.qubits))


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


def _pauli_strings_to_basis_change_with_sweep(
    pauli_strings: list[ops.PauliString], qid_list: list[ops.Qid]
) -> dict[str, float]:
    """Decide single-qubit rotation sweep parameters for basis change.

    Args:
        pauli_strings: A list of QWC Pauli strings.
        qid_list: A list of qubits to apply the basis change on.
    Returns:
        A dictionary mapping parameter names to their values for basis change.
    """
    params_dict = {}

    for qid, qubit in enumerate(qid_list):
        params_dict[f"phi{qid}"] = 1.0
        params_dict[f"theta{qid}"] = 0.0
        for pauli_str in pauli_strings:
            pauli_op = pauli_str.get(qubit, default=ops.I)
            if pauli_op == ops.X:
                params_dict[f"phi{qid}"] = 0.0
                params_dict[f"theta{qid}"] = 1 / 2
                break
            elif pauli_op == ops.Y:
                params_dict[f"phi{qid}"] = 1.0
                params_dict[f"theta{qid}"] = 1 / 2
                break
    return params_dict


def _generate_basis_change_circuits(
    normalized_circuits_to_pauli: list[CircuitToPauliStringsParameters],
    insert_strategy: circuits.InsertStrategy,
) -> list[circuits.Circuit]:
    """Generates basis change circuits for each group of Pauli strings."""
    pauli_measurement_circuits = list[circuits.Circuit]()

    for params in normalized_circuits_to_pauli:
        input_circuit = params.circuit
        pauli_string_groups = params.pauli_strings

        qid_list = list(sorted(input_circuit.all_qubits()))
        basis_change_circuits = []
        input_circuit_unfrozen = input_circuit.unfreeze()
        for pauli_strings in pauli_string_groups:
            basis_change_circuit = circuits.Circuit(
                input_circuit_unfrozen,
                _pauli_strings_to_basis_change_ops(pauli_strings, qid_list),
                ops.measure(*qid_list, key="result"),
                strategy=insert_strategy,
            )
            basis_change_circuits.append(basis_change_circuit)
        pauli_measurement_circuits.extend(basis_change_circuits)

    return pauli_measurement_circuits


def _generate_basis_change_circuits_with_sweep(
    normalized_circuits_to_pauli: list[CircuitToPauliStringsParameters],
    insert_strategy: circuits.InsertStrategy,
) -> tuple[list[circuits.Circuit], list[study.Sweepable]]:
    """Generates basis change circuits for each group of Pauli strings with sweep."""
    parameterized_circuits = list[circuits.Circuit]()
    sweep_params = list[study.Sweepable]()
    for params in normalized_circuits_to_pauli:
        input_circuit = params.circuit
        pauli_string_groups = params.pauli_strings

        qid_list = list(sorted(input_circuit.all_qubits()))
        phi_symbols = sympy.symbols(f"phi:{len(qid_list)}")
        theta_symbols = sympy.symbols(f"theta:{len(qid_list)}")

        # Create phased gates and measurement operator
        phased_gates = [
            ops.PhasedXPowGate(phase_exponent=(a - 1) / 2, exponent=b)(qubit)
            for a, b, qubit in zip(phi_symbols, theta_symbols, qid_list)
        ]
        measurement_op = ops.M(*qid_list, key="result")

        parameterized_circuit = circuits.Circuit(
            input_circuit.unfreeze(), phased_gates, measurement_op, strategy=insert_strategy
        )
        sweep_param = []
        for pauli_strings in pauli_string_groups:
            sweep_param.append(_pauli_strings_to_basis_change_with_sweep(pauli_strings, qid_list))
        sweep_params.append(sweep_param)
        parameterized_circuits.append(parameterized_circuit)

    return parameterized_circuits, sweep_params


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
    qubits: Sequence[ops.Qid],
    pauli_string_groups: list[list[ops.PauliString]],
    circuit_results: Sequence[ResultDict] | Sequence[study.Result],
    calibration_results: dict[tuple[ops.Qid, ...], SingleQubitReadoutCalibrationResult],
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
        pauli_string_groups: The lists of QWC Pauli string groups that are measured.
        circuit_results: A list of ResultDict obtained
            from running the Pauli measurement circuits.
        calibration_results: A dictionary of SingleQubitReadoutCalibrationResult
            for tuples of qubits present in `pauli_string_groups`.
        pauli_repetitions: The number of repetitions used for Pauli string measurements.
        timestamp: The timestamp of the calibration results.
        disable_readout_mitigation: If set to True, returns no error-mitigated error
            expectation values.

    Returns:
        A list of PauliStringMeasurementResult.
    """

    pauli_measurement_results: list[PauliStringMeasurementResult] = []

    for pauli_group_index, circuit_result in enumerate(circuit_results):
        measurement_results = circuit_result.measurements["result"]
        pauli_strs = pauli_string_groups[pauli_group_index]
        pauli_readout_qubits = _extract_readout_qubits(pauli_strs)

        calibration_result = (
            calibration_results[tuple(pauli_readout_qubits)]
            if not disable_readout_mitigation
            else None
        )

        for pauli_str in pauli_strs:
            qubits_sorted = sorted(pauli_str.qubits)
            qubit_indices = [qubits.index(q) for q in qubits_sorted]

            if disable_readout_mitigation:
                pauli_str_calibration_result = None
                confusion_matrices = _build_many_one_qubits_empty_confusion_matrix(
                    len(qubits_sorted)
                )
            else:
                if calibration_result is None:
                    # This case should be logically impossible if mitigation is on,
                    # so we raise an error.
                    raise ValueError(
                        f"Readout mitigation is enabled, but no calibration result was "
                        f"found for qubits {pauli_readout_qubits}."
                    )
                pauli_str_calibration_result = calibration_result.readout_result_for_qubits(
                    qubits_sorted
                )
                confusion_matrices = _build_many_one_qubits_confusion_matrix(
                    pauli_str_calibration_result
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
                    calibration_result=pauli_str_calibration_result,
                )
            )

    return pauli_measurement_results


def measure_pauli_strings(
    circuits_to_pauli: (
        dict[circuits.FrozenCircuit, list[ops.PauliString]]
        | dict[circuits.FrozenCircuit, list[list[ops.PauliString]]]
        | list[CircuitToPauliStringsParameters]
    ),
    sampler: work.Sampler,
    pauli_repetitions: int,
    readout_repetitions: int,
    num_random_bitstrings: int,
    rng_or_seed: np.random.Generator | int,
    use_sweep: bool = False,
    insert_strategy: circuits.InsertStrategy = circuits.InsertStrategy.INLINE,
) -> list[CircuitToPauliStringsMeasurementResult]:
    """Measures expectation values of Pauli strings on given circuits with/without
    readout error mitigation.

    For each circuit and its associated list of QWC pauli string group, it:
    1.  Constructs circuits to measure the Pauli string expectation value by
        adding basis change moments and measurement operations.
    2.  If `num_random_bitstrings` is greater than zero, performing readout
        benchmarking (shuffled or sweep-based) to calibrate readout errors.
    3.  Mitigates readout errors using the calibrated confusion matrices.
    4.  Calculates and returns both error-mitigated and unmitigated expectation values for
        each Pauli string.

    Args:
        circuits_to_pauli: A dictionary mapping circuits to either:
            - A list of QWC groups (list[list[ops.PauliString]]). Each QWC group
              is a list of PauliStrings that are mutually Qubit-Wise Commuting.
              Pauli strings within the same group will be calculated using the
              same measurement results.
            - A list of PauliStrings (list[ops.PauliString]). In this case, each
              PauliString is treated as its own measurement group.
            - Or a list of CircuitToPauliStringsParameters objects. Each object contains
             a circuit and its associated Pauli strings to measure. It could also contain
             a dictionary mapping Pauli strings or Pauli sums to expected eigen value
             for postselection symmetries.
        sampler: The sampler to use.
        pauli_repetitions: The number of repetitions for each circuit when measuring
            Pauli strings.
        readout_repetitions: The number of repetitions for readout calibration
            in the shuffled benchmarking.
        num_random_bitstrings: The number of random bitstrings to use in readout
            benchmarking.
        rng_or_seed: A random number generator or seed for the readout benchmarking.
        use_sweep: If True, uses parameterized circuits and sweeps parameters
            for both Pauli measurements and readout benchmarking. Defaults to False.
        insert_strategy: The strategy for inserting measurement operations into the circuit.
            Defaults to circuits.InsertStrategy.INLINE.

    Returns:
        A list of CircuitToPauliStringsMeasurementResult objects, where each object contains:
            - The circuit that was measured.
            - A list of PauliStringMeasurementResult objects.
            - The calibration result for single-qubit readout errors.
    """

    normalized_circuits_to_pauli = _validate_input(
        circuits_to_pauli,
        pauli_repetitions,
        readout_repetitions,
        num_random_bitstrings,
        rng_or_seed,
    )

    # Extract unique qubit tuples from input pauli strings
    unique_qubit_tuples = set()
    for circuit_to_pauli in normalized_circuits_to_pauli:
        for pauli_string_groups in circuit_to_pauli.pauli_strings:
            unique_qubit_tuples.add(tuple(_extract_readout_qubits(pauli_string_groups)))
    # qubits_list is a list of qubit tuples
    qubits_list = sorted(unique_qubit_tuples)

    # Build the basis-change circuits for each Pauli string group
    pauli_measurement_circuits: list[circuits.Circuit] = []
    sweep_params: list[study.Sweepable] = []
    circuits_results: Sequence[ResultDict] | Sequence[Sequence[study.Result]] = []
    calibration_results: dict[tuple[ops.Qid, ...], SingleQubitReadoutCalibrationResult] = {}

    benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=pauli_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if use_sweep:
        pauli_measurement_circuits, sweep_params = _generate_basis_change_circuits_with_sweep(
            normalized_circuits_to_pauli, insert_strategy
        )

        # Run benchmarking using sweep for readout calibration
        circuits_results, calibration_results = sc_readout.run_sweep_with_readout_benchmarking(
            sampler=sampler,
            input_circuits=pauli_measurement_circuits,
            sweep_params=sweep_params,
            parameters=benchmarking_params,
            rng_or_seed=rng_or_seed,
            qubits=[list(qubits) for qubits in qubits_list],
        )

    else:
        pauli_measurement_circuits = _generate_basis_change_circuits(
            normalized_circuits_to_pauli, insert_strategy
        )

        # Run shuffled benchmarking for readout calibration
        circuits_results, calibration_results = (
            sc_readout.run_shuffled_circuits_with_readout_benchmarking(
                sampler=sampler,
                input_circuits=pauli_measurement_circuits,
                parameters=benchmarking_params,
                rng_or_seed=rng_or_seed,
                qubits=[list(qubits) for qubits in qubits_list],
            )
        )

    # Process the results to calculate expectation values
    results: list[CircuitToPauliStringsMeasurementResult] = []
    circuit_result_index = 0
    for i, circuit_to_pauli in enumerate(normalized_circuits_to_pauli):
        input_circuit = circuit_to_pauli.circuit
        pauli_string_groups = circuit_to_pauli.pauli_strings

        qubits_in_circuit = tuple(sorted(input_circuit.all_qubits()))

        disable_readout_mitigation = False if num_random_bitstrings != 0 else True

        circuits_results_for_group: Sequence[ResultDict] | Sequence[study.Result] = []
        if use_sweep:
            circuits_results_for_group = cast(Sequence[Sequence[study.Result]], circuits_results)[i]
        else:
            circuits_results_for_group = cast(Sequence[ResultDict], circuits_results)[
                circuit_result_index : circuit_result_index + len(pauli_string_groups)
            ]
            circuit_result_index += len(pauli_string_groups)

        pauli_measurement_results = _process_pauli_measurement_results(
            list(qubits_in_circuit),
            pauli_string_groups,
            circuits_results_for_group,
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

    return results
