# Copyright 2021 The Cirq Developers
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

from cirq.circuits import Circuit
from cirq.ops import FSimGate, Gate, GateOperation, MeasurementGate, Moment, Qid, SingleQubitGate
from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    try_convert_sqrt_iswap_to_fsim,
)
from cirq.google.engine import Engine
from cirq.google.serializable_gate_set import SerializableGateSet


class IncompatibleMomentError(Exception):
    """Error that occurs when a moment is not supported by a calibration routine."""


def make_floquet_request_for_moment(
    moment: Moment,
    options: FloquetPhasedFSimCalibrationOptions,
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
    canonicalize_pairs: bool = False,
    sort_pairs: bool = False,
) -> Optional[FloquetPhasedFSimCalibrationRequest]:
    """Describes a given moment in terms of a Floquet characterization request.

    Args:
        moment: Moment to characterize.
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        canonicalize_pairs: Whether to sort each of the qubit pair so that the first qubit
            is always lower than the second.
        sort_pairs: Whether to sort all the qutibt pairs extracted from the moment which will
            undergo characterization.

    Returns:
        Instance of FloquetPhasedFSimCalibrationRequest that characterizes a given moment, or None
        when it is an empty, measurement or single-qubit gates only moment.

    Raises:
        IncompatibleMomentError when a moment contains operations other than the operations matched
        by gates_translator, or it mixes a single qubit and two qubit gates.
    """

    measurement = False
    single_qubit = False
    gate: Optional[FSimGate] = None
    pairs = []

    for op in moment:
        if not isinstance(op, GateOperation):
            raise IncompatibleMomentError('Moment contains operation different than GateOperation')

        if isinstance(op.gate, MeasurementGate):
            measurement = True
        elif isinstance(op.gate, SingleQubitGate):
            single_qubit = True
        else:
            translated_gate = gates_translator(op.gate)
            if translated_gate is None:
                raise IncompatibleMomentError(
                    f'Moment {moment} contains unsupported non-single qubit operation {op}'
                )
            elif gate is not None and gate != translated_gate:
                raise IncompatibleMomentError(
                    f'Moment {moment} contains operations resolved to two different gates {gate} '
                    f'and {translated_gate}'
                )
            else:
                gate = translated_gate
            pair = cast(
                Tuple[Qid, Qid], tuple(sorted(op.qubits) if canonicalize_pairs else op.qubits)
            )
            pairs.append(pair)

    if gate is None:
        # Either empty, single-qubit or measurement moment.
        return None

    if gate is not None and (measurement or single_qubit):
        raise IncompatibleMomentError(
            f'Moment contains mixed two-qubit operations and '
            f'single-qubit operations or measurement operations.'
        )

    return FloquetPhasedFSimCalibrationRequest(
        pairs=tuple(sorted(pairs) if sort_pairs else pairs), gate=gate, options=options
    )


def make_floquet_request_for_circuit(
    circuit: Circuit,
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    initial: Optional[Tuple[List[FloquetPhasedFSimCalibrationRequest], List[Optional[int]]]] = None,
) -> Tuple[List[FloquetPhasedFSimCalibrationRequest], List[Optional[int]]]:
    """Extracts a minimal set of Floquet characterization requests necessary to characterize given
    circuit.

    The circuit can only be composed of single qubit operations, measurement operations and
    operations supported by gates_translator.

    Args:
        circuit: Circuit to characterize.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            return value of make_floquet_request_for_circuit invoked on another circuit. This might
            be used to find a minimal set of moments to characterize across many circuits.

    Returns:
        Tuple of:
          - list of calibration requests,
          - list of indices of the generated characterization requests for each moment in the
            supplied circuit. If None occurs at certain position, it means that the related moment
            does not require characterization.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """

    pairs_map: Dict[Tuple[Tuple[Qid, Qid], ...], int] = {}

    def append_if_missing(calibration: FloquetPhasedFSimCalibrationRequest) -> int:
        if calibration.pairs not in pairs_map:
            index = len(calibrations)
            calibrations.append(calibration)
            pairs_map[calibration.pairs] = index
            return index
        else:
            return pairs_map[calibration.pairs]

    def merge_into_calibrations(calibration: FloquetPhasedFSimCalibrationRequest) -> int:
        new_pairs = set(calibration.pairs)
        for index in pairs_map.values():
            assert calibration.gate == calibrations[index].gate
            assert calibration.options == calibrations[index].options
            existing_pairs = calibrations[index].pairs
            if new_pairs.issubset(existing_pairs):
                return index
            elif new_pairs.issuperset(existing_pairs):
                calibrations[index] = calibration
                return index
            else:
                new_qubit_pairs = calibration.qubit_to_pair
                existing_qubit_pairs = calibrations[index].qubit_to_pair
                if all(
                    (
                        new_qubit_pairs[q] == existing_qubit_pairs[q]
                        for q in set(new_qubit_pairs.keys()).intersection(
                            existing_qubit_pairs.keys()
                        )
                    )
                ):
                    calibrations[index] = FloquetPhasedFSimCalibrationRequest(
                        gate=calibration.gate,
                        pairs=tuple(sorted(new_pairs.union(existing_pairs))),
                        options=options,
                    )
                    return index

        index = len(calibrations)
        calibrations.append(calibration)
        pairs_map[calibration.pairs] = index
        return index

    if initial is None:
        calibrations: List[FloquetPhasedFSimCalibrationRequest] = []
        moments_map: List[Optional[int]] = []
    else:
        calibrations, moments_map = initial

    for moment in circuit:
        calibration = make_floquet_request_for_moment(
            moment, options, gates_translator, canonicalize_pairs=True, sort_pairs=True
        )

        if calibration is not None:
            if merge_subsets:
                index = merge_into_calibrations(calibration)
            else:
                index = append_if_missing(calibration)
            moments_map.append(index)
        else:
            moments_map.append(None)

    return calibrations, moments_map


def run_characterizations(
    calibrations: Sequence[PhasedFSimCalibrationRequest],
    engine: Engine,
    processor_id: str,
    gate_set: SerializableGateSet,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> List[PhasedFSimCalibrationResult]:
    """Runs calibration requests on the Engine.

    Args:
        calibrations: List of calibrations to perform described in a request object.
        engine: cirq.google.Engine object used for running the calibrations.
        processor_id: processor_id passed to engine.run_calibrations method.
        gate_set: Gate set to use for characterization request.
        max_layers_per_request: Maximum number of calibration requests issued to cirq.Engine at a
            single time. Defaults to 1.
        progress_func: Optional callback function that might be used to report the calibration
            progress. The callback is called with two integers, the first one being a number of
            layers already calibrated and the second one the total number of layers to calibrate.

    Returns:
        List of PhasedFSimCalibrationResult for each requested calibration.
    """
    if max_layers_per_request < 1:
        raise ValueError(
            f'Miaximum number of layers pere request must be at least 1, '
            f'{max_layers_per_request} given'
        )

    if not calibrations:
        return []

    requests = [
        [
            calibration.to_calibration_layer()
            for calibration in calibrations[offset : offset + max_layers_per_request]
        ]
        for offset in range(0, len(calibrations), max_layers_per_request)
    ]

    results = []

    for request in requests:
        job = engine.run_calibration(request, processor_id=processor_id, gate_set=gate_set)
        results += [
            calibration.parse_result(result)
            for calibration, result in zip(calibrations, job.calibration_results())
        ]
        if progress_func:
            progress_func(len(results), len(calibrations))

    return results


def run_floquet_characterization_for_circuit(
    circuit: Circuit,
    engine: Engine,
    processor_id: str,
    gate_set: SerializableGateSet,
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[PhasedFSimCalibrationResult], List[Optional[int]]]:
    """Extracts moments within a circuit to characterize and characterizes them against engine.

    The method calls floquet_characterization_for_circuit to extract moments to characterize and
    run_characterizations to characterize them.

    Args:
        circuit: Circuit to characterize.
        engine: cirq.google.Engine object used for running the calibrations.
        processor_id: processor_id passed to engine.run_calibrations method.
        gate_set: Gate set to use for characterization request.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        max_layers_per_request: Maximum number of calibration requests issued to cirq.Engine at a
            single time. Defaults to 1.
        progress_func: Optional callback function that might be used to report the calibration
            progress. The callback is called with two integers, the first one being a number of
            layers already calibrated and the second one the total number of layers to calibrate.

    Returns:
        Tuple of:
          - list of PhasedFSimCalibrationResult for each recognized moment to characterize.
          - list of indices of the generated characterization requests for each moment in the
            supplied circuit. If None occurs at certain position, it means that the related moment
            does not require characterization.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    requests, allocations = make_floquet_request_for_circuit(
        circuit, options, gates_translator, merge_subsets=merge_subsets
    )
    results = run_characterizations(
        requests,
        engine,
        processor_id,
        gate_set,
        max_layers_per_request=max_layers_per_request,
        progress_func=progress_func,
    )
    return results, allocations
