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
import dataclasses
import itertools
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    TYPE_CHECKING,
)

from cirq.circuits import Circuit
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq.ops import (
    FSimGate,
    Gate,
    GateOperation,
    MeasurementGate,
    Moment,
    Operation,
    Qid,
    SingleQubitGate,
    WaitGate,
)
from cirq.work import Sampler
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    merge_matching_results,
    try_convert_sqrt_iswap_to_fsim,
    PhasedFSimCalibrationOptions,
    RequestT,
    LocalXEBPhasedFSimCalibrationRequest,
)
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import Engine, QuantumEngineSampler
from cirq_google.serializable_gate_set import SerializableGateSet

if TYPE_CHECKING:
    import cirq

_CALIBRATION_IRRELEVANT_GATES = MeasurementGate, SingleQubitGate, WaitGate


@dataclasses.dataclass(frozen=True)
class CircuitWithCalibration:
    """Circuit with characterization data annotations.

    Attributes:
        circuit: Circuit instance.
        moment_to_calibration: Maps each moment within a circuit to an index of a characterization
            request or response. None means that there is no characterization data for that moment.
    """

    circuit: Circuit
    moment_to_calibration: Sequence[Optional[int]]


def prepare_characterization_for_moment(
    moment: Moment,
    options: PhasedFSimCalibrationOptions[RequestT],
    *,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    canonicalize_pairs: bool = False,
    sort_pairs: bool = False,
) -> Optional[RequestT]:
    """Describes a given moment in terms of a characterization request.

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
        Instance of a calibration request that characterizes a given moment, or None
        when it is an empty, measurement or single-qubit gates only moment.

    Raises:
        IncompatibleMomentError when a moment contains operations other than the operations matched
        by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    pairs_and_gate = _list_moment_pairs_to_characterize(
        moment, gates_translator, canonicalize_pairs=canonicalize_pairs, permit_mixed_moments=False
    )
    if pairs_and_gate is None:
        return None

    pairs, gate = pairs_and_gate
    return options.create_phased_fsim_request(
        pairs=tuple(sorted(pairs) if sort_pairs else pairs), gate=gate
    )


def prepare_floquet_characterization_for_moment(
    moment: Moment,
    options: FloquetPhasedFSimCalibrationOptions,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
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
    return prepare_characterization_for_moment(
        moment=moment,
        options=options,
        gates_translator=gates_translator,
        canonicalize_pairs=canonicalize_pairs,
        sort_pairs=sort_pairs,
    )


def _list_moment_pairs_to_characterize(
    moment: Moment,
    gates_translator: Callable[[Gate], Optional[PhaseCalibratedFSimGate]],
    canonicalize_pairs: bool,
    permit_mixed_moments: bool,
) -> Optional[Tuple[List[Tuple[Qid, Qid]], Gate]]:
    """Helper function to describe a given moment in terms of a characterization request.

    Args:
        moment: Moment to characterize.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization.
        canonicalize_pairs: Whether to sort each of the qubit pair so that the first qubit
            is always lower than the second.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Tuple with list of pairs to characterize and gate that should be used for characterization,
        or None when no gate to characterize exists in a given moment.

    Raises:
        IncompatibleMomentError when a moment contains operations other than the operations matched
        by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    other_operation = False
    gate: Optional[FSimGate] = None
    pairs = []

    for op in moment:
        if not isinstance(op, GateOperation):
            raise IncompatibleMomentError('Moment contains operation different than GateOperation')

        if isinstance(op.gate, _CALIBRATION_IRRELEVANT_GATES):
            other_operation = True
        else:
            translated = gates_translator(op.gate)
            if translated is None:
                raise IncompatibleMomentError(
                    f'Moment {moment} contains unsupported non-single qubit operation {op}'
                )

            if gate is not None and gate != translated.engine_gate:
                raise IncompatibleMomentError(
                    f'Moment {moment} contains operations resolved to two different gates {gate} '
                    f'and {translated.engine_gate}'
                )
            else:
                gate = translated.engine_gate

            pair = cast(
                Tuple[Qid, Qid], tuple(sorted(op.qubits) if canonicalize_pairs else op.qubits)
            )
            pairs.append(pair)

    if gate is None:
        # Either empty, single-qubit or measurement moment.
        return None
    elif not permit_mixed_moments and other_operation:
        raise IncompatibleMomentError(
            f'Moment contains mixed two-qubit operations and either single-qubit measurement or '
            f'wait operations.'
        )

    return pairs, gate


def prepare_characterization_for_moments(
    circuit: Circuit,
    options: PhasedFSimCalibrationOptions[RequestT],
    *,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    initial: Optional[Sequence[RequestT]] = None,
) -> Tuple[CircuitWithCalibration, List[RequestT]]:
    """Extracts a minimal set of characterization requests necessary to characterize given circuit.

    This prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    See also prepare_characterization_for_circuits_moments that operates on a list of circuits.

    Args:
        circuit: Circuit to characterize.
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: If `True` then this method tries to merge moments into the other moments
            listed previously if they can be characterized together (they have no conflicting
            operations). Otherwise, only moments of exactly the same structure are characterized
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            the requests field of the return value of prepare_characterization_for_moments invoked
            on another circuit. This might be used to find a minimal set of moments to characterize
            across many circuits.

    Returns:
        circuit_with_calibration:
            The circuit and its mapping from moments to indices into the list of calibration
            requests (the second returned value).
        calibrations:
            A list of calibration requests for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    if initial is None:
        initial = []

    allocations: List[Optional[int]] = []
    calibrations = list(initial)
    pairs_map = {calibration.pairs: index for index, calibration in enumerate(calibrations)}

    for moment in circuit:
        calibration = prepare_characterization_for_moment(
            moment,
            options,
            gates_translator=gates_translator,
            canonicalize_pairs=True,
            sort_pairs=True,
        )

        if calibration is not None:
            if merge_subsets:
                index = _merge_into_calibrations(calibration, calibrations, pairs_map, options)
            else:
                index = _append_into_calibrations_if_missing(calibration, calibrations, pairs_map)
            allocations.append(index)
        else:
            allocations.append(None)

    return CircuitWithCalibration(circuit, allocations), calibrations


def prepare_characterization_for_circuits_moments(
    circuits: List[Circuit],
    options: PhasedFSimCalibrationOptions[RequestT],
    *,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    initial: Optional[Sequence[RequestT]] = None,
) -> Tuple[List[CircuitWithCalibration], List[RequestT]]:
    """Extracts a minimal set of characterization requests necessary to characterize given circuits.

    This prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    See also prepare_characterization_for_moments that operates on a single circuit.

    Args:
        circuits: Circuits list to characterize.
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: If `True` then this method tries to merge moments into the other moments
            listed previously if they can be characterized together (they have no conflicting
            operations). Otherwise, only moments of exactly the same structure are characterized
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            the requests field of the return value of prepare_characterization_for_moments invoked
            on another circuit. This might be used to find a minimal set of moments to characterize
            across many circuits.

    Returns:
        circuits_with_calibration:
            The circuit and its mapping from moments to indices into the list of calibration
            requests (the second returned value). When list of circuits was passed on input, this
            will be a list of CircuitWithCalibration objects corresponding to each circuit on the
            input list.
        calibrations:
            A list of calibration requests for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    requests = list(initial) if initial is not None else []
    circuits_with_calibration = []
    for circuit in circuits:
        circuit_with_calibration, requests = prepare_characterization_for_moments(
            circuit,
            options,
            gates_translator=gates_translator,
            merge_subsets=merge_subsets,
            initial=requests,
        )
        circuits_with_calibration.append(circuit_with_calibration)
    return circuits_with_calibration, requests


def prepare_floquet_characterization_for_moments(
    circuit: Circuit,
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    initial: Optional[Sequence[FloquetPhasedFSimCalibrationRequest]] = None,
) -> Tuple[CircuitWithCalibration, List[FloquetPhasedFSimCalibrationRequest]]:
    """Extracts a minimal set of Floquet characterization requests necessary to characterize given
    circuit.

    This variant of prepare method works on moments of the circuit and assumes that all the
    two-qubit gates to calibrate are not mixed with other gates in a moment. The method groups
    together moments of similar structure to minimize the number of characterizations requested.

    If merge_subsets parameter is True then the method tries to merge moments into the other moments
    listed previously if they can be characterized together (they have no conflicting operations).
    If merge_subsets is False then only moments of exactly the same structure are characterized
    together.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit to characterize.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        initial: The characterization requests obtained by a previous scan of another circuit; i.e.,
            the requests field of the return value of make_floquet_request_for_circuit invoked on
            another circuit. This might be used to find a minimal set of moments to characterize
            across many circuits.

    Returns:
        Tuple of:
          - Circuit and its mapping from moments to indices into the list of calibration requests
            (the second returned value).
          - List of PhasedFSimCalibrationRequest for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    return cast(
        Tuple[CircuitWithCalibration, List[FloquetPhasedFSimCalibrationRequest]],
        prepare_characterization_for_moments(
            circuit,
            options,
            gates_translator=gates_translator,
            merge_subsets=merge_subsets,
            initial=initial,
        ),
    )


def prepare_characterization_for_operations(
    circuit: Union[Circuit, Iterable[Circuit]],
    options: PhasedFSimCalibrationOptions[RequestT],
    *,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    permit_mixed_moments: bool = False,
) -> List[RequestT]:
    """Extracts a minimal set of characterization requests necessary to characterize all the
    operations within a circuit(s).

    This prepare method works on two-qubit operations of the circuit. The method extracts
    all the operations and groups them in a way to minimize the number of characterizations
    requested, depending on the connectivity.

    Contrary to prepare_characterization_for_moments, this method ignores moments structure
    and is less accurate because certain errors caused by cross-talk are ignored.

    The major advantage of this method is that the number of generated characterization requests is
    bounded by four for grid-like devices, where for
    prepare_characterization_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit or circuits to characterize. Only circuits with qubits of type GridQubit
            that can be covered by HALF_GRID_STAGGERED_PATTERN are supported
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        List of PhasedFSimCalibrationRequest for each group of operations to characterize.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """

    circuits = [circuit] if isinstance(circuit, Circuit) else circuit
    pairs, gate = _extract_all_pairs_to_characterize(
        circuits, gates_translator, permit_mixed_moments
    )

    if gate is None:
        return []

    characterizations = []
    for pattern in HALF_GRID_STAGGERED_PATTERN:
        pattern_pairs = [pair for pair in pairs if pair in pattern]
        if pattern_pairs:
            characterizations.append(
                options.create_phased_fsim_request(pairs=tuple(sorted(pattern_pairs)), gate=gate)
            )

    if sum((len(characterization.pairs) for characterization in characterizations)) != len(pairs):
        raise ValueError('Unable to cover all interactions with HALF_GRID_STAGGERED_PATTERN')

    return characterizations


def prepare_floquet_characterization_for_operations(
    circuit: Union[Circuit, Iterable[Circuit]],
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    permit_mixed_moments: bool = False,
) -> List[FloquetPhasedFSimCalibrationRequest]:
    """Extracts a minimal set of Floquet characterization requests necessary to characterize all the
    operations within a circuit(s).

    This variant of prepare method works on two-qubit operations of the circuit. The method extracts
    all the operations and groups them in a way to minimize the number of characterizations
    requested, depending on the connectivity.

    Contrary to prepare_floquet_characterization_for_moments, this method ignores moments structure
    and is less accurate because certain errors caused by cross-talks are ignored.

    The major advantage of this method is that the number of generated characterization requests is
    bounded by four for grid-like devices, where for the
    prepare_floquet_characterization_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit or circuits to characterize. Only circuits with qubits of type GridQubit
            that can be covered by HALF_GRID_STAGGERED_PATTERN are supported
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        List of FloquetPhasedFSimCalibrationRequest for each group of operations to characterize.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    return prepare_characterization_for_operations(
        circuit=circuit,
        options=options,
        gates_translator=gates_translator,
        permit_mixed_moments=permit_mixed_moments,
    )


def _extract_all_pairs_to_characterize(
    circuits: Iterable[Circuit],
    gates_translator: Callable[[Gate], Optional[PhaseCalibratedFSimGate]],
    permit_mixed_moments: bool,
) -> Tuple[Set[Tuple[Qid, Qid]], Optional[Gate]]:
    """Extracts the set of all two-qubit operations from the circuits.

    Args:
        circuits: Circuits to extract the operations from.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Tuple with set of all two-qubit interacting pairs and a common gate that represents those
        interactions. The gate can be used for characterization purposes. If no interactions are
        present the gate is None.
    """

    all_pairs = set()
    common_gate = None
    for circuit in circuits:
        for moment in circuit:
            pairs_and_gate = _list_moment_pairs_to_characterize(
                moment,
                gates_translator,
                canonicalize_pairs=True,
                permit_mixed_moments=permit_mixed_moments,
            )

            if pairs_and_gate is not None:
                pairs, gate = pairs_and_gate

                if common_gate is None:
                    common_gate = gate
                elif common_gate != gate:
                    raise ValueError(
                        f'Only a single type of gate is supported, got {gate} and {common_gate}'
                    )

                all_pairs.update(pairs)

    return all_pairs, common_gate


def _append_into_calibrations_if_missing(
    calibration: RequestT,
    calibrations: List[RequestT],
    pairs_map: Dict[Tuple[Tuple[Qid, Qid], ...], int],
) -> int:
    """Adds calibration to the calibrations list if not already present.

    This function uses equivalence of calibration.pairs as a presence check.

    Args:
        calibration: Calibration to be added.
        calibrations: List of calibrations to be mutated. The list is expanded only if a calibration
            is not on the list already.
        pairs_map: Map from pairs parameter of each calibration on the calibrations list to the
            index on that list. This map will be updated if the calibrations list us expanded.

    Returns:
        Index of the calibration on the updated calibrations list. If the calibration was added, it
        points to the last element of a list. If not, it points to already existing element.
    """
    if calibration.pairs not in pairs_map:
        index = len(calibrations)
        calibrations.append(calibration)
        pairs_map[calibration.pairs] = index
        return index
    else:
        return pairs_map[calibration.pairs]


def _merge_into_calibrations(
    calibration: RequestT,
    calibrations: List[RequestT],
    pairs_map: Dict[Tuple[Tuple[Qid, Qid], ...], int],
    options: PhasedFSimCalibrationOptions[RequestT],
) -> int:
    """Merges a calibration into list of calibrations.

    If calibrations contains an item of which pairs could be expanded to include a new calibration
    pairs, without breaking a moment structure, then those two calibrations will be merged together
    and used as a calibration for both old and newly added calibration.
    If no calibration like that exists, the list will be expanded by the calibration item.

    Args:
        calibration: Calibration to be added.
        calibrations: List of calibrations to be mutated.
        pairs_map: Map from pairs parameter of each calibration on the calibrations list to the
            index on that list. This map will be updated if the calibrations list us updated.
        options: Calibrations options to use when creating a new requests.

    Returns:
        Index of the calibration on the updated calibrations list. If the calibration was added, it
        points to the last element of a list. If not, it points to already existing element.
    """
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
                    for q in set(new_qubit_pairs.keys()).intersection(existing_qubit_pairs.keys())
                )
            ):
                calibrations[index] = options.create_phased_fsim_request(
                    gate=calibration.gate,
                    pairs=tuple(sorted(new_pairs.union(existing_pairs))),
                )
                return index

    index = len(calibrations)
    calibrations.append(calibration)
    pairs_map[calibration.pairs] = index
    return index


def _run_calibrations_via_engine(
    calibration_requests: Sequence[PhasedFSimCalibrationRequest],
    engine: Engine,
    processor_id: str,
    gate_set: SerializableGateSet,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
):
    """Helper function for run_calibrations.

    This batches and runs calibration requests the normal way: by using engine.run_calibration.
    This function assumes that all inputs have been validated (by `run_calibrations`).
    """
    results = []
    nested_calibration_layers = [
        [
            calibration.to_calibration_layer()
            for calibration in calibration_requests[offset : offset + max_layers_per_request]
        ]
        for offset in range(0, len(calibration_requests), max_layers_per_request)
    ]

    for cal_layers in nested_calibration_layers:
        job = engine.run_calibration(cal_layers, processor_id=processor_id, gate_set=gate_set)
        request_results = job.calibration_results()
        results += [
            calibration.parse_result(result, job)
            for calibration, result in zip(calibration_requests, request_results)
        ]
        if progress_func:
            progress_func(len(results), len(calibration_requests))
    return results


def _run_local_calibrations_via_sampler(
    calibration_requests: Sequence[PhasedFSimCalibrationRequest],
    sampler: 'cirq.Sampler',
):
    """Helper function used by `run_calibrations` to run Local calibrations with a Sampler."""
    return [
        run_local_xeb_calibration(
            cast(LocalXEBPhasedFSimCalibrationRequest, calibration_request), sampler
        )
        for calibration_request in calibration_requests
    ]


def run_calibrations(
    calibrations: Sequence[PhasedFSimCalibrationRequest],
    sampler: Union[Engine, Sampler],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> List[PhasedFSimCalibrationResult]:
    """Runs calibration requests on the Engine.

    Args:
        calibrations: List of calibrations to perform described in a request object.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.QuantumEngineSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        gate_set: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
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
            f'Maximum number of layers per request must be at least 1, {max_layers_per_request} '
            f'given'
        )

    if not calibrations:
        return []

    calibration_request_types = set(type(cr) for cr in calibrations)
    if len(calibration_request_types) > 1:
        raise ValueError(
            f"All calibrations must be of the same type. You gave: {calibration_request_types}"
        )
    (calibration_request_type,) = calibration_request_types

    if isinstance(sampler, Engine):
        engine: Optional[Engine] = sampler
    elif isinstance(sampler, QuantumEngineSampler):
        engine = sampler.engine
        (processor_id,) = sampler._processor_ids
        gate_set = sampler._gate_set
    else:
        engine = None

    if engine is not None:
        if processor_id is None:
            raise ValueError('processor_id must be provided.')
        if gate_set is None:
            raise ValueError('gate_set must be provided.')

        if calibration_request_type == LocalXEBPhasedFSimCalibrationRequest:
            sampler = engine.sampler(processor_id=processor_id, gate_set=gate_set)
            return _run_local_calibrations_via_sampler(calibrations, sampler)

        return _run_calibrations_via_engine(
            calibrations,
            engine,
            processor_id,
            gate_set,
            max_layers_per_request,
            progress_func,
        )

    if calibration_request_type == LocalXEBPhasedFSimCalibrationRequest:
        return _run_local_calibrations_via_sampler(calibrations, sampler=cast(Sampler, sampler))

    if isinstance(sampler, PhasedFSimEngineSimulator):
        return sampler.get_calibrations(calibrations)

    raise ValueError(
        f'Unsupported sampler/request combination: Sampler {sampler} cannot run '
        f'calibration request of type {calibration_request_type}'
    )


def make_zeta_chi_gamma_compensation_for_moments(
    circuit_with_calibration: CircuitWithCalibration,
    characterizations: List[PhasedFSimCalibrationResult],
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
) -> CircuitWithCalibration:
    """Compensates circuit moments against errors in zeta, chi and gamma angles.

    This method creates a new circuit with a single-qubit Z gates added in a such way so that
    zeta, chi and gamma angles discovered by characterizations are cancelled-out and set to 0.

    This function preserves a moment structure of the circuit. All single qubit gates appear on new
    moments in the final circuit.

    Args:
        circuit_with_calibration: A CircuitWithCalibration (likely returned from run_calibrations)
            whose mapping argument corresponds to the results in the characterizations argument.
        characterizations: List of characterization results (likely returned from run_calibrations).
            This should correspond to the circuit and mapping in the circuit_with_calibration
            argument.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.

    Returns:
        Calibrated circuit together with its calibration metadata in CircuitWithCalibration object.
        The calibrated circuit has single-qubit Z gates added which compensates for the true gates
        imperfections.
        The moment to calibration mapping is updated for the new circuit so that successive
        calibrations could be applied.
    """
    return _make_zeta_chi_gamma_compensation(
        circuit_with_calibration, characterizations, gates_translator, permit_mixed_moments=False
    )


def make_zeta_chi_gamma_compensation_for_operations(
    circuit: Circuit,
    characterizations: List[PhasedFSimCalibrationResult],
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    permit_mixed_moments: bool = False,
) -> Circuit:
    """Compensates circuit operations against errors in zeta, chi and gamma angles.

    This method creates a new circuit with a single-qubit Z gates added in a such way so that
    zeta, chi and gamma angles discovered by characterizations are cancelled-out and set to 0.

    Contrary to make_zeta_chi_gamma_compensation_for_moments this method does not match
    characterizations to the moment structure of the circuits and thus is less accurate because
    some errors caused by cross-talks are not mitigated.

    The major advantage of this method over make_zeta_chi_gamma_compensation_for_moments is that it
    can work with arbitrary set of characterizations that cover all the interactions of the circuit
    (up to assumptions of merge_matching_results method). In particular, for grid-like devices the
    number of characterizations is bounded by four, where in the case of
    make_zeta_chi_gamma_compensation_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    This function preserves a moment structure of the circuit. All single qubit gates appear on new
    moments in the final circuit.

    Args:
        circuit: Circuit to calibrate.
        characterizations: List of characterization results (likely returned from run_calibrations).
            All the characterizations must be compatible in sense of merge_matching_results, they
            will be merged together.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow mixing single-qubit and two-qubit gates in a single
            moment.

    Returns:
        Calibrated circuit with a single-qubit Z gates added which compensates for the true gates
        imperfections.
    """

    characterization = merge_matching_results(characterizations)
    moment_to_calibration = [0] * len(circuit)
    calibrated = _make_zeta_chi_gamma_compensation(
        CircuitWithCalibration(circuit, moment_to_calibration),
        [characterization] if characterization is not None else [],
        gates_translator,
        permit_mixed_moments=permit_mixed_moments,
    )
    return calibrated.circuit


def _make_zeta_chi_gamma_compensation(
    circuit_with_calibration: CircuitWithCalibration,
    characterizations: List[PhasedFSimCalibrationResult],
    gates_translator: Callable[[Gate], Optional[PhaseCalibratedFSimGate]],
    permit_mixed_moments: bool,
) -> CircuitWithCalibration:

    if permit_mixed_moments:
        raise NotImplementedError('Mixed moments compensation ist supported yet')

    if len(circuit_with_calibration.circuit) != len(circuit_with_calibration.moment_to_calibration):
        raise ValueError('Moment allocations does not match circuit length')

    default_phases = PhasedFSimCharacterization(zeta=0.0, chi=0.0, gamma=0.0)

    compensated = Circuit()
    compensated_moment_to_calibration: List[Optional[int]] = []
    for moment, characterization_index in zip(
        circuit_with_calibration.circuit, circuit_with_calibration.moment_to_calibration
    ):
        parameters = None
        if characterization_index is not None:
            parameters = characterizations[characterization_index]

        decompositions: List[Tuple[Tuple[Operation, ...], ...]] = []
        other: List[Operation] = []
        new_moment_moment_to_calibration: Optional[List[Optional[int]]] = None
        for op in moment:
            if not isinstance(op, GateOperation):
                raise IncompatibleMomentError(
                    'Moment contains operation different than GateOperation'
                )

            if isinstance(op.gate, _CALIBRATION_IRRELEVANT_GATES):
                other.append(op)
                continue

            a, b = op.qubits
            translated = gates_translator(op.gate)
            if translated is None:
                raise IncompatibleMomentError(
                    f'Moment {moment} contains unsupported non-single qubit operation {op}'
                )

            if parameters is None:
                raise ValueError(f'Missing characterization data for moment {moment}')

            pair_parameters = parameters.get_parameters(a, b)
            if pair_parameters is None:
                raise ValueError(f'Missing characterization data for pair {(a, b)} in {parameters}')
            pair_parameters = pair_parameters.merge_with(default_phases)

            corrections = FSimPhaseCorrections.from_characterization(
                (a, b),
                translated,
                pair_parameters,
                characterization_index,
            )
            decompositions.append(corrections.operations)

            if new_moment_moment_to_calibration is None:
                new_moment_moment_to_calibration = corrections.moment_to_calibration
            else:
                assert (
                    new_moment_moment_to_calibration == corrections.moment_to_calibration
                ), f'Inconsistent decompositions with a moment {moment}'

        if other and decompositions:
            raise IncompatibleMomentError(f'Moment {moment} contains mixed operations')
        elif other:
            compensated += Moment(other)
            compensated_moment_to_calibration.append(characterization_index)
        elif decompositions:
            for operations in itertools.zip_longest(*decompositions, fillvalue=()):
                compensated += Moment(operations)
            assert new_moment_moment_to_calibration is not None  # Required for mypy
            compensated_moment_to_calibration += new_moment_moment_to_calibration

    return CircuitWithCalibration(compensated, compensated_moment_to_calibration)


@dataclasses.dataclass(frozen=True)
class FSimPhaseCorrections:
    """Operations that compensate for zeta, chi and gamma angles of an approximate FSimGate gate.

    Attributes:
        operations: Tuple of tuple of operations that describe the gate. The first index iterates
            over moments of the composed operation.
        moment_to_calibration: List of indices pointing to the calibration for each moment in the
            composed operation.
    """

    operations: Tuple[Tuple[Operation, ...], ...]
    moment_to_calibration: List[Optional[int]]

    @classmethod
    def from_characterization(
        cls,
        qubits: Tuple[Qid, Qid],
        gate_calibration: PhaseCalibratedFSimGate,
        parameters: PhasedFSimCharacterization,
        characterization_index: Optional[int],
    ) -> 'FSimPhaseCorrections':
        """Creates an operation that compensates for zeta, chi and gamma angles of the supplied
        gate and characterization.

        Args:
            qubits: Qubits that the gate should act on.
            gate_calibration: Original, imperfect gate that is supposed to run on the hardware
                together with phase information.
            parameters: The real parameters of the supplied gate.
            characterization_index: characterization index to use at each moment with gate.
        """
        operations = gate_calibration.with_zeta_chi_gamma_compensated(qubits, parameters)
        moment_to_calibration = [None, characterization_index, None]

        return cls(operations, moment_to_calibration)

    def as_circuit(self) -> Circuit:
        return Circuit(self.operations)


def run_floquet_characterization_for_moments(
    circuit: Circuit,
    sampler: Union[Engine, Sampler],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Extracts moments within a circuit to characterize and characterizes them against engine.

    The method calls prepare_floquet_characterization_for_moments to extract moments to characterize
    and run_calibrations to characterize them.

    Args:
        circuit: Circuit to characterize.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.QuantumEngineSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        gate_set: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
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
          - Circuit and its mapping from moments to indices into the list of characterized requests
            (the second returned value).
          - List of PhasedFSimCalibrationRequest for each characterized moment.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    circuit_calibration, requests = prepare_floquet_characterization_for_moments(
        circuit, options, gates_translator, merge_subsets=merge_subsets
    )
    results = run_calibrations(
        requests,
        sampler,
        processor_id,
        gate_set,
        max_layers_per_request=max_layers_per_request,
        progress_func=progress_func,
    )
    return circuit_calibration, results


def run_zeta_chi_gamma_compensation_for_moments(
    circuit: Circuit,
    sampler: Union[Engine, Sampler],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    options: FloquetPhasedFSimCalibrationOptions = (
        THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    ),
    gates_translator: Callable[
        [Gate], Optional[PhaseCalibratedFSimGate]
    ] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Compensates circuit against errors in zeta, chi and gamma angles by running calibrations on
    the engine.

    The method calls prepare_floquet_characterization_for_moments to extract moments to
    characterize, run_calibrations to characterize them and
    make_zeta_chi_gamma_compensation_for_moments to compensate the circuit with characterization
    data.

    Args:
        circuit: Circuit to characterize and calibrate.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.QuantumEngineSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        gate_set: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
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
          - Calibrated circuit together with its calibration metadata in CircuitWithCalibration
            object. The calibrated circuit has single-qubit Z gates added which compensates for the
            true gates imperfections.
            The moment to calibration mapping is updated for the new circuit so that successive
            calibrations could be applied.
          - List of characterizations results that were obtained in order to calibrate the circuit.
    """
    circuit_with_calibration, requests = prepare_floquet_characterization_for_moments(
        circuit, options, gates_translator, merge_subsets=merge_subsets
    )
    characterizations = run_calibrations(
        calibrations=requests,
        sampler=sampler,
        processor_id=processor_id,
        gate_set=gate_set,
        max_layers_per_request=max_layers_per_request,
        progress_func=progress_func,
    )
    calibrated_circuit = make_zeta_chi_gamma_compensation_for_moments(
        circuit_with_calibration, characterizations, gates_translator
    )
    return calibrated_circuit, characterizations
