from typing import Callable, List, Optional, Tuple, Union, cast

from cirq.circuits import Circuit
from cirq.ops import (
    FSimGate,
    Gate,
    GateOperation,
    MeasurementGate,
    Moment,
    Qid,
    SingleQubitGate
)
from cirq.google.calibration.engine_simulator import (
    PhasedFSimEngineSimulator
)
from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    sqrt_iswap_gates_translator
)
from cirq.google.engine import Engine
from cirq.google.serializable_gate_set import SerializableGateSet


def floquet_characterization_for_moment(
        moment: Moment,
        options: FloquetPhasedFSimCalibrationOptions,
        gate_set: SerializableGateSet,
        gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator,
        pairs_in_canonical_order: bool = False,
        pairs_sorted: bool = False
) -> Optional[FloquetPhasedFSimCalibrationRequest]:

    measurement = False
    single_qubit = False
    gate: Optional[FSimGate] = None
    pairs = []

    for op in moment:
        if not isinstance(op, GateOperation):
            raise IncompatibleMomentError(
                'Moment contains operation different than GateOperation')

        if isinstance(op.gate, MeasurementGate):
            measurement = True
        elif isinstance(op.gate, SingleQubitGate):
            single_qubit = True
        else:
            translated_gate = gates_translator(op.gate)
            if translated_gate is None:
                raise IncompatibleMomentError(f'Moment contains non-single qubit operation {op} '
                                              f'with gate that is not equal to cirq.ISWAP ** -0.5')
            elif gate is not None and gate != translated_gate:
                raise IncompatibleMomentError(f'Moment contains operations resolved to two '
                                              f'different gates {gate} and {translated_gate}')
            else:
                gate = translated_gate
            pair = cast(Tuple[Qid, Qid],
                        tuple(sorted(op.qubits) if pairs_in_canonical_order else op.qubits))
            pairs.append(pair)

    if gate is None:
        # Either empty, single-qubit or measurement moment.
        return None

    if gate is not None and (measurement or single_qubit):
        raise IncompatibleMomentError(f'Moment contains mixed two-qubit operations and '
                                      f'single-qubit operations or measurement operations.')

    return FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        gate_set=gate_set,
        pairs=tuple(sorted(pairs) if pairs_sorted else pairs),
        options=options
    )


def floquet_characterization_for_circuit(
        circuit: Circuit,
        options: FloquetPhasedFSimCalibrationOptions,
        gate_set: SerializableGateSet,
        gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator,
        merge_sub_sets: bool = True,
        initial: Optional[
            Tuple[List[FloquetPhasedFSimCalibrationRequest], List[Optional[int]]]] = None
) -> Tuple[List[FloquetPhasedFSimCalibrationRequest], List[Optional[int]]]:
    """
    Returns:
        Tuple of:
          - list of calibration requests,
          - list of indices of the generated calibration requests for each
            moment in the supplied circuit. If None occurs at certain position,
            it means that the related moment was not recognized for calibration.
    """

    def append_if_missing(calibration: FloquetPhasedFSimCalibrationRequest) -> int:
        if calibration.pairs not in pairs_map:
            index = len(calibrations)
            calibrations.append(calibration)
            pairs_map[calibration.pairs] = index
            return index
        else:
            return pairs_map[calibration.pairs]

    def merge_into_calibrations(calibration: FloquetPhasedFSimCalibrationRequest) -> int:
        calibration_pairs = set(calibration.pairs)
        for pairs, index in pairs_map.items():
            if calibration_pairs.issubset(pairs):
                return index
            elif calibration_pairs.issuperset(pairs):
                calibrations[index] = calibration
                return index

        index = len(calibrations)
        calibrations.append(calibration)
        pairs_map[calibration.pairs] = index
        return index

    if initial is None:
        calibrations = []
        moments_map = []
    else:
        calibrations, moments_map = initial

    pairs_map = {}

    for moment in circuit:
        calibration = floquet_characterization_for_moment(moment, options, gate_set, gates_translator,
                                                          pairs_in_canonical_order=True,
                                                          pairs_sorted=True)

        if calibration is not None:
            if merge_sub_sets:
                index = merge_into_calibrations(calibration)
            else:
                index = append_if_missing(calibration)
            moments_map.append(index)
        else:
            moments_map.append(None)

    return calibrations, moments_map


def run_characterizations(calibrations: List[PhasedFSimCalibrationRequest],
                          engine: Union[Engine, PhasedFSimEngineSimulator],
                          processor_id: str,
                          handler_name: str,
                          max_layers_per_request: int = 1,
                          progress_func: Optional[Callable[[int, int], None]] = None
                          ) -> List[PhasedFSimCalibrationResult]:
    if max_layers_per_request < 1:
        raise ValueError(f'Miaximum number of layers pere request must be at least 1, '
                         f'{max_layers_per_request} given')

    if not calibrations:
        return []

    gate_sets = [calibration.gate_set for calibration in calibrations]
    gate_set = gate_sets[0]
    if not all(gate_set == other for other in gate_sets):
        raise ValueError('All calibrations that run together must be defined for a single gate set')

    if isinstance(engine, Engine):
        results = []

        if progress_func:
            progress_func(len(results), len(calibrations))

        requests = [
            [calibration.to_calibration_layer(handler_name)
             for calibration in calibrations[offset:offset + max_layers_per_request]]
            for offset in range(0, len(calibrations), max_layers_per_request)
        ]

        for request in requests:
            job = engine.run_calibration(request,
                                         processor_id=processor_id,
                                         gate_set=gate_set)
            request_results = job.calibration_results()
            results += [calibration.parse_result(result)
                        for calibration, result in zip(calibrations, request_results)]
            if progress_func:
                progress_func(len(results), len(calibrations))

    elif isinstance(engine, PhasedFSimEngineSimulator):
        results = engine.get_calibrations(calibrations)
    else:
        raise ValueError(f'Unsupported engine type {type(engine)}')

    return results


def run_floquet_characterization_for_circuit(
        circuit: Circuit,
        engine: Union[Engine, PhasedFSimEngineSimulator],
        processor_id: str,
        handler_name: str,
        options: FloquetPhasedFSimCalibrationOptions,
        gate_set: SerializableGateSet,
        gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator,
        merge_sub_sets: bool = True,
        max_layers_per_request: int = 1,
        progress_func: Optional[Callable[[int, int], None]] = None
) -> List[Optional[PhasedFSimCalibrationResult]]:
    requests, mapping = floquet_characterization_for_circuit(
        circuit, options, gate_set, gates_translator, merge_sub_sets=merge_sub_sets)
    results = run_characterizations(requests, engine, processor_id, handler_name,
                                    max_layers_per_request=max_layers_per_request,
                                    progress_func=progress_func)
    return [results[index] if index is not None else None for index in mapping]
