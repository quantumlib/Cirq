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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import dataclasses
from itertools import zip_longest


from cirq.circuits import Circuit
from cirq.ops import (
    FSimGate,
    Gate,
    GateOperation,
    MeasurementGate,
    Moment,
    Operation,
    Qid,
    SingleQubitGate,
    rz,
)
from cirq.google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    try_convert_sqrt_iswap_to_fsim,
)
from cirq.google.engine import Engine
from cirq.google.serializable_gate_set import SerializableGateSet


@dataclasses.dataclass(frozen=True)
class CircuitWithCalibration:
    """Circuit with characterization data annotations.

    Attributes:
        circuit: Circuit instance.
        moment_to_calibration: Maps each moment within a circuit to an index of a characterization
            request or response. None means that there is no characterization data for that moment.
    """

    circuit: Circuit
    moment_to_calibration: List[Optional[int]]


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
    initial: Optional[Sequence[FloquetPhasedFSimCalibrationRequest]] = None,
) -> Tuple[CircuitWithCalibration, List[FloquetPhasedFSimCalibrationRequest]]:
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

    if initial is None:
        allocations: List[Optional[int]] = []
        calibrations: List[FloquetPhasedFSimCalibrationRequest] = []
        pairs_map: Dict[Tuple[Tuple[Qid, Qid], ...], int] = {}
    else:
        allocations = []
        calibrations = list(initial)
        pairs_map = {calibration.pairs: index for index, calibration in enumerate(calibrations)}

    for moment in circuit:
        calibration = make_floquet_request_for_moment(
            moment, options, gates_translator, canonicalize_pairs=True, sort_pairs=True
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


def _append_into_calibrations_if_missing(
    calibration: FloquetPhasedFSimCalibrationRequest,
    calibrations: List[FloquetPhasedFSimCalibrationRequest],
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
    calibration: FloquetPhasedFSimCalibrationRequest,
    calibrations: List[FloquetPhasedFSimCalibrationRequest],
    pairs_map: Dict[Tuple[Tuple[Qid, Qid], ...], int],
    options: FloquetPhasedFSimCalibrationOptions,
) -> int:
    """Merges a calibration into list of calibrations.

    If calibrations contains an item of which pairs could be expanded to include a new calibration
    pairs, without breaking a moment structure, then those two calibrations will be merged together
    and used as a calibration for both old and newly added calibration.
    If no calibration like that exists, the list will be expanded by calibration item.

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


def run_characterizations(
    calibrations: Sequence[PhasedFSimCalibrationRequest],
    engine: Union[Engine, PhasedFSimEngineSimulator],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> List[PhasedFSimCalibrationResult]:
    """Runs calibration requests on the Engine.

    Args:
        calibrations: List of calibrations to perform described in a request object.
        engine: cirq.google.Engine or cirq.google.PhasedFSimEngineSimulator object used for running
            the calibrations. When cirq.google.Engine then processor_id and gate_set arguments must
            be provided as well.
        processor_id: processor_id passed to engine.run_calibrations method. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
        gate_set: Gate set to use for characterization request. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
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

    if isinstance(engine, Engine):
        if processor_id is None:
            raise ValueError('processor_id must be provided when running on the engine')
        if gate_set is None:
            raise ValueError('gate_set must be provided when running on the engine')

        results = []

        requests = [
            [
                calibration.to_calibration_layer()
                for calibration in calibrations[offset : offset + max_layers_per_request]
            ]
            for offset in range(0, len(calibrations), max_layers_per_request)
        ]

        for request in requests:
            job = engine.run_calibration(request, processor_id=processor_id, gate_set=gate_set)
            request_results = job.calibration_results()
            results += [
                calibration.parse_result(result)
                for calibration, result in zip(calibrations, request_results)
            ]
            if progress_func:
                progress_func(len(results), len(calibrations))

    elif isinstance(engine, PhasedFSimEngineSimulator):
        results = engine.get_calibrations(calibrations)
    else:
        raise ValueError(f'Unsupported engine type {type(engine)}')

    return results


def zeta_chi_gamma_calibration_for_moments(
    circuit_with_calibration: CircuitWithCalibration,
    characterizations: List[PhasedFSimCalibrationResult],
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
) -> CircuitWithCalibration:
    """Compensates circuit against errors in zeta, chi and gamma angles.

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

            if isinstance(op.gate, (MeasurementGate, SingleQubitGate)):
                other.append(op)
                continue

            a, b = op.qubits
            translated_gate = gates_translator(op.gate)
            if translated_gate is None:
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
                (a, b), translated_gate, pair_parameters, characterization_index
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
            for operations in zip_longest(*decompositions, fillvalue=()):
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
        gate: FSimGate,
        parameters: PhasedFSimCharacterization,
        characterization_index: Optional[int],
    ) -> 'FSimPhaseCorrections':
        """Creates an operation that compensates for zeta, chi and gamma angles of the supplied
        gate and characterization.

        Args:
            qubits: Qubits that the gate should act on.
            gate: Original, imperfect gate that is supposed to run on the hardware.
            parameters: The real parameters of the supplied gate.
            characterization_index: characterization index to use at each moment with gate.
        """
        assert parameters.zeta is not None, "Zeta value must not be None"
        zeta = parameters.zeta

        assert parameters.gamma is not None, "Gamma value must not be None"
        gamma = parameters.gamma

        assert parameters.chi is not None, "Chi value must not be None"
        chi = parameters.chi

        a, b = qubits
        alpha = 0.5 * (zeta + chi)
        beta = 0.5 * (zeta - chi)

        operations = (
            (rz(0.5 * gamma - alpha).on(a), rz(0.5 * gamma + alpha).on(b)),
            (gate.on(a, b),),
            (rz(0.5 * gamma - beta).on(a), rz(0.5 * gamma + beta).on(b)),
        )

        moment_to_calibration = [None, characterization_index, None]

        return cls(operations, moment_to_calibration)

    def as_circuit(self) -> Circuit:
        return Circuit(self.operations)


def run_floquet_characterization_for_circuit(
    circuit: Circuit,
    engine: Union[Engine, PhasedFSimEngineSimulator],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    options: FloquetPhasedFSimCalibrationOptions = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Extracts moments within a circuit to characterize and characterizes them against engine.

    The method calls floquet_characterization_for_circuit to extract moments to characterize and
    run_characterizations to characterize them.

    Args:
        circuit: Circuit to characterize.
        engine: cirq.google.Engine or cirq.google.PhasedFSimEngineSimulator object used for running
            the calibrations. When cirq.google.Engine then processor_id and gate_set arguments must
            be provided as well.
        processor_id: processor_id passed to engine.run_calibrations method. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
        gate_set: Gate set to use for characterization request. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
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
    circuit_calibration, requests = make_floquet_request_for_circuit(
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
    return circuit_calibration, results


def run_zeta_chi_gamma_calibration_for_moments(
    circuit: Circuit,
    engine: Union[Engine, PhasedFSimEngineSimulator],
    processor_id: Optional[str] = None,
    gate_set: Optional[SerializableGateSet] = None,
    options: FloquetPhasedFSimCalibrationOptions = (
        THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    ),
    gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
    merge_subsets: bool = True,
    max_layers_per_request: int = 1,
    progress_func: Optional[Callable[[int, int], None]] = None,
) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Compensates circuit against errors in zeta, chi and gamma angles by running calibrations on
    the engine.

    Args:
        circuit: Circuit to characterize and calibrate.
        engine: cirq.google.Engine or cirq.google.PhasedFSimEngineSimulator object used for running
            the calibrations. When cirq.google.Engine then processor_id and gate_set arguments must
            be provided as well.
        processor_id: processor_id passed to engine.run_calibrations method. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
        gate_set: Gate set to use for characterization request. Can be None when
            cirq.google.PhasedFSimEngineSimulator is used as an engine.
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
    circuit_with_calibration, requests = make_floquet_request_for_circuit(
        circuit, options, gates_translator, merge_subsets=merge_subsets
    )
    characterizations = run_characterizations(
        calibrations=requests,
        engine=engine,
        processor_id=processor_id,
        gate_set=gate_set,
        max_layers_per_request=max_layers_per_request,
        progress_func=progress_func,
    )
    calibrated_circuit = zeta_chi_gamma_calibration_for_moments(
        circuit_with_calibration, characterizations, gates_translator
    )
    return calibrated_circuit, characterizations
