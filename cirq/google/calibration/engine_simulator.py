from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import random

from cirq.circuits import (
    Circuit,
    PointOptimizer,
    PointOptimizationSummary
)
from cirq.ops import (
    FSimGate,
    Gate,
    MeasurementGate,
    Operation,
    PhasedFSimGate,
    Qid,
    QubitOrderOrList,
    SingleQubitGate,
    WaitGate
)
from cirq.sim import (
    Simulator,
    SimulatesSamples,
    SimulatesIntermediateStateVector,
    SparseSimulatorStep
)
from cirq.study import ParamResolver

from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationRequest,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimParameters,
    sqrt_iswap_gates_translator
)


SQRT_ISWAP_PARAMETERS = PhasedFSimParameters(
    theta=np.pi / 4,
    zeta=0.0,
    chi=0.0,
    gamma=0.0,
    phi=0.0
)


class PhasedFSimEngineSimulator(SimulatesSamples, SimulatesIntermediateStateVector):

    def __init__(
            self,
            simulator: Simulator, *,
            drift_generator: Callable[[Qid, Qid, FSimGate], PhasedFSimGate],
            gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator
    ) -> None:
        self._simulator = simulator
        self._drift_generator = drift_generator
        self._gates_translator = gates_translator
        self._drifted_gates: Dict[Tuple[Qid, Qid, FSimGate], PhasedFSimGate] = {}

    @staticmethod
    def create_with_ideal_sqrt_iswap(simulator: Optional[Simulator] = None
                                     ) -> 'PhasedFSimEngineSimulator':

        def sample_gate(_1: Qid, _2: Qid, gate: FSimGate) -> PhasedFSimGate:
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(gate.phi, 0.0), (
                f'Expected ISWAP ** -0.5 like gate, got {gate}'
            )
            return PhasedFSimGate(
                theta=np.pi / 4,
                zeta=0.0,
                chi=0.0,
                gamma=0.0,
                phi=0.0
            )

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator,
            drift_generator=sample_gate,
            gates_translator=sqrt_iswap_gates_translator
        )

    @staticmethod
    def create_with_random_gaussian_sqrt_iswap(
            mean: PhasedFSimParameters,
            *,
            simulator: Optional[Simulator] = None,
            sigma: PhasedFSimParameters = PhasedFSimParameters(
                theta=0.02,
                zeta=0.05,
                chi=0.05,
                gamma=0.05,
                phi=0.02
            ),
            rand: Optional[Union[int, random.Random]] = None
    ) -> 'PhasedFSimEngineSimulator':

        def sample_gate(_1: Qid, _2: Qid, gate: FSimGate) -> PhasedFSimGate:
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(gate.phi, 0.0), (
                f'Expected ISWAP ** -0.5 like gate, got {gate}'
            )

            def sample_value(gaussian_mean: float, gaussian_sigma: float) -> float:
                if gaussian_sigma:
                    return rand.gauss(gaussian_mean, gaussian_sigma)
                else:
                    return gaussian_mean

            return PhasedFSimGate(
                theta=sample_value(mean.theta, sigma.theta),
                zeta=sample_value(mean.zeta, sigma.zeta),
                chi=sample_value(mean.chi, sigma.chi),
                gamma=sample_value(mean.gamma, sigma.gamma),
                phi=sample_value(mean.phi, sigma.phi)
            )

        if mean.any_none():
            raise ValueError(f'All mean values must be provided, got {mean=}')

        if rand is not None:
            if isinstance(rand, int):
                rand = random.Random(rand)
            elif not isinstance(rand, random.Random):
                raise ValueError(
                    f'Provided rand argument {rand} is neither of type int or random.Random')
        else:
            rand = random.Random(rand)

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator,
            drift_generator=sample_gate,
            gates_translator=sqrt_iswap_gates_translator
        )

    @staticmethod
    def create_from_dictionary_sqrt_iswap(
            parameters: Dict[Tuple[Qid, Qid], Union[Dict[str, float], PhasedFSimParameters]],
            *,
            simulator: Optional[Simulator] = None,
            ideal_when_missing_gate: bool = False,
            ideal_when_missing_parameter: bool = False
    ) -> 'PhasedFSimEngineSimulator':

        def sample_gate(a: Qid, b: Qid, gate: FSimGate) -> PhasedFSimGate:
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(gate.phi, 0.0), (
                f'Expected ISWAP ** -0.5 like gate, got {gate}'
            )

            pair = (a, b) if a < b else (b, a)

            if pair in parameters:
                pair_parameters = parameters[pair]
                if not isinstance(pair_parameters, PhasedFSimParameters):
                    pair_parameters = PhasedFSimParameters(**pair_parameters)

                if pair_parameters.any_none():
                    if not ideal_when_missing_parameter:
                        raise ValueError(f'Missing parameter value for pair {pair}, '
                                         f'parameters={pair_parameters}')
                    pair_parameters = pair_parameters.merge_with(SQRT_ISWAP_PARAMETERS)
            elif ideal_when_missing_gate:
                pair_parameters = SQRT_ISWAP_PARAMETERS
            else:
                raise ValueError(f'Missing parameters for pair {pair}')

            return PhasedFSimGate(**pair_parameters.asdict())

        for a, b in parameters:
            if a > b:
                raise ValueError(
                    f'All qubit pairs must be given in canonical order where the first qubit is '
                    f'less than the second, got {a} > {b}')

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator,
            drift_generator=sample_gate,
            gates_translator=sqrt_iswap_gates_translator
        )

    @staticmethod
    def create_from_characterizations_sqrt_iswap(
            characterizations: Iterable[PhasedFSimCalibrationResult],
            *,
            simulator: Optional[Simulator] = None,
            ideal_when_missing_gate: bool = False,
            ideal_when_missing_parameter: bool = False
    ) -> 'PhasedFSimEngineSimulator':
        parameters = {}
        for characterization in characterizations:
            gate = characterization.gate
            if (not isinstance(gate, FSimGate) or
                    not np.isclose(gate.theta, np.pi / 4) or
                    not np.isclose(gate.phi, 0.0)):
                raise ValueError(f'Expected ISWAP ** -0.5 like gate, got {gate}')

            for (a, b), pair_parameters in characterization.parameters.items():
                if a > b:
                    a, b = b, a
                    pair_parameters = pair_parameters.for_qubits_swapped()
                if (a, b) in parameters:
                    # TODO: Add support for multi-moment simulation, where each moment can define
                    #  different parameters for a given qubits pair.
                    raise ValueError(f'Pair ({(a, b)}) appears in multiple moments, multi-moment '
                                     f'simulation is not supported.')
                parameters[(a, b)] = pair_parameters

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
            parameters,
            simulator=simulator,
            ideal_when_missing_gate=ideal_when_missing_gate,
            ideal_when_missing_parameter=ideal_when_missing_parameter
        )

    def final_state_vector(self, program: Circuit) -> np.array:
        result = self.simulate(program)
        return cast(SparseSimulatorStep, result).state_vector()

    def get_calibrations(self,
                         requests: List[PhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        results = []
        for request in requests:
            if isinstance(request, FloquetPhasedFSimCalibrationRequest):
                estimate_theta = request.options.estimate_theta
                estimate_zeta = request.options.estimate_zeta
                estimate_chi = request.options.estimate_chi
                estimate_gamma = request.options.estimate_gamma
                estimate_phi = request.options.estimate_phi
            else:
                raise ValueError(f'Unsupported calibration request {request}')

            translated_gate = self._gates_translator(request.gate)
            if translated_gate is None:
                raise ValueError(f'Calibration request contains unsupported gate {request.gate}')

            parameters = {}
            for a, b in request.pairs:
                drifted = self._get_or_create_gate(a, b, translated_gate)
                parameters[a, b] = PhasedFSimParameters(
                    theta=drifted.theta if estimate_theta else None,
                    zeta=drifted.zeta if estimate_zeta else None,
                    chi=drifted.chi if estimate_chi else None,
                    gamma=drifted.gamma if estimate_gamma else None,
                    phi=drifted.phi if estimate_phi else None,
                )

            results.append(PhasedFSimCalibrationResult(
                gate=request.gate,
                gate_set=request.gate_set,
                parameters=parameters
            ))

        return results

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        converted = self._convert_to_circuit_with_drift(circuit)
        return self._simulator._run(converted, param_resolver, repetitions)

    def _simulator_iterator(
        self,
        circuit: Circuit,
        param_resolver:ParamResolver,
        qubit_order: QubitOrderOrList,
        initial_state: np.ndarray,
    ) -> Iterator:
        converted = self._convert_to_circuit_with_drift(circuit)
        return self._simulator._simulator_iterator(
            converted, param_resolver, qubit_order, initial_state)

    def _convert_to_circuit_with_drift(self, circuit: Circuit) -> Circuit:
        copied = Circuit(circuit)
        converter = self._PhasedFSimConverter(self)
        converter.optimize_circuit(copied)
        return copied

    def _get_or_create_gate(self, a: Qid, b: Qid, gate: FSimGate) -> PhasedFSimGate:
        drifted_gate = self._drifted_gates.get((a, b, gate), None)
        if drifted_gate is None:
            drifted_gate = self._drift_generator(a, b, gate)
            self._drifted_gates[(a, b, gate)] = drifted_gate
            self._drifted_gates[(b, a, gate)] = PhasedFSimGate(
                theta=drifted_gate.theta,
                zeta=-drifted_gate.zeta,
                chi=-drifted_gate.chi,
                gamma=drifted_gate.gamma,
                phi=drifted_gate.phi
            )
        return drifted_gate

    class _PhasedFSimConverter(PointOptimizer):

        def __init__(self, outer: 'PhasedFSimEngineSimulator') -> None:
            super().__init__()
            self._outer = outer

        def optimization_at(
                self, circuit: Circuit, index: int, op: Operation
        ) -> Optional[PointOptimizationSummary]:

            if isinstance(op.gate, (MeasurementGate, SingleQubitGate, WaitGate)):
                new_op = op
            else:
                translated_gate = self._outer._gates_translator(op.gate)
                if translated_gate is None:
                    raise IncompatibleMomentError(f'Moment contains non-single qubit operation '
                                                  f'{op} with unsupported gate')
                a, b = op.qubits
                new_op = self._outer._get_or_create_gate(a, b, translated_gate).on(a, b)

            return PointOptimizationSummary(
                clear_span=1, clear_qubits=op.qubits, new_operations=new_op
            )
