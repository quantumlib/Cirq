from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

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
    SingleQubitGate,
    WaitGate
)
from cirq.sim import SimulatesSamples
from cirq.study import ParamResolver

from cirq.google.calibration.phased_fsim import (
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimParameters,
    sqrt_iswap_gates_translator
)


class PhasedFSimEngineSimulator(SimulatesSamples):

    def __init__(
            self,
            simulator: SimulatesSamples,
            drift_generator: Callable[[Qid, Qid, FSimGate], PhasedFSimGate],
            gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator
    ) -> None:
        self._simulator = simulator
        self._drift_generator = drift_generator
        self._gates_translator = gates_translator
        self._drifted_gates: Dict[Tuple[Qid, Qid, FSimGate], PhasedFSimGate] = {}

    @staticmethod
    def create_with_ideal_sqrt_iswap(simulator: SimulatesSamples) -> 'PhasedFSimEngineSimulator':

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

        return PhasedFSimEngineSimulator(
            simulator,
            drift_generator=sample_gate,
            gates_translator=sqrt_iswap_gates_translator
        )

    @staticmethod
    def create_with_random_gaussian_sqrt_iswap(
            simulator: SimulatesSamples,
            mean: PhasedFSimParameters,
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
            return PhasedFSimGate(
                theta=rand.gauss(mean.theta, sigma.theta),
                zeta=rand.gauss(mean.zeta, sigma.zeta),
                chi=rand.gauss(mean.chi, sigma.chi),
                gamma=rand.gauss(mean.gamma, sigma.gamma),
                phi=rand.gauss(mean.phi, sigma.phi)
            )

        if mean.any_none():
            raise ValueError(f'All mean values must be provided, got {mean=}')

        if sigma.any_none():
            raise ValueError(f'All sigma values must be provided, got {sigma=}')

        if rand is not None:
            if isinstance(rand, int):
                rand = random.Random(rand)
            elif not isinstance(rand, random.Random):
                raise ValueError(
                    f'Provided rand argument {rand} is neither of type int or random.Random')
        else:
            rand = random.Random(rand)

        return PhasedFSimEngineSimulator(
            simulator,
            drift_generator=sample_gate,
            gates_translator=sqrt_iswap_gates_translator
        )

    @staticmethod
    def create_from_dictionary_sqrt_iswap(
            simulator: SimulatesSamples,
            parameters: Dict[str, Callable[[Qid, Qid], float]],
            ideal_when_missing_gate: bool = False,
            ideal_when_missing_parameter: bool = False
    ) -> 'PhasedFSimEngineSimulator':
        return NotImplemented

    @staticmethod
    def create_from_characterizations_sqrt_iswap(
            simulator: SimulatesSamples,
            characterizations: Iterable[PhasedFSimCalibrationResult],
            ideal_when_missing_gate: bool = False,
            ideal_when_missing_parameter: bool = False
    ) -> 'PhasedFSimEngineSimulator':
        return NotImplemented

    def get_calibrations(self,
                         requests: List[PhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        return NotImplemented

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        converted = self._convert_to_circuit_with_drift(circuit)
        return self._simulator._run(converted, param_resolver, repetitions)

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
                    raise IncompatibleMomentError(
                        f'Moment contains non-single qubit operation {op} with gate that is not '
                        f'equal to cirq.ISWAP ** -0.5'
                    )
                a, b = op.qubits
                new_op = self._outer._get_or_create_gate(a, b, translated_gate)

            return PointOptimizationSummary(
                clear_span=1, clear_qubits=op.qubits, new_operations=new_op
            )
