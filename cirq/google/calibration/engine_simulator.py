from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import random

from cirq.circuits import Circuit
from cirq.ops import (
    FSimGate,
    Gate,
    PhasedFSimGate,
    Qid
)
from cirq.sim import SimulatesSamples
from cirq.study import ParamResolver

from cirq.google.calibration.phased_fsim import (
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

        # TODO: Check if all values of mean and sigma are filled-in.

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

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        return self._simulator._run(circuit, param_resolver, repetitions)

    def get_calibrations(self,
                         requests: List[PhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        return NotImplemented
