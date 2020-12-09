from typing import Callable, Dict, List, Optional

import numpy as np

from cirq.circuits import Circuit
from cirq.ops import (
    FSimGate,
    Gate,
    PhasedFSimGate
)
from cirq.sim import SimulatesSamples
from cirq.study import ParamResolver

from cirq.google.calibration.phased_fsim import (
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    sqrt_iswap_gates_translator
)


class PhasedFSimEngineSimulator(SimulatesSamples):

    def __init__(
            self,
            simulator: SimulatesSamples,
            drift_generator: Callable[[FSimGate], PhasedFSimGate],
            gates_translator: Callable[[Gate], Optional[FSimGate]] = sqrt_iswap_gates_translator
    ) -> None:
        self._simulator = simulator
        self._drift_generator = drift_generator
        self._gates_translator = gates_translator

    @staticmethod
    def create_with_ideal_sqrt_iswap() -> 'PhasedFSimEngineSimulator':
        return NotImplemented

    @staticmethod
    def create_with_random_gaussian(
            reference: PhasedFSimGate,
            sigma_theta: float = 0.02,
            sigma_zeta: float = 0.05,
            sigma_chi: float = 0.05,
            sigma_gamma: float = 0.05,
            sigma_phi: float = 0.02
    ) -> 'PhasedFSimEngineSimulator':
        return NotImplemented

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        return self._simulator._run(circuit, param_resolver, repetitions)

    def get_calibrations(self,
                         requests: List[PhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        return NotImplemented
