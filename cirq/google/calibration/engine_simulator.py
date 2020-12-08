from typing import Dict, List

import numpy as np

from cirq.circuits import Circuit
from cirq.sim import SimulatesSamples
from cirq.study import ParamResolver

from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult
)


class PhasedFSimEngineSimulator(SimulatesSamples):

    def __init__(self, simulator: SimulatesSamples) -> None:
        self._simulator = simulator

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        return self._simulator._run(circuit, param_resolver, repetitions)

    def run_calibrations(self,
                         requests: List[FloquetPhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        return NotImplemented
