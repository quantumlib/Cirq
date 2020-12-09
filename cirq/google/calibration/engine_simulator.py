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

    def _run(self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
             ) -> Dict[str, np.ndarray]:
        return self._simulator._run(circuit, param_resolver, repetitions)

    def get_calibrations(self,
                         requests: List[PhasedFSimCalibrationRequest]
                         ) -> List[PhasedFSimCalibrationResult]:
        return NotImplemented
