from typing import List

from cirq.google.calibration.phased_fsim import (
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult
)
from cirq.google.engine import Engine


def run_characterizations(calibrations: List[PhasedFSimCalibrationRequest],
                          engine: Engine,
                          processor_id: str,
                          handler_name: str
                          ) -> List[PhasedFSimCalibrationResult]:
    if not calibrations:
        return []

    gate_sets = [calibration.gate_set for calibration in calibrations]
    gate_set = gate_sets[0]
    if not all(gate_set == other for other in gate_sets):
        raise ValueError('All calibrations that run together must be defined for a shared gate set')

    requests = [calibration.to_calibration_layer(handler_name) for calibration in calibrations]
    job = engine.run_calibration(requests,
                                 processor_id=processor_id,
                                 gate_set=gate_set)
    return [calibration.parse_result(result)
            for calibration, result in zip(calibrations, job.calibration_results())]
