from typing import List

from cirq.google.calibration.phased_fsim import (
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
)
from cirq.google.engine import Engine
from cirq.google.serializable_gate_set import SerializableGateSet


def run_characterizations(
    calibrations: List[PhasedFSimCalibrationRequest],
    engine: Engine,
    processor_id: str,
    gate_set: SerializableGateSet,
) -> List[PhasedFSimCalibrationResult]:
    """Runs calibration requests on the Engine.

    Args:
        calibrations: List of calibrations to perform described in a request object.
        engine: cirq.google.Engine object used for running the calibrations.
        processor_id: processor_id passed to engine.run_calibrations method.
        gate_set: Gate set to use for characterization request.

    Returns:
        List of PhasedFSimCalibrationResult for each requested calibration.
    """
    if not calibrations:
        return []

    requests = [calibration.to_calibration_layer() for calibration in calibrations]
    job = engine.run_calibration(requests, processor_id=processor_id, gate_set=gate_set)
    return [
        calibration.parse_result(result)
        for calibration, result in zip(calibrations, job.calibration_results())
    ]
