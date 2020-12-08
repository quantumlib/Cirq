from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import abc
from cirq.circuits import Circuit
from cirq.ops import Gate, Moment, Qid
from cirq.google.engine import CalibrationLayer, Engine
from cirq.google.serializable_gate_set import SerializableGateSet

if TYPE_CHECKING:
    # Workaround for mypy custom dataclasses
    from dataclasses import dataclass as json_serializable_dataclass
else:
    from cirq.protocols import json_serializable_dataclass


@json_serializable_dataclass(frozen=True)
class PhasedFSimParameters:
    theta: Optional[float] = None
    zeta: Optional[float] = None
    chi: Optional[float] = None
    gamma: Optional[float] = None
    phi: Optional[float] = None


@json_serializable_dataclass
class FloquetPhasedFSimCalibrationOptions:
    estimate_theta: bool
    estimate_zeta: bool
    estimate_chi: bool
    estimate_gamma: bool
    estimate_phi: bool


class PhasedFSimCalibrationRequest(abc.ABC):
    gate: Gate  # Any gate which can be described by cirq.PhasedFSim
    gate_set: SerializableGateSet
    qubits: Tuple[Tuple[Qid, Qid]]

    @abc.abstractmethod
    def to_calibration_layer(self) -> CalibrationLayer:
        pass


class PhasedFSimCalibrationResult(abc.ABC):
    parameters: Dict[Tuple[Qid, Qid], PhasedFSimParameters]
    gate: Gate
    gate_set: SerializableGateSet


@json_serializable_dataclass
class FloquetPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    options: FloquetPhasedFSimCalibrationOptions

    def to_calibration_layer(self) -> CalibrationLayer:
        return NotImplemented


@json_serializable_dataclass
class FloquetPhasedFSimCalibrationResult(PhasedFSimCalibrationResult):
    options: FloquetPhasedFSimCalibrationOptions


def run_calibrations(engine: Engine,
                     calibrations: List[PhasedFSimCalibrationRequest]
                     ) -> List[PhasedFSimCalibrationResult]:
    return NotImplemented


def default_fsim_floquet_options(
        gate: Gate) -> Optional[FloquetPhasedFSimCalibrationOptions]:
    return NotImplemented


def floquet_calibration_for_moment(
        moment: Moment,
        options_generator: Callable[
            [Gate], Optional[FloquetPhasedFSimCalibrationOptions]
        ] = default_fsim_floquet_options
) -> FloquetPhasedFSimCalibrationRequest:
    return NotImplemented


def floquet_calibration_for_circuit(
        circuit: Circuit,
        options_generator: Callable[
            [Gate], FloquetPhasedFSimCalibrationOptions
        ] = default_fsim_floquet_options,
        merge_sub_sets: bool = True
) -> Tuple[List[FloquetPhasedFSimCalibrationRequest], List[Optional[int]]]:
    """
    Returns:
        Tuple of:
          - list of calibration requests,
          - list of indices of the generated calibration requests for each
            moment in the supplied circuit. If None occurs at certain position,
            it means that the related moment was not recognized for calibration.
    """
    return NotImplemented


def run_floquet_calibration_for_circuit(
        engine: Engine,
        circuit: Circuit,
        options_generator: Callable[
            [Gate], FloquetPhasedFSimCalibrationOptions
        ] = default_fsim_floquet_options,
        merge_sub_sets: bool = True
) -> List[FloquetPhasedFSimCalibrationResult]:
    return NotImplemented
