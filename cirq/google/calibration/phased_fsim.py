from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import abc
import collections
import dataclasses
import re

from cirq.circuits import Circuit
from cirq.ops import Gate, Qid
from cirq.google.api import v2
from cirq.google.engine import CalibrationLayer, CalibrationResult, Engine
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

    def asdict(self) -> Dict[str, float]:
        """Converts parameters to a dictionary that maps angles name to values.
        """
        return dataclasses.asdict(self)

    def all_none(self) -> bool:
        """Checks if all the angles are None."""
        return self.theta is None and self.zeta is None and self.chi is None and self.gamma is None and self.phi is None

    def any_none(self) -> bool:
        """Checks if any of the angles is None."""
        return self.theta is None or self.zeta is None or self.chi is None or self.gamma is None or self.phi is None

    def parameters_for_qubits_swapped(self) -> 'PhasedFSimParameters':
        """Parameters for the gate with qubits swapped between each other.

        The angles theta, gamma and phi are kept unchanged. The angles zeta and chi are negated for the gate with
        swapped qubits.

        Returns:
            New instance with angles adjusted for swapped qubits.
        """
        return PhasedFSimParameters(
            theta=self.theta,
            zeta=-self.zeta if self.zeta is not None else None,
            chi=-self.chi if self.chi is not None else None,
            gamma=self.gamma,
            phi=self.phi
        )

    def merge_with(self, other: 'PhasedFSimParameters') -> 'PhasedFSimParameters':
        """Substitutes missing parameter with values from other.

        Args:
            other: Parameters to use for None values.

        Returns:
            New instance of PhasedFSimParameters with values from this instance if they are set or values from other
            when some parameter is None.
        """
        return PhasedFSimParameters(
            theta=other.theta if self.theta is None else self.theta,
            zeta=other.zeta if self.zeta is None else self.zeta,
            chi=other.chi if self.chi is None else self.chi,
            gamma=other.gamma if self.gamma is None else self.gamma,
            phi=other.phi if self.phi is None else self.phi,
        )

    def override_by(self, other: 'PhasedFSimParameters') -> 'PhasedFSimParameters':
        """Overrides other parameters that are not None.

        Args:
            other: Parameters to use for override.

        Returns:
            New instance of PhasedFSimParameters with values from other if set (values from other that are not None).
            Otherwise the current values are used.
        """
        return other.merge_with(self)


@json_serializable_dataclass(frozen=True)
class FloquetPhasedFSimCalibrationOptions:
    characterize_theta: bool
    characterize_zeta: bool
    characterize_chi: bool
    characterize_gamma: bool
    characterize_phi: bool


@json_serializable_dataclass(frozen=True)
class PhasedFSimCalibrationResult:
    # TODO: Fix json serialization (the default one doesn't work with tuples as dictionary keys).
    parameters: Dict[Tuple[Qid, Qid], PhasedFSimParameters]
    gate: Gate
    gate_set: SerializableGateSet

    def get_parameters(self, a: Qid, b: Qid) -> Optional['PhasedFSimParameters']:
        if (a, b) in self.parameters:
            return self.parameters[(a, b)]
        elif (b, a) in self.parameters:
            return self.parameters[(b, a)].parameters_for_qubits_swapped()
        else:
            return None


@json_serializable_dataclass(frozen=True)
class PhasedFSimCalibrationRequest(abc.ABC):
    gate: Gate  # Any gate which can be described by cirq.PhasedFSim
    gate_set: SerializableGateSet
    pairs: Tuple[Tuple[Qid, Qid]]

    @abc.abstractmethod
    def to_calibration_layer(self, handler_name: str) -> CalibrationLayer:
        pass

    @abc.abstractmethod
    def parse_result(self, result: CalibrationResult) -> PhasedFSimCalibrationResult:
        pass


@json_serializable_dataclass(frozen=True)
class FloquetPhasedFSimCalibrationResult(PhasedFSimCalibrationResult):
    options: FloquetPhasedFSimCalibrationOptions


@json_serializable_dataclass(frozen=True)
class FloquetPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    options: FloquetPhasedFSimCalibrationOptions

    def to_calibration_layer(self, handler_name: str) -> CalibrationLayer:
        circuit = Circuit([self.gate.on(*pair) for pair in self.pairs])
        return CalibrationLayer(
            calibration_type='floquet_phased_fsim_characterization',
            program=circuit,
            args={
                'est_theta': self.options.characterize_theta,
                'est_zeta': self.options.characterize_zeta,
                'est_chi': self.options.characterize_chi,
                'est_gamma': self.options.characterize_gamma,
                'est_phi': self.options.characterize_phi,
                'readout_corrections': True
            }
        )

    def parse_result(self, result: CalibrationResult) -> PhasedFSimCalibrationResult:
        decoded = collections.defaultdict(lambda: {})
        for keys, values in result.metrics['angles']:
            for key, value in zip(keys, values):
                match = re.match(r'(\d+)_(.+)', key)
                if not match:
                    raise ValueError(f'Unknown metric name {key}')
                index = int(match[1])
                name = match[2]
                decoded[index][name] = value

        parsed = {}
        for data in decoded.values():
            a = v2.qubit_from_proto_id(data['0'])
            b = v2.qubit_from_proto_id(data['1'])
            parsed[(a, b)] = PhasedFSimParameters(
                theta=data.get('theta_est', None),
                zeta=data.get('zeta_est', None),
                chi=data.get('chi_est', None),
                gamma=data.get('gamma_est', None),
                phi=data.get('phi_est', None)
            )

        return FloquetPhasedFSimCalibrationResult(
            parameters=parsed,
            gate=self.gate,
            gate_set=self.gate_set,
            options=self.options
        )


def run_calibrations(calibrations: List[PhasedFSimCalibrationRequest],
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
