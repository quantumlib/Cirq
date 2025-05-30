# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from math import exp
from typing import Sequence, TYPE_CHECKING, Optional, Dict
from dataclasses import dataclass

import cirq
from cirq.devices.noise_model import validate_all_measurements

if TYPE_CHECKING:
    from cirq_google.engine import calibration


@dataclass(frozen=True)
class PerQubitDepolarizingWithDampedReadoutNoiseModel(cirq.NoiseModel):
    """NoiseModel with T1 decay on gates and damping/bitflip on measurement.

    With this model, T1 decay is added after all non-measurement gates, then
    amplitude damping followed by bitflip error is added before all measurement
    gates. Idle qubits are unaffected by this model.

    As with the DepolarizingWithDampedReadoutNoiseModel, this model does not
    allow a moment to contain both measurement and non-measurement gates.
    """

    depol_probs: Optional[Dict[cirq.Qid, float]] = None
    bitflip_probs: Optional[Dict[cirq.Qid, float]] = None
    decay_probs: Optional[Dict[cirq.Qid, float]] = None

    def __post_init__(self):
        for probs, desc in [
            (self.depol_probs, "depolarization prob"),
            (self.bitflip_probs, "readout error prob"),
            (self.decay_probs, "readout decay prob"),
        ]:
            if probs:
                for qubit, prob in probs.items():
                    cirq.validate_probability(prob, f'{desc} of {qubit}')

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self.is_virtual_moment(moment):
            return moment
        moments = []
        if validate_all_measurements(moment):
            if self.decay_probs:
                moments.append(
                    cirq.Moment(
                        cirq.AmplitudeDampingChannel(self.decay_probs[q])(q) for q in system_qubits
                    )
                )
            if self.bitflip_probs:
                moments.append(
                    cirq.Moment(
                        cirq.BitFlipChannel(self.bitflip_probs[q])(q) for q in system_qubits
                    )
                )
            moments.append(moment)
            return moments
        else:
            moments.append(moment)
            if self.depol_probs:
                gated_qubits = [q for q in system_qubits if moment.operates_on_single_qubit(q)]
                if gated_qubits:
                    moments.append(
                        cirq.Moment(
                            cirq.DepolarizingChannel(self.depol_probs[q])(q) for q in gated_qubits
                        )
                    )
            return moments

    def _json_dict_(self) -> dict:
        return {
            'depol_probs': self.depol_probs,
            'bitflip_probs': self.bitflip_probs,
            'decay_probs': self.decay_probs,
        }

    @classmethod
    def _from_json_dict_(cls, depol_probs=None, bitflip_probs=None, decay_probs=None, **kwargs):
        return cls(depol_probs=depol_probs, bitflip_probs=bitflip_probs, decay_probs=decay_probs)


def simple_noise_from_calibration_metrics(
    calibration: calibration.Calibration,
    depol_noise: bool = False,
    damping_noise: bool = False,
    readout_decay_noise: bool = False,
    readout_error_noise: bool = False,
) -> cirq.NoiseModel:
    """Creates a reasonable PerQubitDepolarizingWithDampedReadoutNoiseModel
    using the provided calibration data.

    Args:
        calibration: a Calibration object (cirq/google/engine/calibration.py).
            This object can be retrieved from the engine by calling
            'get_latest_calibration()' or 'get_calibration()' using the ID of
            the target processor.
        depol_noise: Enables per-gate depolarization if True.
        damping_noise: Enables per-gate amplitude damping if True.
            Currently unimplemented.
        readout_decay_noise: Enables pre-readout amplitude damping if True.
        readout_error_noise: Enables pre-readout bitflip errors if True.

    Returns:
        A PerQubitDepolarizingWithDampedReadoutNoiseModel with error
            probabilities generated from the provided calibration data.
    Raises:
        NotImplementedError: If `damping_noise` is True, as this is not yet
            supported.
        ValueError: If none of the noises is set to True.
    """
    if not any([depol_noise, damping_noise, readout_decay_noise, readout_error_noise]):
        raise ValueError('At least one error type must be specified.')
    assert calibration is not None
    depol_probs: dict[cirq.Qid, float] = {}
    readout_decay_probs: dict[cirq.Qid, float] = {}
    readout_error_probs: dict[cirq.Qid, float] = {}

    if depol_noise:
        # In the single-qubit case, Pauli error and the depolarization fidelity
        # differ by a factor of 4/3.
        depol_probs = {
            calibration.key_to_qubit(qubit): calibration.value_to_float(pauli_err) * 4 / 3
            for qubit, pauli_err in calibration['single_qubit_rb_pauli_error_per_gate'].items()
        }
    if damping_noise:
        # TODO: implement per-gate amplitude damping noise.
        # Github issue: https://github.com/quantumlib/Cirq/issues/2807
        raise NotImplementedError('Gate damping is not yet supported.')

    if readout_decay_noise:
        # Copied from Sycamore readout duration in known_devices.py
        # TODO: replace with polling from DeviceSpecification.
        # Github issue: https://github.com/quantumlib/Cirq/issues/2832
        readout_micros = 1
        readout_decay_probs = {
            calibration.key_to_qubit(qubit): 1
            - exp(-1 * readout_micros / calibration.value_to_float(t1))
            for qubit, t1 in calibration['single_qubit_idle_t1_micros'].items()
        }
    if readout_error_noise:
        readout_error_probs = {
            calibration.key_to_qubit(qubit): calibration.value_to_float(err)
            for qubit, err in calibration['single_qubit_readout_separation_error'].items()
        }
    return PerQubitDepolarizingWithDampedReadoutNoiseModel(
        depol_probs=depol_probs, decay_probs=readout_decay_probs, bitflip_probs=readout_error_probs
    )
