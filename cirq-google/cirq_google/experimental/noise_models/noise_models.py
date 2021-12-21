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

from math import exp
from typing import Dict, Sequence, TYPE_CHECKING

import cirq
from cirq.devices.noise_model import validate_all_measurements

if TYPE_CHECKING:
    from cirq_google.engine import calibration


class PerQubitDepolarizingWithDampedReadoutNoiseModel(cirq.NoiseModel):
    """NoiseModel with T1 decay on gates and damping/bitflip on measurement.

    With this model, T1 decay is added after all non-measurement gates, then
    amplitude damping followed by bitflip error is added before all measurement
    gates. Idle qubits are unaffected by this model.

    As with the DepolarizingWithDampedReadoutNoiseModel, this model does not
    allow a moment to contain both measurement and non-measurement gates.
    """

    def __init__(
        self,
        depol_probs: Dict[cirq.Qid, float] = None,
        bitflip_probs: Dict[cirq.Qid, float] = None,
        decay_probs: Dict[cirq.Qid, float] = None,
    ):
        """A depolarizing noise model with damped readout error.

        All error modes are specified on a per-qubit basis. To omit a given
        error mode from the noise model, leave its dict blank when initializing
        this object.

        Args:
            depol_probs: Dict of depolarizing probabilities for each qubit.
            bitflip_probs: Dict of bit-flip probabilities during measurement.
            decay_probs: Dict of T1 decay probabilities during measurement.
                Bitflip noise is applied first, then amplitude decay.
        """
        for probs, desc in [
            (depol_probs, "depolarization prob"),
            (bitflip_probs, "readout error prob"),
            (decay_probs, "readout decay prob"),
        ]:
            if probs:
                for qubit, prob in probs.items():
                    cirq.validate_probability(prob, f'{desc} of {qubit}')
        self.depol_probs = depol_probs
        self.bitflip_probs = bitflip_probs
        self.decay_probs = decay_probs

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


def simple_noise_from_calibration_metrics(
    calibration: 'calibration.Calibration',
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
    depol_probs: Dict[cirq.Qid, float] = {}
    readout_decay_probs: Dict[cirq.Qid, float] = {}
    readout_error_probs: Dict[cirq.Qid, float] = {}

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
