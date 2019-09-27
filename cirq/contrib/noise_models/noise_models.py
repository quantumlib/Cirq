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

from typing import Sequence, TYPE_CHECKING

from cirq import devices, value, ops, protocols

if TYPE_CHECKING:
    import cirq


def _moment_is_measurements(moment: 'cirq.Moment') -> bool:
    """Whether the moment is nothing but measurement gates.

    If a moment is a mixture of measurement and non-measurement gates
    this will throw a ValueError.
    """
    is_measurements = None
    for gate in moment:
        if protocols.is_measurement(gate):
            if is_measurements is None:
                is_measurements = True
            elif is_measurements:
                # Check that *all* are measurements
                pass
            else:
                raise ValueError("Moment must be all measurements "
                                 "or all operations.")
        else:
            if is_measurements is None:
                is_measurements = False
            elif not is_measurements:
                # Check that all are *not* measurements
                pass
            else:
                raise ValueError("Moment must be all measurements "
                                 "or all operations.")
    return is_measurements


class DepolarizingNoiseModel(devices.NoiseModel):
    """Applies depolarizing noise to each qubit individually at the end of
    every moment.

    If a circuit contains measurements, they be in moments that don't
    also contain gates.

    Args:
        depol_prob: Depolarizing probability.
    """

    def __init__(self, depol_prob: float):
        value.validate_probability(depol_prob, 'depol prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if _moment_is_measurements(moment):
            return moment

        return [moment,
                ops.Moment(self.qubit_noise_gate(q) for q in system_qubits)]


class DepolarizingWithReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingNoiseModel with probabilistic bit flips preceding
    measurement.

    This simulates readout error.

    If a circuit contains measurements, they be in moments that don't
    also contain gates.

    Args:
        depol_prob: Depolarizing probability.
        bitflip_prob: Probability of a bit-flip during measurement.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float):
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if _moment_is_measurements(moment):
            return [
                ops.Moment(self.readout_noise_gate(q) for q in system_qubits),
                moment,
            ]
        return [
            moment,
            ops.Moment(self.qubit_noise_gate(q) for q in system_qubits),
        ]


class DepolarizingWithDampedReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingWithReadoutNoiseModel with T1 decay preceding
    measurement.

    This simulates asymmetric readout error. The noise is structured
    so the T1 decay is applied, then the readout bitflip, then measurement.

    If a circuit contains measurements, they be in moments that don't
    also contain gates.

    Args:
        depol_prob: Depolarizing probability.
        bitflip_prob: Probability of a bit-flip during measurement.
        decay_prob: Probability of T1 decay during measurement.
    """

    def __init__(self, depol_prob: float,
                 bitflip_prob: float,
                 decay_prob: float, ):
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        value.validate_probability(decay_prob, 'bitflip prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if _moment_is_measurements(moment):
            return [
                ops.Moment(self.readout_decay_gate(q) for q in system_qubits),
                ops.Moment(self.readout_noise_gate(q) for q in system_qubits),
                moment
            ]
        else:
            return [
                moment,
                ops.Moment(self.qubit_noise_gate(q) for q in system_qubits)
            ]
