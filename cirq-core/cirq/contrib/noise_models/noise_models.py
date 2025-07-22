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

from typing import Sequence, TYPE_CHECKING

from cirq import circuits, devices, ops, value
from cirq.devices.noise_model import validate_all_measurements

if TYPE_CHECKING:
    import cirq


@value.value_equality()
class DepolarizingNoiseModel(devices.NoiseModel):
    """Applies depolarizing noise to each qubit individually at the end of
    every moment.

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, prepend: bool = False):
        """A depolarizing noise model

        Args:
            depol_prob: Depolarizing probability.
            prepend: If True, put noise before affected gates. Default: False.
        """
        value.validate_probability(depol_prob, 'depol prob')
        self._depol_prob = depol_prob
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self._prepend = prepend

    @property
    def depol_prob(self):
        """The depolarizing probability."""
        return self._depol_prob

    def _value_equality_values_(self):
        return self.depol_prob, self._prepend

    def __repr__(self) -> str:
        return (
            f'cirq.contrib.noise_models.DepolarizingNoiseModel('
            f'{self.depol_prob!r}, prepend={self._prepend!r})'
        )

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if validate_all_measurements(moment) or self.is_virtual_moment(moment):  # pragma: no cover
            return moment

        output = [
            moment,
            circuits.Moment(
                self.qubit_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits
            ),
        ]
        return output[::-1] if self._prepend else output

    def _json_dict_(self) -> dict[str, object]:
        return {'depol_prob': self.depol_prob, 'prepend': self._prepend}


@value.value_equality()
class ReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with probabilistic bit flips preceding measurement.

    This simulates readout error. Note that since noise is applied before the
    measurement moment, composing this model on top of another noise model will
    place the bit flips immediately before the measurement (regardless of the
    previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, bitflip_prob: float, prepend: bool = True):
        """A noise model with readout error.

        Args:
            bitflip_prob: Probability of a bit-flip during measurement.
            prepend: If True, put noise before affected gates. Default: True.
        """
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self._bitflip_prob = bitflip_prob
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)
        self._prepend = prepend

    @property
    def bitflip_prob(self):
        """The probability of a bit-flip during measurement."""
        return self._bitflip_prob

    def _value_equality_values_(self):
        return self.bitflip_prob, self._prepend

    def __repr__(self) -> str:
        p = self.bitflip_prob
        return f'cirq.contrib.noise_models.ReadoutNoiseModel({p!r}, prepend={self._prepend!r})'

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self.is_virtual_moment(moment):
            return moment
        if validate_all_measurements(moment):
            output = [
                circuits.Moment(
                    self.readout_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits
                ),
                moment,
            ]
            return output if self._prepend else output[::-1]
        return moment

    def _json_dict_(self) -> dict[str, object]:
        return {'bitflip_prob': self.bitflip_prob, 'prepend': self._prepend}


@value.value_equality()
class DampedReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with T1 decay preceding measurement.

    This simulates asymmetric readout error. Note that since noise is applied
    before the measurement moment, composing this model on top of another noise
    model will place the T1 decay immediately before the measurement
    (regardless of the previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, decay_prob: float, prepend: bool = True):
        """A depolarizing noise model with damped readout error.

        Args:
            decay_prob: Probability of T1 decay during measurement.
            prepend: If True, put noise before affected gates. Default: True.
        """
        value.validate_probability(decay_prob, 'decay_prob')
        self._decay_prob = decay_prob
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)
        self._prepend = prepend

    @property
    def decay_prob(self):
        """The probability of T1 decay during measurement."""
        return self._decay_prob

    def _value_equality_values_(self):
        return self.decay_prob, self._prepend

    def __repr__(self) -> str:
        return (
            f'cirq.contrib.noise_models.DampedReadoutNoiseModel('
            f'{self.decay_prob!r}, prepend={self._prepend!r})'
        )

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self.is_virtual_moment(moment):
            return moment
        if validate_all_measurements(moment):
            output = [
                circuits.Moment(
                    self.readout_decay_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits
                ),
                moment,
            ]
            return output if self._prepend else output[::-1]
        return moment

    def _json_dict_(self) -> dict[str, object]:
        return {'decay_prob': self.decay_prob, 'prepend': self._prepend}


@value.value_equality()
class DepolarizingWithReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingNoiseModel with probabilistic bit flips preceding
    measurement.
    This simulates readout error.
    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float):
        """A depolarizing noise model with readout error.
        Args:
            depol_prob: Depolarizing probability.
            bitflip_prob: Probability of a bit-flip during measurement.
        """
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self._depol_prob = depol_prob
        self._bitflip_prob = bitflip_prob
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)

    @property
    def depol_prob(self):
        """The depolarizing probability."""
        return self._depol_prob

    @property
    def bitflip_prob(self):
        """The probability of a bit-flip during measurement."""
        return self._bitflip_prob

    def _value_equality_values_(self):
        return self.depol_prob, self.bitflip_prob

    def __repr__(self) -> str:
        p = self.depol_prob
        b = self.bitflip_prob
        return f'cirq.contrib.noise_models.DepolarizingWithReadoutNoiseModel({p!r}, {b!r})'

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if validate_all_measurements(moment):
            return [circuits.Moment(self.readout_noise_gate(q) for q in system_qubits), moment]
        return [moment, circuits.Moment(self.qubit_noise_gate(q) for q in system_qubits)]

    def _json_dict_(self) -> dict[str, object]:
        return {'depol_prob': self.depol_prob, 'bitflip_prob': self.bitflip_prob}


@value.value_equality()
class DepolarizingWithDampedReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingWithReadoutNoiseModel with T1 decay preceding
    measurement.
    This simulates asymmetric readout error. The noise is structured
    so the T1 decay is applied, then the readout bitflip, then measurement.
    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float, decay_prob: float):
        """A depolarizing noise model with damped readout error.
        Args:
            depol_prob: Depolarizing probability.
            bitflip_prob: Probability of a bit-flip during measurement.
            decay_prob: Probability of T1 decay during measurement.
                Bitflip noise is applied first, then amplitude decay.
        """
        value.validate_probability(depol_prob, 'depol prob')
        value.validate_probability(bitflip_prob, 'bitflip prob')
        value.validate_probability(decay_prob, 'decay_prob')
        self._depol_prob = depol_prob
        self._bitflip_prob = bitflip_prob
        self._decay_prob = decay_prob
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)

    @property
    def depol_prob(self):
        """The depolarizing probability."""
        return self._depol_prob

    @property
    def bitflip_prob(self):
        """Probability of a bit-flip during measurement."""
        return self._bitflip_prob

    @property
    def decay_prob(self):
        """The probability of T1 decay during measurement."""
        return self._decay_prob

    def _value_equality_values_(self):
        return self.depol_prob, self.bitflip_prob, self.decay_prob

    def __repr__(self) -> str:
        return (
            'cirq.contrib.noise_models.DepolarizingWithDampedReadoutNoiseModel('
            f'{self.depol_prob!r}, {self.bitflip_prob!r}, {self.decay_prob!r})'
        )

    def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if validate_all_measurements(moment):
            return [
                circuits.Moment(self.readout_decay_gate(q) for q in system_qubits),
                circuits.Moment(self.readout_noise_gate(q) for q in system_qubits),
                moment,
            ]
        else:
            return [moment, circuits.Moment(self.qubit_noise_gate(q) for q in system_qubits)]

    def _json_dict_(self) -> dict[str, object]:
        return {
            'depol_prob': self.depol_prob,
            'bitflip_prob': self.bitflip_prob,
            'decay_prob': self.decay_prob,
        }
