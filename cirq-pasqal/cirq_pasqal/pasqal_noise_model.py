# Copyright 2020 The Cirq Developers
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
from typing import List, Dict, Sequence, Any

import cirq
import cirq_pasqal


class PasqalNoiseModel(cirq.devices.NoiseModel):
    """A noise model for Pasqal neutral atom device"""

    def __init__(self, device: cirq.devices.Device):
        self.noise_op_dict = self.get_default_noise_dict()
        if not isinstance(device, cirq_pasqal.PasqalDevice):
            raise TypeError(
                "The noise model varies between Pasqal's devices. "
                "Please specify the one you intend to execute the "
                "circuit on."
            )
        self.device = device

    def get_default_noise_dict(self) -> Dict[str, Any]:
        """Returns the current noise parameters"""
        default_noise_dict = {
            str(cirq.ops.YPowGate()): cirq.ops.depolarize(1e-2),
            str(cirq.ops.ZPowGate()): cirq.ops.depolarize(1e-2),
            str(cirq.ops.XPowGate()): cirq.ops.depolarize(1e-2),
            str(cirq.ops.PhasedXPowGate(phase_exponent=0)): cirq.ops.depolarize(1e-2),
            str(cirq.ops.HPowGate(exponent=1)): cirq.ops.depolarize(1e-2),
            str(cirq.ops.CNotPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
            str(cirq.ops.CZPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
            str(cirq.ops.CCXPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
            str(cirq.ops.CCZPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
        }
        return default_noise_dict

    def noisy_moment(
        self, moment: cirq.ops.Moment, system_qubits: Sequence[cirq.ops.Qid]
    ) -> List[cirq.ops.Operation]:
        """Returns a list of noisy moments.
        The model includes
        - Depolarizing noise with gate-dependent strength
        Args:
            moment: ideal moment
            system_qubits: List of qubits
        Returns:
            List of ideal and noisy moments
        """
        noise_list = []
        for op in moment:
            op_str = self.get_op_string(op)
            noise_op = self.noise_op_dict.get(op_str, cirq.ops.depolarize(5e-2))
            for qubit in op.qubits:
                noise_list.append(noise_op.on(qubit))
        return list(moment) + noise_list

    def get_op_string(self, cirq_op: cirq.ops.Operation) -> str:
        """Find the string representation for a given operation.

        Args:
            cirq_op: A cirq operation.

        Returns:
            String representing the gate operations.

        Raises:
            ValueError: If the operations gate is not supported.
        """
        if not self.device.is_pasqal_device_op(cirq_op) or isinstance(
            cirq_op.gate, cirq.ops.MeasurementGate
        ):
            raise ValueError('Got unknown operation:', cirq_op)

        return str(cirq_op.gate)
