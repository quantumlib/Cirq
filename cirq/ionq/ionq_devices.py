# Copyright 2021 The Cirq Developers
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

"""Devices for IonQ hardware."""

from cirq import devices, ops
from cirq.ionq import serializer
from cirq.ops import common_gates, parity_gates


class IonQAPIDevice(devices.Device):
    """A device that uses the gates exposed by the IonQ API."""

    def __init__(self, qubits: Sequence[devices.LineQubit], atol=1e-8):
        self.qubits = frozenset(qubits)
        self.atol = atol
        all_gates_valid = lambda x: True
        near_1_mod_2 = lambda x: return abs(x.exponent % 2 - 1) < self.atol
        self._is_api_gate_dispatch: Dict[Type['cirq.Gate'], Callable] = {
            common_gates.XPowGate: all_gates_valid,
            common_gates.YPowGate: all_gates_valid,
            common_gates.ZPowGate: all_gates_valid,
            parity_gates.XXPowGate: all_gates_valid,
            parity_gates.YYPowGate: all_gates_valid,
            parity_gates.ZZPowGate: all_gates_valid,
            common_gates.CNotPowGate: near_1_mod_2,
            common_gates.HPowGate: near_1_mod_2,
            common_gates.SwapPowGate: near_1_mod_2,
            common_gates.MeasurementGate: all_gates_valid,
        }

    def qubit_set(self) -> Optional[AbstractSet['cirq.Qid']]:
        return self.qubits

    def validate_operation(self, operation: 'cirq.Operation'):
        if operation.gate is None:
            raise ValueError(f'IonQAPIDevice does not support operations with no gates {operation}')
        if not self.has_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')

    def has_api_gate(self, operation):
        gate = operation.gate
        for gate_mro_type in type(gate).mro():
            if gate_mro_type in self._is_api_gate_dispatch:
                return self._is_api_gate_dispatch[gate_mro_type](operation)


    def decompose_operation(self, operation: 'cirq.Operation')  -> ops.OP_TREE:
        if has_api_gate(operation):
            return operation


class IonQNativeDevice(devices.Device):
    """A device that uses the native gates of the IonQ hardware."""

    pass
