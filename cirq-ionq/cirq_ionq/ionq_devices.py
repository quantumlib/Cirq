# Copyright 2021 The Cirq Developers
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
"""Devices for IonQ hardware."""

from typing import Sequence
from typing import Union

import cirq
from cirq_ionq import ionq_gateset


class IonQAPIDevice(cirq.Device):
    """A device that uses the QIS gates exposed by the IonQ API.

    When using this device in constructing a circuit, it will convert one and two qubit gates
    that are not supported by the API into those supported by the API if they have a unitary
    matrix (support the unitary protocol).

    Note that this device does not do any compression of the resulting circuit, i.e. it may
    result in a series of single qubit gates that could be executed using far fewer elements.

    The gates supported by the API are
        * `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`
        * `cirq.XXPowGate`, `cirq.YYPowGate`, `cirq.ZZPowGate`
        * `cirq.CNOT`, `cirq.H`, `cirq.SWAP`
        * `cirq.MeasurementGate`
    """

    def __init__(self, qubits: Union[Sequence[cirq.LineQubit], int], atol=1e-8):
        """Construct the device.

        Args:
            qubits: The qubits upon which this device acts or the number of qubits. If the number
                of qubits, then the qubits will be `cirq.LineQubit`s from 0 to this number minus
                one.
            atol: The absolute tolerance used for gate calculations and decompositions.
        """
        if isinstance(qubits, int):
            self.qubits = frozenset(cirq.LineQubit.range(qubits))
        else:
            self.qubits = frozenset(qubits)
        self.atol = atol
        self.gateset = ionq_gateset.IonQTargetGateset()
        self._metadata = cirq.DeviceMetadata(
            self.qubits, [(a, b) for a in self.qubits for b in self.qubits if a != b]
        )

    @property
    def metadata(self) -> cirq.DeviceMetadata:
        return self._metadata

    def validate_operation(self, operation: cirq.Operation):
        if operation.gate is None:
            raise ValueError(
                f'IonQAPIDevice does not support operations with no gates {operation}.'
            )
        if not self.is_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')
        if not set(operation.qubits).intersection(self.metadata.qubit_set):
            raise ValueError(f'Operation with qubits not on the device. Qubits: {operation.qubits}')

    def is_api_gate(self, operation: cirq.Operation) -> bool:
        return operation in self.gateset
