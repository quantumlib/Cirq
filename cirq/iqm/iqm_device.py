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
"""IQM devices https://iqm.fi/devices"""  # TODO: PQC-5

from typing import cast
import cirq
from cirq import devices, ops


class Adonis(devices.Device):
    """IQM's five-qubit superconducting device with pairwise connectivity.
    Details: https://iqm.fi/devices
    """
    # TODO: PQC-5

    QUBIT_DIAGRAM = "-Q-\n" \
                    "QQQ\n" \
                    "-Q-\n"

    SUPPORTED_GATES = (
        ops.CZPowGate,
        ops.ISwapPowGate,
        ops.XPowGate,
        ops.YPowGate,
        ops.MeasurementGate
    )

    def __init__(self):
        """Instantiate the description of an Adonis device"""
        self.qubits = cirq.GridQubit.from_diagram(self.QUBIT_DIAGRAM)

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        super().validate_operation(operation)

        if not isinstance(operation, cirq.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        if not isinstance(operation.gate, Adonis.SUPPORTED_GATES):
            raise ValueError('Unsupported gate type: {!r}'.format(operation.gate))

        # TODO check that operation qubits are on device

        if len(operation.qubits) == 2 and not isinstance(operation.gate, ops.MeasurementGate):
            first_qubit, second_qubit = operation.qubits
            if not cast(cirq.GridQubit, first_qubit).is_adjacent(second_qubit):
                raise ValueError('Unsupported qubit connectivity required for {!r}'.format(operation))
