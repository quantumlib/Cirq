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
"""IQM builds quantum computers based on superconducting transmon technology.
For more information, see https://meetiqm.com.
"""

from typing import cast, Iterable, Set
import cirq
from cirq import devices, ops, protocols


class Adonis(devices.Device):
    """IQM's five-qubit superconducting device with pairwise connectivity.
    The qubits are connected thus:
    ```
            (0,1)
              |
    (1,0) - (1,1) - (1,2)
              |
            (2,1)
    ```
    where the lines denote which qubit pairs can be subject to the two-qubit
    gate.

    Each qubit can be rotated about the x, y, and z axes by an arbitrary angle,
    i.e. the device supports XPowGate, YPowGate, and ZPowGate. Adonis' two
    qubit-gate is CZPowGate (aka. CPHASE(𝛼)) again accepting an arbitrary
    exponent.

    The qubits can be measured simultaneously or separately during any moment.
    """

    @staticmethod
    def is_native_operation(op: ops.Operation):
        supported_gates = (ops.CZPowGate, ops.XPowGate, ops.YPowGate,
                           ops.ZPowGate, ops.MeasurementGate)
        return (isinstance(op, ops.TaggedOperation) or isinstance(
            op, ops.GateOperation)) and isinstance(op.gate, supported_gates)

    def __init__(self):
        """Instantiate the description of an Adonis device"""
        qubit_diagram = "-Q-\n" \
                        "QQQ\n" \
                        "-Q-\n"
        self.qubits = cirq.GridQubit.from_diagram(qubit_diagram)

    def decompose_operation(self, op: 'cirq.Operation') -> 'cirq.OP_TREE':
        super().decompose_operation(op)

        if Adonis.is_native_operation(op):
            return op

        return protocols.decompose(op,
                                   keep=Adonis.is_native_operation,
                                   on_stuck_raise=None)

    def validate_circuit(self, circuit: 'cirq.Circuit'):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        super().validate_operation(operation)

        if not isinstance(operation.untagged, cirq.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        if not Adonis.is_native_operation(operation):
            raise ValueError('Unsupported gate type: {!r}'.format(
                operation.gate))

        for qubit in operation.qubits:
            if qubit not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(qubit))

        if len(operation.qubits) == 2 and not isinstance(
                operation.gate, ops.MeasurementGate):
            first_qubit, second_qubit = operation.qubits
            if not cast(cirq.GridQubit, first_qubit).is_adjacent(second_qubit):
                raise ValueError(
                    'Unsupported qubit connectivity required for {!r}'.format(
                        operation))


def _verify_unique_measurement_keys(operations: Iterable['cirq.Operation']):
    seen_keys: Set[str] = set()
    for op in operations:
        if protocols.is_measurement(op):
            key = protocols.measurement_key(op)
            if key in seen_keys:
                raise ValueError('Measurement key {} repeated'.format(key))
            seen_keys.add(key)
