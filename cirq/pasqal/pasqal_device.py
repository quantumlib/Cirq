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
from typing import FrozenSet, Sequence

import cirq
from cirq.ops import NamedQubit


@cirq.value.value_equality
class PasqalDevice(cirq.devices.Device):
    """A generic Pasqal device.

    The most general of Pasqal devices, enforcing only restrictions expected to
    be shared by all future devices. Serves as the parent class of all Pasqal
    devices, but can also be used on its own for hosting a nearly unconstrained
    device. When used as a circuit's device, the qubits have to be of the type
    cirq.NamedQubit and assumed to be all connected, the idea behind it being
    that after submission, all optimization and transpilation necessary for its
    execution on the specified device are handled internally by Pasqal.
    """

    def __init__(self, qubits: Sequence[cirq.ops.Qid]) -> None:
        """Initializes a device with some qubits.

        Args:
            qubits (NamedQubit): Qubits on the device, exclusively unrelated to
                a physical position.
        Raises:
            TypeError: if the wrong qubit type is provided.
        """

        for q in qubits:
            if not isinstance(q, self.supported_qubit_type):
                raise TypeError('Unsupported qubit type: {!r}. This device '
                                'supports qubit types: {}'.format(
                                    q, self.supported_qubit_type))

        if len(qubits) > self.maximum_qubit_number:
            raise ValueError('Too many qubits. {} accepts at most {} '
                             'qubits.'.format(type(self),
                                              self.maximum_qubit_number))

        self.qubits = qubits

    @property
    def supported_qubit_type(self):
        return (NamedQubit,)

    @property
    def maximum_qubit_number(self):
        return 100

    def qubit_set(self) -> FrozenSet[cirq.Qid]:
        return frozenset(self.qubits)

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def decompose_operation(self,
                            operation: cirq.ops.Operation) -> 'cirq.OP_TREE':

        decomposition = [operation]

        if not isinstance(operation,
                          (cirq.ops.GateOperation, cirq.ParallelGateOperation)):
            raise TypeError("{!r} is not a gate operation.".format(operation))

        # Try to decompose the operation into elementary device operations
        if not PasqalDevice.is_pasqal_device_op(operation):
            decomposition = cirq.protocols.decompose(
                operation, keep=PasqalDevice.is_pasqal_device_op)

        return decomposition

    @staticmethod
    def is_pasqal_device_op(op: cirq.ops.Operation) -> bool:

        if not isinstance(op, cirq.ops.Operation):
            raise ValueError('Got unknown operation:', op)
        return (len(op.qubits) > 1) or isinstance(
            op.gate, (cirq.ops.IdentityGate, cirq.ops.MeasurementGate,
                      cirq.ops.PhasedXPowGate, cirq.ops.XPowGate,
                      cirq.ops.YPowGate, cirq.ops.ZPowGate))

    def validate_operation(self, operation: cirq.ops.Operation):
        """
        Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate

        Raises:
            ValueError: If the operation is not valid
        """

        if not isinstance(operation,
                          (cirq.GateOperation, cirq.ParallelGateOperation)):
            raise ValueError("Unsupported operation")

        if not self.is_pasqal_device_op(operation):
            raise ValueError('{!r} is not a supported '
                             'gate'.format(operation.gate))

        for qub in operation.qubits:
            if not isinstance(qub, self.supported_qubit_type):
                raise ValueError('{} is not a valid qubit for gate {!r}. This '
                                 'device accepts gates on qubits of type: '
                                 '{}'.format(qub, operation.gate,
                                             self.supported_qubit_type))
            if qub not in self.qubit_set():
                raise ValueError('{} is not part of the device.'.format(qub))

        if isinstance(operation.gate, cirq.ops.MeasurementGate):
            if operation.gate.invert_mask != ():
                raise NotImplementedError("Measurements on Pasqal devices "
                                          "don't support invert_mask.")

    def validate_circuit(self, circuit: 'cirq.Circuit') -> None:
        """Raises an error if the given circuit is invalid on this device.

        A circuit is invalid if any of its moments are invalid or if there
        is a non-empty moment after a moment with a measurement.

        Args:
            circuit: The circuit to validate

        Raises:
            ValueError: If the given circuit can't be run on this device
        """
        super().validate_circuit(circuit)

        # Measurements must be in the last non-empty moment
        has_measurement_occurred = False
        for moment in circuit:
            if has_measurement_occurred:
                if len(moment.operations) > 0:
                    raise ValueError("Non-empty moment after measurement")
            for operation in moment.operations:
                if isinstance(operation.gate, cirq.ops.MeasurementGate):
                    has_measurement_occurred = True

    def __repr__(self):
        return 'pasqal.PasqalDevice(qubits={!r})'.format(sorted(self.qubits))

    def _value_equality_values_(self):
        return (self.qubits)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['qubits'])
