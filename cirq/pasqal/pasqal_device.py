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
from typing import FrozenSet, Iterable, Callable, List

import cirq
from cirq.ops import NamedQubit


@cirq.value.value_equality
class PasqalDevice(cirq.devices.Device):
    """A generic Pasqal device."""

    def __init__(self, qubits: Iterable[NamedQubit]) -> None:
        """Initializes a device with some qubits.

        Args:
            qubits (NamedQubit): Qubits on the device, exclusively unrelated to
                a physical position.
        Raises:
            TypeError: if the wrong qubit type is provided.
        """

        for q in qubits:
            if not isinstance(q, NamedQubit):
                raise TypeError('Unsupported qubit type: {!r}'.format(q))

        self.qubits = qubits

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
        if not self.is_pasqal_device_op(operation):
            decomposition = PasqalConverter().pasqal_convert(
                operation, keep=self.is_pasqal_device_op)

        return decomposition

    def is_pasqal_device_op(self, op: cirq.ops.Operation) -> bool:

        if not isinstance(op, cirq.ops.Operation):
            raise ValueError('Got unknown operation:', op)

        valid_op = isinstance(
            op.gate, (cirq.ops.IdentityGate, cirq.ops.MeasurementGate,
                      cirq.ops.XPowGate, cirq.ops.YPowGate, cirq.ops.ZPowGate))

        if not valid_op:    # To prevent further checking if already passed
            if isinstance(op.gate, (cirq.ops.HPowGate,
                                    cirq.ops.CNotPowGate, cirq.ops.CZPowGate)):
                valid_op = (op.gate.exponent == 1)

        return valid_op

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
            if not isinstance(qub, NamedQubit):
                raise ValueError('{} is not a named qubit '
                                 'for gate {!r}'.format(qub, operation.gate))

    def __repr__(self):
        return 'pasqal.PasqalDevice(qubits={!r})'.format(sorted(self.qubits))

    def _value_equality_values_(self):
        return (self.qubits)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['qubits'])


class PasqalConverter(cirq.neutral_atoms.ConvertToNeutralAtomGates):
    """A gate converter for compatibility with Pasqal processors.

    Modified version of ConvertToNeutralAtomGates, in which the 'converter'
    method takes the 'keep' function as an input.
    """

    def pasqal_convert(self, op: cirq.ops.Operation,
                       keep: Callable[[cirq.ops.Operation],
                                      bool]) -> List[cirq.ops.Operation]:
        def on_stuck_raise(bad):
            return TypeError(
                "Don't know how to work with {!r}. "
                "It isn't a native PasqalDevice operation, "
                "a 1 or 2 qubit gate with a known unitary, "
                "or composite.".format(bad))

        return cirq.protocols.decompose(
            op,
            keep=keep,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise)
