# Copyright 2018 The Cirq Developers
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

import abc
from typing import TYPE_CHECKING, Optional, AbstractSet

if TYPE_CHECKING:
    import cirq


class Device(metaclass=abc.ABCMeta):
    """Hardware constraints for validating circuits."""

    def qubit_set(self) -> Optional[AbstractSet['cirq.Qid']]:
        """Returns a set or frozenset of qubits on the device, if possible.

        Returns:
            If the device has a finite set of qubits, then a set or frozen set
            of all qubits on the device is returned.

            If the device has no well defined finite set of qubits (e.g.
            `cirq.UnconstrainedDevice` has this property), then `None` is
            returned.
        """

        # Compatibility hack to work with devices that were written before this
        # method was defined.
        for name in ['qubits', '_qubits']:
            if hasattr(self, name):
                val = getattr(self, name)
                if callable(val):
                    val = val()
                return frozenset(val)

        # Default to the qubits being unknown.
        return None

    def decompose_operation(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        """Returns a device-valid decomposition for the given operation.

        This method is used when adding operations into circuits with a device
        specified, to avoid spurious failures due to e.g. using a Hadamard gate
        that must be decomposed into native gates.
        """
        return operation

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        """Raises an exception if an operation is not valid.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """

    def validate_circuit(self, circuit: 'cirq.Circuit') -> None:
        """Raises an exception if a circuit is not valid.

        Args:
            circuit: The circuit to validate.

        Raises:
            ValueError: The circuit isn't valid for this device.
        """
        for moment in circuit:
            self.validate_moment(moment)

    def validate_moment(self, moment: 'cirq.Moment') -> None:
        """Raises an exception if a moment is not valid.

        Args:
            moment: The moment to validate.

        Raises:
            ValueError: The moment isn't valid for this device.
        """
        for operation in moment.operations:
            self.validate_operation(operation)

    def can_add_operation_into_moment(
        self, operation: 'cirq.Operation', moment: 'cirq.Moment'
    ) -> bool:
        """Determines if it's possible to add an operation into a moment.

        For example, on the XmonDevice two CZs shouldn't be placed in the same
        moment if they are on adjacent qubits.

        Args:
            operation: The operation being added.
            moment: The moment being transformed.

        Returns:
            Whether or not the moment will validate after adding the operation.
        """
        return not moment.operates_on(operation.qubits)
