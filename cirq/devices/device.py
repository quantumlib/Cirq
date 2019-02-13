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

from typing import TYPE_CHECKING

import abc

from cirq.value import Duration


if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq

# Note: circuit/schedule types specified by name to avoid circular references.


class Device(metaclass=abc.ABCMeta):
    """Hardware constraints for validating circuits and schedules."""

    def decompose_operation(self, operation: 'cirq.Operation'
                            ) -> 'cirq.OP_TREE':
        """Returns a device-valid decomposition for the given operation.

        This method is used when adding operations into circuits with a device
        specified, to avoid spurious failures due to e.g. using a Hadamard gate
        that must be decomposed into native gates.
        """
        return operation

    @abc.abstractmethod
    def duration_of(self, operation: 'cirq.Operation') -> Duration:
        pass

    @abc.abstractmethod
    def validate_operation(self, operation: 'cirq.Operation') -> None:
        """Raises an exception if an operation is not valid.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """
        pass

    @abc.abstractmethod
    def validate_scheduled_operation(
            self,
            schedule: 'cirq.Schedule',
            scheduled_operation: 'cirq.ScheduledOperation'
    ) -> None:
        """Raises an exception if the scheduled operation is not valid.

        Args:
            schedule: The schedule to validate against.
            scheduled_operation: The scheduled operation to validate.

        Raises:
            ValueError: If the scheduled operation is not valid for the
                schedule.
        """
        pass

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

    def can_add_operation_into_moment(self,
                                      operation: 'cirq.Operation',
                                      moment: 'cirq.Moment') -> bool:
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

    @abc.abstractmethod
    def validate_schedule(self, schedule: 'cirq.Schedule') -> None:
        """Raises an exception if a schedule is not valid.

        Args:
            schedule: The schedule to validate.

        Raises:
            ValueError: The schedule isn't valid for this device.
        """
        pass
