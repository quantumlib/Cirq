# Copyright 2018 Google LLC
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
from typing import TYPE_CHECKING

from cirq.value import Duration


if TYPE_CHECKING:
    # pylint: disable=unused-import
    from cirq.circuits import Circuit
    from cirq.ops import Operation
    from cirq.schedules import Schedule, ScheduledOperation

# Note: circuit/schedule types specified by name to avoid circular references.


class Device(metaclass=abc.ABCMeta):
    """Hardware constraints for validating circuits and schedules."""

    @abc.abstractmethod
    def duration_of(self, operation: 'Operation') -> Duration:
        pass

    @abc.abstractmethod
    def validate_operation(self, operation: 'Operation') -> None:
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
            schedule: 'Schedule',
            scheduled_operation: 'ScheduledOperation'
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

    @abc.abstractmethod
    def validate_circuit(self, circuit: 'Circuit') -> None:
        """Raises an exception if a circuit is not valid.

        Args:
            circuit: The circuit to validate.

        Raises:
            ValueError: The circuit isn't valid for this device.
        """
        pass

    @abc.abstractmethod
    def validate_schedule(self, schedule: 'Schedule') -> None:
        """Raises an exception if a schedule is not valid.

        Args:
            schedule: The schedule to validate.

        Raises:
            ValueError: The schedule isn't valid for this device.
        """
        pass
