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

from cirq import ops
from cirq.devices import Device
from cirq.value import Duration, Timestamp


class ScheduledOperation:
    """An operation that happens over a specified time interval."""

    def __init__(self,
                 time: Timestamp,
                 duration: Duration,
                 operation: ops.Operation) -> None:
        """Initializes the scheduled operation.

        Args:
            time: When the operation starts.
            duration: How long the operation lasts.
            operation: The operation.
        """
        self.time = time
        self.duration = duration
        self.operation = operation

    @staticmethod
    def op_at_on(operation: ops.Operation,
                 time: Timestamp,
                 device: Device):
        """Creates a scheduled operation with a device-determined duration."""
        return ScheduledOperation(time,
                                  device.duration_of(operation),
                                  operation)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.time == other.time and
                self.operation == other.operation and
                self.duration == other.duration)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ScheduledOperation,
                     self.time,
                     self.operation,
                     self.duration))

    def __str__(self):
        return '{} during [{}, {})'.format(
            self.operation, self.time, self.time + self.duration)

    def __repr__(self):
        return 'ScheduledOperation({}, {}, {})'.format(
            repr(self.time), repr(self.duration), repr(self.operation))
