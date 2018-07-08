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

from typing import Iterable, List, TYPE_CHECKING, Union, cast

from sortedcontainers import SortedListWithKey

from cirq.circuits import Circuit
from cirq.devices import Device
from cirq.ops import QubitId
from cirq.schedules.scheduled_operation import ScheduledOperation
from cirq.value import Duration, Timestamp

if TYPE_CHECKING:
    from typing import Optional  # pylint: disable=unused-import

    from cirq.ops import Operation  # pylint: disable=unused-import


class Schedule:
    """A quantum program with operations happening at specific times.

    Supports schedule[time] point lookups and
        schedule[inclusive_start_time:exclusive_end_time] slice lookups.


    Attributes:
        device: The hardware this will schedule on.
        scheduled_operations: A SortedListWithKey containing the
            ScheduledOperations for this schedule. The key is the start time
            of the ScheduledOperation.
    """

    def __init__(self,
            device: Device,
            scheduled_operations: Iterable[ScheduledOperation] = ()
            ) -> None:
        """Initializes a new schedule.

        Args:
            device: The hardware this schedule will run on.
            scheduled_operations: Initial list of operations to apply. These
                will be moved into a sorted list, with a key equal to each
                operation's start time.
        """
        self.device = device
        self.scheduled_operations = SortedListWithKey(scheduled_operations,
                                                      key=lambda e: e.time)
        self._max_duration = max(
            [e.duration for e in self.scheduled_operations] or [Duration()])

    def __eq__(self, other):
        if not isinstance(other, Schedule):
            return NotImplemented
        return self.scheduled_operations == other.scheduled_operations

    def __ne__(self, other):
        return not self == other

    __hash__ = None  # type: ignore

    def query(self, *,  # Forces keyword args.
              time: Timestamp,
              duration: Duration = Duration(),
              qubits: Iterable[QubitId] = None,
              include_query_end_time=False,
              include_op_end_times=False) -> List[ScheduledOperation]:
        """Finds operations by time and qubit.

        Args:
            time: Operations must end after this time to be returned.
            duration: Operations must start by time+duration to be
                returned.
            qubits: If specified, only operations touching one of the included
                qubits will be returned.
            include_query_end_time: Determines if the query interval includes
                its end time. Defaults to no.
            include_op_end_times: Determines if the scheduled operation
                intervals include their end times or not. Defaults to no.

        Returns:
            A list of scheduled operations meeting the specified conditions.
        """
        earliest_time = time - self._max_duration
        end_time = time + duration
        qubits = None if qubits is None else frozenset(qubits)

        def overlaps_interval(op):
            if not include_op_end_times and op.time + op.duration == time:
                return False
            if not include_query_end_time and op.time == end_time:
                return False
            return op.time + op.duration >= time and op.time <= end_time

        def overlaps_qubits(op):
            if qubits is None:
                return True
            return not qubits.isdisjoint(op.operation.qubits)

        potential_matches = self.scheduled_operations.irange_key(earliest_time,
                                                                 end_time)
        return [op
                for op in potential_matches
                if overlaps_interval(op) and overlaps_qubits(op)]

    def __getitem__(self, item: Union[Timestamp, slice]):
        """Finds operations overlapping a given time or time slice.

        Args:
            item: Either a Timestamp or a slice containing start and stop
                Timestamps.

        Returns:
            The scheduled operations that occurs during the given time.
        """
        if isinstance(item, slice):
            if item.step:
                raise ValueError('Step not supported.')
            start = cast(Timestamp, item.start)
            stop = cast(Timestamp, item.stop)
            return self.query(time=start, duration=stop - start)
        return self.query(time=item, include_query_end_time=True)

    def operations_happening_at_same_time_as(
        self, scheduled_operation: ScheduledOperation
    ) -> List[ScheduledOperation]:
        """Finds operations happening at the same time as the given operation.

        Args:
            scheduled_operation: The operation specifying the time to query.

        Returns:
            Scheduled operations that overlap with the given operation.
        """
        overlaps = self.query(
            time=scheduled_operation.time,
            duration=scheduled_operation.duration)
        return [e for e in overlaps if e != scheduled_operation]

    def include(self, scheduled_operation: ScheduledOperation):
        """Adds a scheduled operation to the schedule.

        Args:
            scheduled_operation: The operation to add.

        Raises:
            ValueError:
                The operation collided with something already in the schedule.
        """
        collisions = self.query(time=scheduled_operation.time,
                                duration=scheduled_operation.duration,
                                qubits=scheduled_operation.operation.qubits)
        if collisions:
            raise ValueError('Operation {} has collisions: {}'.format(
                scheduled_operation.operation, collisions))
        self.scheduled_operations.add(scheduled_operation)
        self._max_duration = max(self._max_duration,
                                 scheduled_operation.duration)

    def exclude(self, scheduled_operation: ScheduledOperation) -> bool:
        """Omits a scheduled operation from the schedule, if present.

        Args:
            scheduled_operation: The operation to try to remove.

        Returns:
            True if the operation was present and is now removed, False if it
            was already not present.
        """
        try:
            self.scheduled_operations.remove(scheduled_operation)
            return True
        except ValueError:
            return False

    def to_circuit(self) -> Circuit:
        """Convert the schedule to a circuit.

        This discards most timing information from the schedule, but does place
        operations that are scheduled at the same time in the same Moment.
        """
        circuit = Circuit()
        ops = []  # type: List[Operation]
        time = None  # type: Optional[Timestamp]
        for so in self.scheduled_operations:
            if so.time != time:
                circuit.append(ops)
                ops = [so.operation]
                time = so.time
            else:
                ops.append(so.operation)
        circuit.append(ops)
        return circuit
