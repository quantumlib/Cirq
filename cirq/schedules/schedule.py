from typing import Iterable
from typing import List, Union

from sortedcontainers import SortedListWithKey

from cirq.chips import Chip
from cirq.ops import QubitId
from cirq.schedules.scheduled_operation import ScheduledOperation
from cirq.time import Duration, Timestamp


class Schedule:
    """A quantum program with operations happening at specific times."""

    def __init__(self,
                 chip: Chip,
                 scheduled_operations: Iterable[ScheduledOperation] = ()):
        """Initializes a new schedule.

        Args:
            chip: The hardware this schedule will run on.
            scheduled_operations: Initial list of operations to apply.
        """
        self.chip = chip
        self.scheduled_operations = SortedListWithKey(scheduled_operations,
                                                      key=lambda e: e.time)
        self._max_duration = max(
            [e.duration for e in self.scheduled_operations] or [Duration()])

    def query(self,
              *positional_args,
              time: Timestamp,
              duration: Duration = Duration(),
              qubits: Iterable[QubitId] = None) -> List[ScheduledOperation]:
        """Finds operations by time and qubit.

        Args:
            time: Operations must end after this time to be returned.
            duration: Operations must start by time+duration to be
                returned.
            qubits: If specified, only operations touching one of the included
                qubits will be returned.

        Returns:
            A list of scheduled operations meeting the specified conditions.
        """
        assert not positional_args
        earliest_time = time - self._max_duration
        end_time = time + duration
        qubits = None if qubits is None else frozenset(qubits)

        return [op
                for op in self.scheduled_operations.irange_key(earliest_time,
                                                               end_time)
                if op.time + op.duration >= time and (
                    qubits is None or
                    not qubits.isdisjoint(op.operation.qubits))]

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
            return self.query(time=item.start, duration=item.stop - item.start)
        return self.query(time=item)

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
            raise ValueError('Collisions: {}'.format(collisions))
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
