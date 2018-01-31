from cirq import ops
from cirq.time import Duration, Timestamp


class ScheduledOperation:
    """An operation that happens over a specified time interval."""

    def __init__(self,
                 time: Timestamp,
                 duration: Duration,
                 operation: ops.Operation):
        """Initializes the scheduled operation.

        Args:
            time: When the operation starts.
            duration: How long the operation lasts.
            operation: The operation.
        """
        self.time = time
        self.duration = duration
        self.operation = operation

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
