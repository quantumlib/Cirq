import abc

import cirq
from cirq.time import Duration

# Note: circuit/schedule types specified by name to avoid circular references.


class Chip:
    """Hardware constraints for validating circuits and schedules."""

    @abc.abstractmethod
    def max_operation_duration(self) -> Duration:
        pass

    @abc.abstractmethod
    def duration_of(self, operation: 'cirq.ops.Operation') -> Duration:
        pass

    @abc.abstractmethod
    def validate_operation(self, operation: 'cirq.ops.Operation'
                           ) -> type(None):
        """Raises an exception if an operation is not valid.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this chip.
        """
        pass

    @abc.abstractmethod
    def validate_scheduled_operation(
            self,
            schedule: 'cirq.schedules.Schedule',
            scheduled_operation: 'cirq.schedules.ScheduledOperation'
    ) -> type(None):
        pass

    @abc.abstractmethod
    def validate_circuit(self, circuit: 'cirq.circuits.Circuit') -> type(None):
        """Raises an exception if a circuit is not valid.

        Args:
            circuit: The circuit to validate.

        Raises:
            ValueError: The circuit isn't valid for this chip.
        """
        pass

    @abc.abstractmethod
    def validate_schedule(self, schedule: 'cirq.schedules.Schedule'
                          ) -> type(None):
        """Raises an exception if a schedule is not valid.

        Args:
            schedule: The schedule to validate.

        Raises:
            ValueError: The schedule isn't valid for this chip.
        """
        pass
