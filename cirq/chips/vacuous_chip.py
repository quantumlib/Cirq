from cirq.chips.chip import Chip
from cirq.time import Duration


class VacuousChipType(Chip):
    """A chip that allows everything."""

    def max_operation_duration(self):
        return Duration(picos=0)

    def duration_of(self, operation):
        return Duration(picos=0)

    def validate_operation(self, operation):
        pass

    def validate_scheduled_operation(self, schedule, scheduled_operation):
        pass

    def validate_circuit(self, circuit):
        pass

    def validate_schedule(self, schedule):
        pass


VacuousChip = VacuousChipType()
