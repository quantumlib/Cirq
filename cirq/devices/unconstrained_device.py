from cirq.devices.device import Device
from cirq.time import Duration


class _UnconstrainedDeviceType(Device):
    """A device that allows everything."""

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


UnconstrainedDevice = _UnconstrainedDeviceType()
