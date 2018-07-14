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

"General methods for creating Schedules from Circuits."

from cirq.circuits import  Circuit
from cirq.devices import Device
from cirq.schedules import Schedule
from cirq.schedules import ScheduledOperation
from cirq.value import Timestamp


def moment_by_moment_schedule(device: Device, circuit: Circuit):
    """Returns a schedule aligned with the moment structure of the Circuit.

    This method attempts to create a schedule in which each moment of a circuit
    is scheduled starting at the same time. Given the constraints of the
    given device, such a schedule may not be possible, in this case the
    the method will raise a ValueError with a description of the conflict.

    The schedule that is produced will take each moments and schedule the
    operations in this moment in a time slice of length equal to the maximum
    time of an operation in the moment.

    Returns:
        A Schedule for the circuit.

    Raises:
        ValueError: if the scheduling cannot be done.
    """
    schedule = Schedule(device)
    t = Timestamp()
    for moment in circuit:
        if not moment.operations:
            continue
        for op in moment.operations:
            scheduled_op = ScheduledOperation.op_at_on(op, t, device)
            # Raises a ValueError describing the problem if this cannot be
            # scheduled.
            schedule.include(scheduled_operation=scheduled_op)
            # Raises ValueError at first sign of a device conflict.
            device.validate_scheduled_operation(schedule, scheduled_op)
        # Increment time for next moment by max of ops during this moment.
        max_duration = max(device.duration_of(op) for op in moment.operations)
        t += max_duration
    return schedule
