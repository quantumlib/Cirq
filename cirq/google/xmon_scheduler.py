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

from typing import Iterable
from typing import List, Union

from sortedcontainers import SortedListWithKey

from cirq import ops
from cirq.circuits import Circuit
from cirq.google.xmon_device import XmonDevice
from cirq.schedules import Schedule
from cirq.schedules.scheduled_operation import ScheduledOperation
from cirq.time import Duration, Timestamp


def _schedule_exp11_plane(schedule: Schedule, t: Timestamp, ops2):
    t0 = t
    dev = schedule.device
    d = dev.duration_of(ops.Exp11Gate().on(ops.QubitLoc(0, 0)))

    while ops2:
        i = 0
        while i < len(ops2):
            op = ops2[i]
            qs = {n for q in op.qubits for n in dev.neighbors_of(q)}
            if not schedule.query(time=t, duration=d, qubits=qs):
                schedule.include(ScheduledOperation.op_at_on(op, t, dev))
                del ops2[i]
            else:
                i += 1
        t += d

    return t - t0


def _schedule_single_plane(schedule: Schedule, t: Timestamp, ops2):
    scheduled_ops = [ScheduledOperation.op_at_on(op, t, schedule.device)
                     for op in ops2]
    duration = (Duration()
                if not scheduled_ops
                else max(e.duration for e in scheduled_ops))
    for scheduled_op in scheduled_ops:
        schedule.include(scheduled_op)
    return duration


def xmon_schedule_greedy(xmon_device: XmonDevice,
                         circuit: Circuit) -> Schedule:
    """Produces a schedule by greedily combining gates into layers.

    Args:
        xmon_device: The device the circuit will run on.
        circuit: The circuit to schedule.

    Returns:
        The schedule.
    """
    qubits = sorted(circuit.qubits(), key=lambda e: e.x + e.y*1000)
    ranges_min = {q: 0 for q in qubits}
    ranges_max = {q: 0 for q in qubits}

    schedule = Schedule(xmon_device)
    t = Timestamp()

    while any(e is not None for e in ranges_min.values()):
        for q in qubits:
            m = ranges_min[q]
            while (m is not None and
                       (not circuit.operation_at(q, m) or
                        isinstance(circuit.operation_at(q, m).gate,
                                   ops.Exp11Gate))):
                m = circuit.next_moment_operating_on([q],
                                                     start_moment_index=m + 1)
            ranges_max[q] = m

        ops_11 = []
        for q in qubits:
            if ranges_min[q] is not None:
                n = (ranges_max[q]
                     if ranges_max[q] is not None
                     else len(circuit.moments))
                for i in range(ranges_min[q], n):
                    op = circuit.operation_at(q, i)
                    if (op is not None and
                            op.qubits[0] is q and
                            (ranges_max[op.qubits[1]] is None or
                                     ranges_max[op.qubits[1]] > i)):
                        ops_11.append(op)

        t += _schedule_exp11_plane(schedule, t, ops_11)

        ops_1 = []
        for q in qubits:
            i = ranges_max[q]
            if i is not None:
                op = circuit.operation_at(q, i)
                if op is not None:
                    ops_1.append(op)

        t += _schedule_single_plane(schedule, t, ops_1)

        ranges_min = {q: e + 1 if e is not None else None
                      for q, e in ranges_max.items()}

    return schedule
