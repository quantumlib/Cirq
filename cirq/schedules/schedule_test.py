# Copyright 2017 Google LLC
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
import pytest

from cirq import ops
from cirq.chips import VacuousChip
from cirq.schedules import Duration, Timestamp, ScheduledOperation, Schedule
from cirq.testing import EqualsTester


def test_query_point_operation():
    q = ops.QubitId(0, 0)
    zero = Timestamp(picos=0)
    ps = Duration(picos=1)
    op = ScheduledOperation(zero, Duration(), ops.H(q))
    schedule = Schedule(chip=VacuousChip, scheduled_operations=[op])

    assert schedule.query(time=zero) == [op]
    assert schedule.query(time=zero + ps) == []
    assert schedule.query(time=zero - ps) == []

    assert schedule.query(time=zero, qubits=[]) == []
    assert schedule.query(time=zero, qubits=[q]) == [op]

    assert schedule.query(time=zero, duration=ps) == [op]
    assert schedule.query(time=zero - ps, duration=ps) == [op]
    assert schedule.query(time=zero - 2 * ps, duration=ps) == []
    assert schedule.query(time=zero - ps, duration=3 * ps) == [op]
    assert schedule.query(time=zero + ps, duration=ps) == []


def test_query_overlapping_operations():
    q = ops.QubitId(0, 0)
    zero = Timestamp(picos=0)
    ps = Duration(picos=1)
    op1 = ScheduledOperation(zero, 2 * ps, ops.H(q))
    op2 = ScheduledOperation(zero + ps, 2 * ps, ops.H(q))
    schedule = Schedule(chip=VacuousChip, scheduled_operations=[op2, op1])

    assert schedule.query(time=zero - 0.5 * ps, duration=ps) == [op1]
    assert schedule.query(time=zero, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 0.5 * ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 1.5 * ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 2.0 * ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 2.5 * ps, duration=ps) == [op2]
    assert schedule.query(time=zero + 3.0 * ps, duration=ps) == [op2]
    assert schedule.query(time=zero + 3.5 * ps, duration=ps) == []


def test_slice_operations():
    q0 = ops.QubitId(0, 0)
    q1 = ops.QubitId(0, 1)
    zero = Timestamp(picos=0)
    ps = Duration(picos=1)
    op1 = ScheduledOperation(zero, ps, ops.H(q0))
    op2 = ScheduledOperation(zero + 2 * ps, 2 * ps, ops.CZ(q0, q1))
    op3 = ScheduledOperation(zero + 10 * ps, ps, ops.H(q1))
    schedule = Schedule(chip=VacuousChip, scheduled_operations=[op1, op2, op3])

    assert schedule[zero] == [op1]
    assert schedule[zero:zero] == [op1]
    assert schedule[zero:zero + ps] == [op1]
    assert schedule[zero:zero + 2 * ps] == [op1, op2]
    assert schedule[zero:zero + 20 * ps] == [op1, op2, op3]
    assert schedule[zero + 2.5 * ps:zero + 20 * ps] == [op2, op3]
    assert schedule[zero + 5 * ps:zero + 20 * ps] == [op3]


def test_include():
    q0 = ops.QubitId(0, 0)
    q1 = ops.QubitId(0, 1)
    zero = Timestamp(picos=0)
    ps = Duration(picos=1)
    schedule = Schedule(chip=VacuousChip)

    op0 = ScheduledOperation(zero, ps, ops.H(q0))
    schedule.include(op0)
    with pytest.raises(ValueError):
        schedule.include(ScheduledOperation(zero, ps, ops.H(q0)))
        schedule.include(ScheduledOperation(zero + 0.5 * ps, ps, ops.H(q0)))
    op1 = ScheduledOperation(zero + 2 * ps, ps, ops.H(q0))
    schedule.include(op1)
    op2 = ScheduledOperation(zero + 0.5 * ps, ps, ops.H(q1))
    schedule.include(op2)

    assert schedule.query(time=zero, duration=ps * 10) == [op0, op2, op1]


def test_exclude():
    q = ops.QubitId(0, 0)
    zero = Timestamp(picos=0)
    ps = Duration(picos=1)
    op = ScheduledOperation(zero, ps, ops.H(q))
    schedule = Schedule(chip=VacuousChip, scheduled_operations=[op])

    assert not schedule.exclude(ScheduledOperation(zero + ps, ps, ops.H(q)))
    assert not schedule.exclude(ScheduledOperation(zero, ps, ops.X(q)))
    assert schedule.query(time=zero, duration=ps * 10) == [op]
    assert schedule.exclude(ScheduledOperation(zero, ps, ops.H(q)))
    assert schedule.query(time=zero, duration=ps * 10) == []
    assert not schedule.exclude(ScheduledOperation(zero, ps, ops.H(q)))
