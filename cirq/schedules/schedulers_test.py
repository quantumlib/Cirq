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

from typing import cast

import pytest

from cirq import ops
from cirq.circuits import Circuit
from cirq.circuits import Moment
from cirq.devices import Device
from cirq.schedules import (
    Schedule, ScheduledOperation, moment_by_moment_schedule,
)
from cirq.value import Duration, Timestamp


class LineQubit:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((LineQubit, self.x))

    def is_adjacent(self, other):
        return abs(self.x - other.x) == 1


class _TestDevice(Device):
    """A device for testing that only supports H and CZ gates on 10 qubits.

    The H gate take 20 nanos, and the CZ gates take 40 nanos.

    This device has 10 QubitLine qubits in a line, with x values ranging from
    0 to 9 (inclusive).
    """

    def __init__(self):
        self.qubits = [LineQubit(x) for x in range(10)]

    def duration_of(self, operation: ops.Operation) -> Duration:
        g = operation.gate
        if isinstance(g, ops.HGate):
            return Duration(nanos=20)
        elif isinstance(g, ops.Rot11Gate):
            return Duration(nanos=40)
        else:
            raise ValueError('Unsupported gate type {}'.format(repr(g)))

    def validate_gate(self, gate: ops.Gate):
        if not isinstance(gate, (ops.HGate, ops.Rot11Gate)):
            raise ValueError('Unsupported gate type {}'.format(repr(gate)))

    def validate_operation(self, operation: ops.Operation):
        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, LineQubit):
                raise ValueError('Unsupported qubit type: {}'.format(repr(q)))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {}'.format(repr(q)))

        if len(operation.qubits) == 2:
            p, q = operation.qubits
            if not cast(LineQubit, p).is_adjacent(q):
                raise ValueError(
                    'Non-local interaction: {}.'.format(repr(operation)))

    def validate_scheduled_operation(self, schedule: Schedule,
        scheduled_operation: ScheduledOperation):
        op = scheduled_operation.operation
        self.validate_operation(op)
        if isinstance(op.gate, ops.Rot11Gate):
            for other in schedule.operations_happening_at_same_time_as(
                scheduled_operation):
                if self.check_if_cz_adjacent(op, other.operation):
                    raise ValueError('Adjacent CZ operations: {} vs {}'.format(
                        scheduled_operation, other))

    def check_if_cz_adjacent(self,
        cz_op: ops.Operation,
        other_op: ops.Operation):
        if isinstance(other_op.gate, ops.HGate):
            return False
        return any(
            cast(LineQubit, q).is_adjacent(p)
            for q in cz_op.qubits for p in other_op.qubits)

    def validate_circuit(self, circuit):
        for moment in circuit.moments:
            for operation in moment.operations:
                self.validate_operation(operation)

    def validate_schedule(self, schedule):
        for scheduled_operation in schedule.scheduled_operations:
            self.validate_scheduled_operation(schedule, scheduled_operation)


def test_moment_by_moment_schedule_no_moments():
    device = _TestDevice()
    circuit = Circuit([])
    schedule = moment_by_moment_schedule(device, circuit)
    assert len(schedule.scheduled_operations) == 0


def test_moment_by_moment_schedule_empty_moment():
    device = _TestDevice()
    circuit = Circuit([Moment(),])
    schedule = moment_by_moment_schedule(device, circuit)
    assert len(schedule.scheduled_operations) == 0


def test_moment_by_moment_schedule_moment_of_single_qubit_ops():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit([Moment(ops.H(q) for q in qubits),])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = Timestamp()
    assert set(schedule.scheduled_operations) == {
        ScheduledOperation.op_at_on(ops.H(q), zero_ns, device) for q in qubits}


def test_moment_by_moment_schedule_moment_of_two_qubit_ops():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit(
        [Moment((ops.CZ(qubits[i], qubits[i + 1]) for i in range(0, 9, 3)))])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = Timestamp()
    expected = set(
        ScheduledOperation.op_at_on(ops.CZ(qubits[i], qubits[i + 1]), zero_ns,
                                    device) for i in range(0, 9, 3))
    assert set(schedule.scheduled_operations) == expected


def test_moment_by_moment_schedule_two_moments():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit([Moment(ops.H(q) for q in qubits),
                       Moment((ops.CZ(qubits[i], qubits[i + 1]) for i in
                               range(0, 9, 3)))])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = Timestamp()
    twenty_ns = Timestamp(nanos=20)
    expected_one_qubit = set(
        ScheduledOperation.op_at_on(ops.H(q), zero_ns, device) for q in qubits)
    expected_two_qubit = set(
        ScheduledOperation.op_at_on(
            ops.CZ(qubits[i], qubits[i + 1]), twenty_ns,
            device) for i in range(0, 9, 3))
    expected = expected_one_qubit.union(expected_two_qubit)
    assert set(schedule.scheduled_operations) == expected


def test_moment_by_moment_schedule_max_duration():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit([
        Moment([ops.H(qubits[0]), ops.CZ(qubits[1], qubits[2])]),
        Moment([ops.H(qubits[0])])])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = Timestamp()
    fourty_ns = Timestamp(nanos=40)
    assert set(schedule.scheduled_operations) == {
        ScheduledOperation.op_at_on(ops.H(qubits[0]), zero_ns, device),
        ScheduledOperation.op_at_on(
            ops.CZ(qubits[1], qubits[2]), zero_ns, device),
        ScheduledOperation.op_at_on(ops.H(qubits[0]), fourty_ns, device),
    }


def test_moment_by_moment_schedule_empty_moment_ignored():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit([Moment([ops.H(qubits[0])]),
                       Moment([]),
                       Moment([ops.H(qubits[0])])])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = Timestamp()
    twenty_ns = Timestamp(nanos=20)
    assert set(schedule.scheduled_operations) == {
        ScheduledOperation.op_at_on(ops.H(qubits[0]), zero_ns, device),
        ScheduledOperation.op_at_on(ops.H(qubits[0]), twenty_ns, device),
    }


def test_moment_by_moment_schedule_validate_operation_fails():
    device = _TestDevice()
    qubits = device.qubits
    circuit = Circuit()
    circuit.append(ops.CNOT(qubits[0], qubits[1]))
    with pytest.raises(ValueError, match="CNOT"):
        _ = moment_by_moment_schedule(device, circuit)


def test_moment_by_moment_schedule_device_validation_fails():
    device = _TestDevice()
    qubits = device.qubits
    circuit = Circuit([Moment([
        ops.CZ(qubits[0], qubits[1]),
        ops.CZ(qubits[2], qubits[3])
    ])])
    with pytest.raises(ValueError, match="Adjacent CZ"):
        _ = moment_by_moment_schedule(device, circuit)
