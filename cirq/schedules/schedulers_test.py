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

from typing import cast

import pytest

import cirq
from cirq import ops, line
from cirq.circuits import Circuit
from cirq.circuits import Moment
from cirq.devices import Device
from cirq.schedules import (
    Schedule, ScheduledOperation, moment_by_moment_schedule,
)


class _TestDevice(Device):
    """A device for testing that only supports H and CZ gates on 10 qubits.

    The H gate take 20 nanos, and the CZ gates take 40 nanos.

    This device has 10 QubitLine qubits in a line, with x values ranging from
    0 to 9 (inclusive).
    """

    def __init__(self):
        self.qubits = [line.LineQubit(x) for x in range(10)]

    def duration_of(self, operation: ops.Operation) -> cirq.Duration:
        if isinstance(operation, ops.GateOperation):
            g = operation.gate
            if isinstance(g, ops.HGate):
                return cirq.Duration(nanos=20)
            if isinstance(g, ops.Rot11Gate):
                return cirq.Duration(nanos=40)
        raise ValueError('Unsupported operation: {!r}'.format(operation))

    def validate_gate(self, gate: ops.Gate):
        if not isinstance(gate, (ops.HGate, ops.Rot11Gate)):
            raise ValueError('Unsupported gate type {!r}'.format(gate))

    def validate_operation(self, operation: ops.Operation):
        if not isinstance(operation, ops.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, line.LineQubit):
                raise ValueError('Unsupported qubit type: {!r}'.format(q))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if len(operation.qubits) == 2:
            p, q = operation.qubits
            if not cast(line.LineQubit, p).is_adjacent(cast(line.LineQubit, q)):
                raise ValueError(
                    'Non-local interaction: {!r}.'.format(operation))

    def validate_scheduled_operation(self,
                                     schedule: Schedule,
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
        return any(cast(line.LineQubit, q).is_adjacent(cast(line.LineQubit, p))
                   for q in cz_op.qubits
                   for p in other_op.qubits)

    def validate_circuit(self, circuit):
        for moment in circuit:
            for operation in moment.operations:
                self.validate_operation(operation)

    def validate_schedule(self, schedule):
        for scheduled_operation in schedule.scheduled_operations:
            self.validate_scheduled_operation(schedule, scheduled_operation)


class NotImplementedOperation(cirq.Operation):
    @property
    def gate(self):
        raise NotImplementedError()

    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_the_test_device():
    device = _TestDevice()

    device.validate_gate(cirq.H)
    with pytest.raises(ValueError):
        device.validate_gate(cirq.X)

    device.validate_operation(cirq.H(cirq.LineQubit(0)))
    with pytest.raises(ValueError):
        device.validate_operation(NotImplementedOperation())

    device.validate_schedule(Schedule(device, []))

    device.validate_schedule(
        Schedule(device, [
            ScheduledOperation.op_at_on(
                cirq.H(cirq.LineQubit(0)),
                cirq.Timestamp(),
                device),
            ScheduledOperation.op_at_on(
                cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)),
                cirq.Timestamp(),
                device)
        ]))

    with pytest.raises(ValueError):
        device.validate_schedule(
            Schedule(device, [ScheduledOperation.op_at_on(
                NotImplementedOperation(),
                cirq.Timestamp(),
                device)]))

    with pytest.raises(ValueError):
        device.validate_schedule(
            Schedule(device, [ScheduledOperation.op_at_on(
                cirq.X(cirq.LineQubit(0)),
                cirq.Timestamp(),
                device)]))

    with pytest.raises(ValueError):
        device.validate_schedule(
            Schedule(device, [ScheduledOperation.op_at_on(
                cirq.H(cirq.NamedQubit('q')),
                cirq.Timestamp(),
                device)]))

    with pytest.raises(ValueError):
        device.validate_schedule(
            Schedule(device, [ScheduledOperation.op_at_on(
                cirq.H(cirq.LineQubit(100)),
                cirq.Timestamp(),
                device)]))

    with pytest.raises(ValueError):
        device.validate_schedule(
            Schedule(device, [ScheduledOperation.op_at_on(
                cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(3)),
                cirq.Timestamp(),
                device)]))


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

    zero_ns = cirq.Timestamp()
    assert set(schedule.scheduled_operations) == {
        ScheduledOperation.op_at_on(ops.H(q), zero_ns, device) for q in qubits}


def test_moment_by_moment_schedule_moment_of_two_qubit_ops():
    device = _TestDevice()
    qubits = device.qubits

    circuit = Circuit(
        [Moment((ops.CZ(qubits[i], qubits[i + 1]) for i in range(0, 9, 3)))])
    schedule = moment_by_moment_schedule(device, circuit)

    zero_ns = cirq.Timestamp()
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

    zero_ns = cirq.Timestamp()
    twenty_ns = cirq.Timestamp(nanos=20)
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

    zero_ns = cirq.Timestamp()
    fourty_ns = cirq.Timestamp(nanos=40)
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

    zero_ns = cirq.Timestamp()
    twenty_ns = cirq.Timestamp(nanos=20)
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
