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

import pytest

from cirq import ops
from cirq.circuits import Circuit
from cirq.google import (ExpWGate, ExpZGate, Exp11Gate, XmonDevice,
                         XmonMeasurementGate, XmonQubit)
from cirq.testing import EqualsTester
from cirq.schedules import Schedule, ScheduledOperation
from cirq.value import Duration, Timestamp


def square_device(width, height, holes=()):
    ns = Duration(nanos=1)
    return XmonDevice(measurement_duration=ns,
                      exp_w_duration=2 * ns,
                      exp_11_duration=3 * ns,
                      qubits=[XmonQubit(x, y)
                              for x in range(width)
                              for y in range(height)
                              if XmonQubit(x, y) not in holes])


def test_init():
    d = square_device(2, 2, holes=[XmonQubit(1, 1)])
    ns = Duration(nanos=1)
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q10 = XmonQubit(1, 0)

    assert d.qubits == {q00, q01, q10}
    assert d.duration_of(ops.Operation(ExpZGate(), (q00,))) == 0 * ns
    assert d.duration_of(ops.Operation(ops.MeasurementGate(),
                                       (q00,))) == ns
    assert d.duration_of(ops.Operation(ExpWGate(), (q00,))) == 2 * ns
    assert d.duration_of(ops.Operation(Exp11Gate(), (q00, q01))) == 3 * ns


def test_validate_operation_adjacent_qubits():
    d = square_device(3, 3)

    d.validate_operation(ops.Operation(
        Exp11Gate(),
        (XmonQubit(0, 0), XmonQubit(1, 0))))

    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            Exp11Gate(),
            (XmonQubit(0, 0), XmonQubit(2, 0))))


def test_validate_measurement_non_adjacent_qubits_ok():
    d = square_device(3, 3)

    d.validate_operation(ops.Operation(
        XmonMeasurementGate(),
        (XmonQubit(0, 0), XmonQubit(2, 0))))


def test_validate_operation_existing_qubits():
    d = square_device(3, 3, holes=[XmonQubit(1, 1)])

    d.validate_operation(ops.Operation(
        Exp11Gate(),
        (XmonQubit(0, 0), XmonQubit(1, 0))))
    d.validate_operation(ops.Operation(ExpZGate(), (XmonQubit(0, 0),)))

    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            Exp11Gate(),
            (XmonQubit(0, 0), XmonQubit(-1, 0))))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(ExpZGate(),
                                           (XmonQubit(-1, 0),)))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            Exp11Gate(),
            (XmonQubit(1, 0), XmonQubit(1, 1))))


def test_validate_operation_supported_gate():
    d = square_device(3, 3)

    class MyGate(ops.Gate):
        pass

    d.validate_operation(ops.Operation(ExpZGate(), [XmonQubit(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(MyGate, [XmonQubit(0, 0)]))


def test_validate_scheduled_operation_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[XmonQubit(1, 1)])
    q0 = XmonQubit(0, 0)
    q1 = XmonQubit(1, 0)
    q2 = XmonQubit(2, 0)
    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            ExpWGate().on(q0), Timestamp(), d),
        ScheduledOperation.op_at_on(
            Exp11Gate().on(q1, q2), Timestamp(), d),
    ])
    with pytest.raises(ValueError):
        d.validate_schedule(s)


def test_validate_scheduled_operation_adjacent_exp_11_exp_z():
    d = square_device(3, 3, holes=[XmonQubit(1, 1)])
    q0 = XmonQubit(0, 0)
    q1 = XmonQubit(1, 0)
    q2 = XmonQubit(2, 0)
    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            ExpZGate().on(q0), Timestamp(), d),
        ScheduledOperation.op_at_on(
            Exp11Gate().on(q1, q2), Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_scheduled_operation_not_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[XmonQubit(1, 1)])
    q0 = XmonQubit(0, 0)
    p1 = XmonQubit(1, 2)
    p2 = XmonQubit(2, 2)
    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            ExpWGate().on(q0), Timestamp(), d),
        ScheduledOperation.op_at_on(
            Exp11Gate().on(p1, p2), Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_circuit_repeat_measurement_keys():
    d = square_device(3, 3)

    circuit = Circuit()
    circuit.append([XmonMeasurementGate('a').on(XmonQubit(0, 0)),
                    XmonMeasurementGate('a').on(XmonQubit(0, 1))])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_validate_schedule_repeat_measurement_keys():
    d = square_device(3, 3)

    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            XmonMeasurementGate('a').on(XmonQubit(0, 0)), Timestamp(), d),
        ScheduledOperation.op_at_on(
            XmonMeasurementGate('a').on(XmonQubit(0, 1)), Timestamp(), d),

    ])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_schedule(s)


def test_xmon_device_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: square_device(3, 3))
    eq.make_equality_pair(lambda: square_device(3, 3, holes=[XmonQubit(1, 1)]))
    eq.make_equality_pair(
        lambda: XmonDevice(Duration(nanos=1), Duration(nanos=2),
                           Duration(nanos=3), []))
    eq.make_equality_pair(
        lambda: XmonDevice(Duration(nanos=1), Duration(nanos=1),
                           Duration(nanos=1), []))
