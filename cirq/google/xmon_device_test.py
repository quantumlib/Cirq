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

import cirq
from cirq.circuits import Circuit
from cirq.devices import GridQubit
from cirq.google import (
    ExpWGate,
    ExpZGate,
    Exp11Gate,
    XmonDevice,
    XmonMeasurementGate
)
from cirq.testing import EqualsTester
from cirq.schedules import Schedule, ScheduledOperation
from cirq.value import Duration, Timestamp


def square_device(width, height, holes=()):
    ns = Duration(nanos=1)
    return XmonDevice(measurement_duration=ns,
                      exp_w_duration=2 * ns,
                      exp_11_duration=3 * ns,
                      qubits=[GridQubit(x, y)
                              for x in range(width)
                              for y in range(height)
                              if GridQubit(x, y) not in holes])


class NotImplementedOperation(cirq.Operation):
    @property
    def gate(self):
        raise NotImplementedError()

    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init():
    d = square_device(2, 2, holes=[GridQubit(1, 1)])
    ns = Duration(nanos=1)
    q00 = GridQubit(0, 0)
    q01 = GridQubit(0, 1)
    q10 = GridQubit(1, 0)

    assert d.qubits == {q00, q01, q10}
    assert d.duration_of(ExpZGate().on(q00)) == 0 * ns
    assert d.duration_of(cirq.measure(q00)) == ns
    assert d.duration_of(cirq.measure(q00, q01)) == ns
    assert d.duration_of(ExpWGate().on(q00)) == 2 * ns
    assert d.duration_of(Exp11Gate().on(q00, q01)) == 3 * ns


def test_validate_operation_adjacent_qubits():
    d = square_device(3, 3)

    d.validate_operation(cirq.GateOperation(
        Exp11Gate(),
        (GridQubit(0, 0), GridQubit(1, 0))))

    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(
            Exp11Gate(),
            (GridQubit(0, 0), GridQubit(2, 0))))


def test_validate_measurement_non_adjacent_qubits_ok():
    d = square_device(3, 3)

    d.validate_operation(cirq.GateOperation(
        XmonMeasurementGate(key=''),
        (GridQubit(0, 0), GridQubit(2, 0))))


def test_validate_operation_existing_qubits():
    d = square_device(3, 3, holes=[GridQubit(1, 1)])

    d.validate_operation(cirq.GateOperation(
        Exp11Gate(),
        (GridQubit(0, 0), GridQubit(1, 0))))
    d.validate_operation(cirq.GateOperation(ExpZGate(), (GridQubit(0, 0),)))

    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(
            Exp11Gate(),
            (GridQubit(0, 0), GridQubit(-1, 0))))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(ExpZGate(),
                                                (GridQubit(-1, 0),)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(
            Exp11Gate(),
            (GridQubit(1, 0), GridQubit(1, 1))))


def test_validate_operation_supported_gate():
    d = square_device(3, 3)

    class MyGate(cirq.Gate):
        pass

    d.validate_operation(cirq.GateOperation(ExpZGate(), [GridQubit(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(MyGate, [GridQubit(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(NotImplementedOperation())


def test_validate_scheduled_operation_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[GridQubit(1, 1)])
    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    q2 = GridQubit(2, 0)
    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            ExpWGate().on(q0), Timestamp(), d),
        ScheduledOperation.op_at_on(
            Exp11Gate().on(q1, q2), Timestamp(), d),
    ])
    with pytest.raises(ValueError):
        d.validate_schedule(s)


def test_validate_scheduled_operation_adjacent_exp_11_exp_z():
    d = square_device(3, 3, holes=[GridQubit(1, 1)])
    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    q2 = GridQubit(2, 0)
    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            ExpZGate().on(q0), Timestamp(), d),
        ScheduledOperation.op_at_on(
            Exp11Gate().on(q1, q2), Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_scheduled_operation_not_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[GridQubit(1, 1)])
    q0 = GridQubit(0, 0)
    p1 = GridQubit(1, 2)
    p2 = GridQubit(2, 2)
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
    circuit.append([XmonMeasurementGate('a').on(GridQubit(0, 0)),
                    XmonMeasurementGate('a').on(GridQubit(0, 1))])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_validate_schedule_repeat_measurement_keys():
    d = square_device(3, 3)

    s = Schedule(d, [
        ScheduledOperation.op_at_on(
            XmonMeasurementGate('a').on(GridQubit(0, 0)), Timestamp(), d),
        ScheduledOperation.op_at_on(
            XmonMeasurementGate('a').on(GridQubit(0, 1)), Timestamp(), d),

    ])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_schedule(s)


def test_xmon_device_eq():
    eq = EqualsTester()
    eq.make_equality_group(lambda: square_device(3, 3))
    eq.make_equality_group(
        lambda: square_device(3, 3,holes=[GridQubit(1, 1)]))
    eq.make_equality_group(
        lambda: XmonDevice(Duration(nanos=1), Duration(nanos=2),
                           Duration(nanos=3), []))
    eq.make_equality_group(
        lambda: XmonDevice(Duration(nanos=1), Duration(nanos=1),
                           Duration(nanos=1), []))
