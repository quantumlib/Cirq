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
import numpy as np
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


def square_device(width: int, height: int, holes=()) -> XmonDevice:
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


def test_xmon_device_str():
    assert str(square_device(2, 2)).strip() == """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip()


def test_get_item_single():
    dev = square_device(4, 3)
    g = cirq.GridQubit

    assert dev[2, 2] == g(2, 2)
    assert dev[0, 0] == g(0, 0)
    assert dev[3, 2] == g(3, 2)
    with pytest.raises(IndexError):
        _ = dev[-1, 2]
    with pytest.raises(IndexError):
        _ = dev[2, -1]
    with pytest.raises(IndexError):
        _ = dev[4, 2]
    with pytest.raises(IndexError):
        _ = dev[2, 3]


def test_get_item_row_slice():
    dev = square_device(3, 4)
    g = cirq.GridQubit

    assert dev[:, 0] == [g(0, 0),
                         g(1, 0),
                         g(2, 0)]
    assert dev[1:, 0] == [g(1, 0),
                          g(2, 0)]
    assert dev[:1, 0] == [g(0, 0)]
    assert dev[:, 3] == [g(0, 3),
                         g(1, 3),
                         g(2, 3)]
    assert dev[::-1, 3] == [g(2, 3),
                            g(1, 3),
                            g(0, 3)]
    assert dev[::2, 3] == [g(0, 3),
                           g(2, 3)]
    assert dev[1::2, 3] == [g(1, 3)]
    assert dev[1:3, 3] == [g(1, 3),
                           g(2, 3)]
    assert dev[1:3:-1, 3] == []
    assert dev[2::-1, 3] == [g(2, 3),
                             g(1, 3),
                             g(0, 3)]
    assert dev[2:1:-1, 3] == [g(2, 3)]
    assert dev[0::-1, 3] == [g(0, 3)]
    assert dev[6:, 3] == []

    with pytest.raises(IndexError):
        _ = dev[:, -1]
    with pytest.raises(IndexError):
        _ = dev[:, 4]
    with pytest.raises(IndexError):
        _ = dev[3::-1, 4] == []
    assert dev[3::-1, 3] == [g(2, 3), g(1, 3), g(0, 3)]
    assert dev[-1::-1, 3] == []


def test_get_item_col_slice():
    dev = square_device(4, 3)
    g = cirq.GridQubit

    assert dev[0, :] == [g(0, 0),
                         g(0, 1),
                         g(0, 2)]
    assert dev[3, :] == [g(3, 0),
                         g(3, 1),
                         g(3, 2)]
    assert dev[3, ::-1] == [g(3, 2),
                            g(3, 1),
                            g(3, 0)]
    assert dev[3, ::2] == [g(3, 0),
                           g(3, 2)]
    assert dev[3, 1::2] == [g(3, 1)]
    assert dev[3, 1:3] == [g(3, 1),
                           g(3, 2)]
    assert dev[3, 1:3:-1] == []
    assert dev[3, 2::-1] == [g(3, 2),
                             g(3, 1),
                             g(3, 0)]
    assert dev[3, 2:1:-1] == [g(3, 2)]
    assert dev[3, 0::-1] == [g(3, 0)]

    with pytest.raises(IndexError):
        _ = dev[-1, :]
    with pytest.raises(IndexError):
        _ = dev[4, :]
    assert dev[3, 10::-1] == [g(3, 2), g(3, 1), g(3, 0)]
    assert dev[3, -1::-1] == []


def test_get_item_grid_slice():
    dev = square_device(2, 3)
    g = cirq.GridQubit

    assert np.alltrue(dev[:, :] == [
        [g(0, 0), g(0, 1), g(0, 2)],
        [g(1, 0), g(1, 1), g(1, 2)],
    ])

    assert np.alltrue(dev[:, :2] == [
        [g(0, 0), g(0, 1)],
        [g(1, 0), g(1, 1)],
    ])

    assert np.alltrue(dev[:1, :] == [
        [g(0, 0), g(0, 1), g(0, 2)],
    ])

    assert np.alltrue(dev[::2, 1::2] == [
        [g(0, 1)],
    ])

    assert np.alltrue(dev[1::-1, 1::-1] == [
        [g(1, 1), g(1, 0)],
        [g(0, 1), g(0, 0)],
    ])

    assert np.alltrue(dev[1:, 1:] == [
        [g(1, 1), g(1, 2)],
    ])


def test_get_item_not_square():
    dev = cirq.google.Bristlecone
    g = cirq.GridQubit

    # Respects the border when picking individuals.
    with pytest.raises(IndexError):
        _ = dev[0, 0]
    with pytest.raises(IndexError):
        _ = dev[2, 2]
    assert dev[3, 3] == g(3, 3)
    assert dev[2, 3] == g(2, 3)

    # Row and column queries work when in range.
    with pytest.raises(IndexError):
        _ = dev[:, -1]
    assert dev[:, 0] == [g(5, 0)]
    assert dev[:, 1] == [g(4, 1), g(5, 1), g(6, 1)]
    assert dev[0, :] == [g(0, 5), g(0, 6)]
    assert dev[1, :] == [g(1, 4), g(1, 5), g(1, 6), g(1, 7)]
    assert dev[2, ::2] == [g(2, 3), g(2, 5), g(2, 7)]
    assert dev[2, ::-2] == [g(2, 8), g(2, 6), g(2, 4)]

    # Full grid fails when invalid corners are included.
    with pytest.raises(IndexError):
        _ = dev[:, :]
    assert np.alltrue(dev[2:3, 2:3] == [])
    with pytest.raises(IndexError):
        _ = dev[2:4, 2:4]
    assert np.alltrue(dev[3:4, 3:4] == [[g(3, 3)]])
    assert np.alltrue(dev[2:4, 3:6] == [
        [g(2, 3), g(2, 4), g(2, 5)],
        [g(3, 3), g(3, 4), g(3, 5)],
    ])
    with pytest.raises(IndexError):
        _ = dev[5:7, :]
    with pytest.raises(IndexError):
        _ = dev[4:7, :]
    assert np.alltrue(dev[:, 5:7] == [
        [g(0, 5), g(0, 6)],
        [g(1, 5), g(1, 6)],
        [g(2, 5), g(2, 6)],
        [g(3, 5), g(3, 6)],
        [g(4, 5), g(4, 6)],
        [g(5, 5), g(5, 6)],
        [g(6, 5), g(6, 6)],
        [g(7, 5), g(7, 6)],
        [g(8, 5), g(8, 6)],
        [g(9, 5), g(9, 6)],
        [g(10, 5), g(10, 6)],
    ])
