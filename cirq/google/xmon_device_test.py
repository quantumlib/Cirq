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
import cirq.google as cg


def square_device(width: int, height: int, holes=()) -> cg.XmonDevice:
    ns = cirq.Duration(nanos=1)
    return cg.XmonDevice(measurement_duration=ns,
                         exp_w_duration=2 * ns,
                         exp_11_duration=3 * ns,
                         qubits=[cirq.GridQubit(row, col)
                                 for col in range(width)
                                 for row in range(height)
                                 if cirq.GridQubit(col, row) not in holes])


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)])
    ns = cirq.Duration(nanos=1)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)

    assert d.qubits == {q00, q01, q10}
    assert d.duration_of(cirq.Z(q00)) == 0 * ns
    assert d.duration_of(cirq.measure(q00)) == ns
    assert d.duration_of(cirq.measure(q00, q01)) == ns
    assert d.duration_of(cirq.X(q00)) == 2 * ns
    assert d.duration_of(cirq.CZ(q00, q01)) == 3 * ns
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.Gate().on(q00))


@cirq.testing.only_test_in_python3
def test_repr():
    d = square_device(2, 2, holes=[])

    assert repr(d) == ("XmonDevice("
                       "measurement_duration=cirq.Duration(picos=1000), "
                       "exp_w_duration=cirq.Duration(picos=2000), "
                       "exp_11_duration=cirq.Duration(picos=3000) "
                       "qubits=[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), "
                       "cirq.GridQubit(1, 0), "
                       "cirq.GridQubit(1, 1)])")


def test_can_add_operation_into_moment():
    d = square_device(2, 2)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    m = cirq.Moment([cirq.CZ(q00, q01)])
    assert not d.can_add_operation_into_moment(
        cirq.CZ(q10, q11), m)


def test_validate_moment():
    d = square_device(2, 2)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    m = cirq.Moment([cirq.CZ(q00, q01),
                     cirq.CZ(q10, q11)])
    with pytest.raises(ValueError):
        d.validate_moment(m)


def test_validate_operation_adjacent_qubits():
    d = square_device(3, 3)

    d.validate_operation(cirq.GateOperation(
        cirq.CZ,
        (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))))

    with pytest.raises(ValueError, match='Non-local interaction'):
        d.validate_operation(cirq.GateOperation(
            cirq.CZ,
            (cirq.GridQubit(0, 0), cirq.GridQubit(2, 0))))


def test_validate_measurement_non_adjacent_qubits_ok():
    d = square_device(3, 3)

    d.validate_operation(cirq.GateOperation(
        cirq.MeasurementGate(key=''),
        (cirq.GridQubit(0, 0), cirq.GridQubit(2, 0))))


def test_validate_operation_existing_qubits():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])

    d.validate_operation(cirq.GateOperation(
        cirq.CZ,
        (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))))
    d.validate_operation(cirq.Z(cirq.GridQubit(0, 0)))

    with pytest.raises(ValueError):
        d.validate_operation(
            cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(-1, 0)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.Z(cirq.GridQubit(-1, 0)))
    with pytest.raises(ValueError):
        d.validate_operation(
            cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)))


def test_validate_operation_supported_gate():
    d = square_device(3, 3)

    class MyGate(cirq.Gate):
        pass

    d.validate_operation(cirq.GateOperation(cirq.Z, [cirq.GridQubit(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(
            MyGate, [cirq.GridQubit(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(NotImplementedOperation())


def test_validate_scheduled_operation_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(2, 0)
    s = cirq.Schedule(d, [
        cirq.ScheduledOperation.op_at_on(
            cirq.X(q0), cirq.Timestamp(), d),
        cirq.ScheduledOperation.op_at_on(
            cirq.CZ(q1, q2), cirq.Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_scheduled_operation_adjacent_exp_11_exp_z():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(2, 0)
    s = cirq.Schedule(d, [
        cirq.ScheduledOperation.op_at_on(
            cirq.Z(q0), cirq.Timestamp(), d),
        cirq.ScheduledOperation.op_at_on(
            cirq.CZ(q1, q2), cirq.Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_scheduled_operation_adjacent_exp_11_measure():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(2, 0)
    s = cirq.Schedule(d, [
        cirq.ScheduledOperation.op_at_on(
            cirq.MeasurementGate().on(q0), cirq.Timestamp(), d),
        cirq.ScheduledOperation.op_at_on(
            cirq.CZ(q1, q2), cirq.Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_scheduled_operation_not_adjacent_exp_11_exp_w():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])
    q0 = cirq.GridQubit(0, 0)
    p1 = cirq.GridQubit(1, 2)
    p2 = cirq.GridQubit(2, 2)
    s = cirq.Schedule(d, [
        cirq.ScheduledOperation.op_at_on(
            cirq.X(q0), cirq.Timestamp(), d),
        cirq.ScheduledOperation.op_at_on(
            cirq.CZ(p1, p2), cirq.Timestamp(), d),
    ])
    d.validate_schedule(s)


def test_validate_circuit_repeat_measurement_keys():
    d = square_device(3, 3)

    circuit = cirq.Circuit()
    circuit.append([cirq.MeasurementGate('a').on(cirq.GridQubit(0, 0)),
                    cirq.MeasurementGate('a').on(cirq.GridQubit(0, 1))])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_validate_schedule_repeat_measurement_keys():
    d = square_device(3, 3)

    s = cirq.Schedule(d, [
        cirq.ScheduledOperation.op_at_on(
            cirq.MeasurementGate('a').on(
                cirq.GridQubit(0, 0)), cirq.Timestamp(), d),
        cirq.ScheduledOperation.op_at_on(
            cirq.MeasurementGate('a').on(
                cirq.GridQubit(0, 1)), cirq.Timestamp(), d),
    ])

    with pytest.raises(ValueError, message='Measurement key a repeated'):
        d.validate_schedule(s)


def test_xmon_device_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: square_device(3, 3))
    eq.make_equality_group(
        lambda: square_device(3, 3, holes=[cirq.GridQubit(1, 1)]))
    eq.make_equality_group(
        lambda: cg.XmonDevice(cirq.Duration(nanos=1), cirq.Duration(nanos=2),
                              cirq.Duration(nanos=3), []))
    eq.make_equality_group(
        lambda: cg.XmonDevice(cirq.Duration(nanos=1), cirq.Duration(nanos=1),
                              cirq.Duration(nanos=1), []))


def test_xmon_device_str():
    assert str(square_device(2, 2)).strip() == """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip()


def test_at():
    d = square_device(3, 3)
    assert d.at(-1, -1) is None
    assert d.at(0, 0) == cirq.GridQubit(0, 0)

    assert d.at(-1, 1) is None
    assert d.at(0, 1) == cirq.GridQubit(0, 1)
    assert d.at(1, 1) == cirq.GridQubit(1, 1)
    assert d.at(2, 1) == cirq.GridQubit(2, 1)
    assert d.at(3, 1) is None

    assert d.at(1, -1) is None
    assert d.at(1, 0) == cirq.GridQubit(1, 0)
    assert d.at(1, 1) == cirq.GridQubit(1, 1)
    assert d.at(1, 2) == cirq.GridQubit(1, 2)
    assert d.at(1, 3) is None


def test_row_and_col():
    d = square_device(2, 3)
    assert d.col(-1) == []
    assert d.col(0) == [cirq.GridQubit(0, 0),
                        cirq.GridQubit(1, 0),
                        cirq.GridQubit(2, 0)]
    assert d.col(1) == [cirq.GridQubit(0, 1),
                        cirq.GridQubit(1, 1),
                        cirq.GridQubit(2, 1)]
    assert d.col(2) == []
    assert d.col(5000) == []

    assert d.row(-1) == []
    assert d.row(0) == [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    assert d.row(1) == [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]
    assert d.row(2) == [cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)]
    assert d.row(3) == []

    b = cg.Bristlecone
    assert b.col(0) == [cirq.GridQubit(5, 0)]
    assert b.row(0) == [cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)]
    assert b.col(1) == [cirq.GridQubit(4, 1),
                        cirq.GridQubit(5, 1),
                        cirq.GridQubit(6, 1)]
