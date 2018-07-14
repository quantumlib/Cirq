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
from cirq.circuits.circuit import _operation_to_unitary_matrix
from cirq import Circuit, InsertStrategy, Moment


def test_equality():
    a = cirq.QubitId()
    b = cirq.QubitId()

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(Circuit(),
                          Circuit([]), Circuit(()))
    eq.add_equality_group(
        Circuit([Moment()]),
        Circuit((Moment(),)))

    # Equality depends on structure and contents.
    eq.add_equality_group(Circuit([Moment([cirq.X(a)])]))
    eq.add_equality_group(Circuit([Moment([cirq.X(b)])]))
    eq.add_equality_group(
        Circuit(
            [Moment([cirq.X(a)]),
             Moment([cirq.X(b)])]))
    eq.add_equality_group(
        Circuit([Moment([cirq.X(a), cirq.X(b)])]))

    # Big case.
    eq.add_equality_group(
        Circuit([
            Moment([cirq.H(a), cirq.H(b)]),
            Moment([cirq.CZ(a, b)]),
            Moment([cirq.H(b)]),
        ]))
    eq.add_equality_group(
        Circuit([
            Moment([cirq.H(a)]),
            Moment([cirq.CNOT(a, b)]),
        ]))


def test_append_single():
    a = cirq.QubitId()

    c = Circuit()
    c.append(())
    assert c == Circuit()

    c = Circuit()
    c.append(cirq.X(a))
    assert c == Circuit([Moment([cirq.X(a)])])

    c = Circuit()
    c.append([cirq.X(a)])
    assert c == Circuit([Moment([cirq.X(a)])])


def test_append_multiple():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    c.append([cirq.X(a), cirq.X(b)], InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(b)])
    ])

    c = Circuit()
    c.append([cirq.X(a), cirq.X(b)], InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
    ])

    c = Circuit()
    c.append(cirq.X(a), InsertStrategy.EARLIEST)
    c.append(cirq.X(b), InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
    ])


@cirq.testing.only_test_in_python3
def test_repr():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit([
        Moment([cirq.H(a)]),
        Moment([cirq.CZ(a, b)]),
    ])
    assert repr(c) == """
Circuit([
    Moment((GateOperation(H, (NamedQubit('a'),)),)),
    Moment((GateOperation(CZ, (NamedQubit('a'), NamedQubit('b'))),))])
    """.strip()


def test_slice():
    a = cirq.QubitId()
    b = cirq.QubitId()
    c = Circuit([
        Moment([cirq.H(a), cirq.H(b)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.H(b)]),
    ])
    assert c[0:1] == Circuit([Moment([cirq.H(a), cirq.H(b)])])
    assert c[::2] == Circuit([
        Moment([cirq.H(a), cirq.H(b)]), Moment([cirq.H(b)])
    ])
    assert c[0:1:2] == Circuit([Moment([cirq.H(a), cirq.H(b)])])
    assert c[1:3:] == Circuit([Moment([cirq.CZ(a, b)]), Moment([cirq.H(b)])])
    assert c[::-1] == Circuit([Moment([cirq.H(b)]), Moment([cirq.CZ(a, b)]),
                               Moment([cirq.H(a), cirq.H(b)])])
    assert c[3:0:-1] == Circuit([Moment([cirq.H(b)]), Moment([cirq.CZ(a, b)])])
    assert c[0:2:-1] == Circuit()


def test_concatenate():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    d = Circuit([Moment([cirq.X(b)])])
    e = Circuit([Moment([cirq.X(a), cirq.X(b)])])

    assert c + d == Circuit([Moment([cirq.X(b)])])
    assert d + c == Circuit([Moment([cirq.X(b)])])
    assert e + d == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
        Moment([cirq.X(b)])
    ])

    d += c
    assert d == Circuit([Moment([cirq.X(b)])])

    c += d
    assert c == Circuit([Moment([cirq.X(b)])])

    f = e + d
    f += e
    assert f == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(a), cirq.X(b)])
    ])

    with pytest.raises(TypeError):
        _ = c + 'a'
    with pytest.raises(TypeError):
        c += 'a'


def test_multiply():
    a = cirq.QubitId()

    c = Circuit()
    d = Circuit([Moment([cirq.X(a)])])

    assert c * 0 == Circuit()
    assert d * 0 == Circuit()
    assert d * 2 == Circuit([Moment([cirq.X(a)]),
                             Moment([cirq.X(a)])])
    assert 1 * c == Circuit()
    assert -1 * d == Circuit()
    assert 1 * d == Circuit([Moment([cirq.X(a)])])

    d *= 3
    assert d == Circuit([Moment([cirq.X(a)]),
                         Moment([cirq.X(a)]),
                         Moment([cirq.X(a)])])

    with pytest.raises(TypeError):
        _ = c * 'a'
    with pytest.raises(TypeError):
        _ = 'a' * c
    with pytest.raises(TypeError):
        c *= 'a'


def test_container_methods():
    a = cirq.QubitId()
    b = cirq.QubitId()
    c = Circuit([
        Moment([cirq.H(a), cirq.H(b)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.H(b)]),
    ])
    assert list(c) == list(c._moments)
    # __iter__
    assert list(iter(c)) == list(c._moments)
    # __reversed__ for free.
    assert list(reversed(c)) == list(reversed(c._moments))
    # __contains__ for free.
    assert Moment([cirq.H(b)]) in c

    assert len(c) == 3


def test_bad_index():
    a = cirq.QubitId()
    b = cirq.QubitId()
    c = Circuit([Moment([cirq.H(a), cirq.H(b)])])
    with pytest.raises(TypeError):
        _ = c['string']

def test_append_strategies():
    a = cirq.QubitId()
    b = cirq.QubitId()
    stream = [cirq.X(a), cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(a)]

    c = Circuit()
    c.append(stream, InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(b), cirq.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b), cirq.X(a)]),
        Moment([cirq.X(b)]),
    ])


def test_insert():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()

    c.insert(0, ())
    assert c == Circuit()

    with pytest.raises(IndexError):
        c.insert(-1, ())
    with pytest.raises(IndexError):
        c.insert(1, ())

    c.insert(0, [cirq.X(a), cirq.CZ(a, b), cirq.X(b)])
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
    ])

    with pytest.raises(IndexError):
        c.insert(550, ())

    c.insert(1, cirq.H(b), strategy=InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.H(b)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
    ])

    c.insert(0, cirq.H(b), strategy=InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a), cirq.H(b)]),
        Moment([cirq.H(b)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
    ])


def test_insert_inline_near_start():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
    ])

    c.insert(1, cirq.X(a), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment(),
    ])

    c.insert(1, cirq.Y(a), strategy=InsertStrategy.INLINE)
    assert c ==Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.Y(a)]),
        Moment(),
    ])

    c.insert(0, cirq.Z(b), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.Z(b)]),
        Moment([cirq.X(a)]),
        Moment([cirq.Y(a)]),
        Moment(),
    ])

def test_insert_into_range():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    c = Circuit([Moment([cirq.X(x)])] * 4)
    c.insert_into_range([cirq.Z(x), cirq.CZ(x, y)], 2, 2)
    actual_text_diagram = c.to_text_diagram().strip()
    expected_text_diagram = """
x: ───X───X───Z───@───X───X───
                  │
y: ───────────────@───────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_next_moment_operating_on():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    assert c.next_moment_operating_on([a]) is None
    assert c.next_moment_operating_on([a], 0) is None
    assert c.next_moment_operating_on([a], 102) is None

    c = Circuit([Moment([cirq.X(a)])])
    assert c.next_moment_operating_on([a]) == 0
    assert c.next_moment_operating_on([a], 0) == 0
    assert c.next_moment_operating_on([a, b]) == 0
    assert c.next_moment_operating_on([a], 1) is None
    assert c.next_moment_operating_on([b]) is None

    c = Circuit([
        Moment(),
        Moment([cirq.X(a)]),
        Moment(),
        Moment([cirq.CZ(a, b)])
    ])

    assert c.next_moment_operating_on([a], 0) == 1
    assert c.next_moment_operating_on([a], 1) == 1
    assert c.next_moment_operating_on([a], 2) == 3
    assert c.next_moment_operating_on([a], 3) == 3
    assert c.next_moment_operating_on([a], 4) is None

    assert c.next_moment_operating_on([b], 0) == 3
    assert c.next_moment_operating_on([b], 1) == 3
    assert c.next_moment_operating_on([b], 2) == 3
    assert c.next_moment_operating_on([b], 3) == 3
    assert c.next_moment_operating_on([b], 4) is None

    assert c.next_moment_operating_on([a, b], 0) == 1
    assert c.next_moment_operating_on([a, b], 1) == 1
    assert c.next_moment_operating_on([a, b], 2) == 3
    assert c.next_moment_operating_on([a, b], 3) == 3
    assert c.next_moment_operating_on([a, b], 4) is None


def test_next_moment_operating_on_distance():
    a = cirq.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment([cirq.X(a)]),
        Moment(),
    ])

    assert c.next_moment_operating_on([a], 0, max_distance=4) is None
    assert c.next_moment_operating_on([a], 1, max_distance=3) is None
    assert c.next_moment_operating_on([a], 2, max_distance=2) is None
    assert c.next_moment_operating_on([a], 3, max_distance=1) is None
    assert c.next_moment_operating_on([a], 4, max_distance=0) is None

    assert c.next_moment_operating_on([a], 0, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=4) == 4
    assert c.next_moment_operating_on([a], 2, max_distance=3) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=2) == 4
    assert c.next_moment_operating_on([a], 4, max_distance=1) == 4

    assert c.next_moment_operating_on([a], 5, max_distance=0) is None
    assert c.next_moment_operating_on([a], 1, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=500) == 4

    # Huge max distances should be handled quickly due to capping.
    assert c.next_moment_operating_on([a], 5, max_distance=10**100) is None


def test_prev_moment_operating_on():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    assert c.prev_moment_operating_on([a]) is None
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([a], 102) is None

    c = Circuit([Moment([cirq.X(a)])])
    assert c.prev_moment_operating_on([a]) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a, b]) == 0
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([b]) is None

    c = Circuit([
        Moment([cirq.CZ(a, b)]),
        Moment(),
        Moment([cirq.X(a)]),
        Moment(),
    ])

    assert c.prev_moment_operating_on([a], 4) == 2
    assert c.prev_moment_operating_on([a], 3) == 2
    assert c.prev_moment_operating_on([a], 2) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a], 0) is None

    assert c.prev_moment_operating_on([b], 4) == 0
    assert c.prev_moment_operating_on([b], 3) == 0
    assert c.prev_moment_operating_on([b], 2) == 0
    assert c.prev_moment_operating_on([b], 1) == 0
    assert c.prev_moment_operating_on([b], 0) is None

    assert c.prev_moment_operating_on([a, b], 4) == 2
    assert c.prev_moment_operating_on([a, b], 3) == 2
    assert c.prev_moment_operating_on([a, b], 2) == 0
    assert c.prev_moment_operating_on([a, b], 1) == 0
    assert c.prev_moment_operating_on([a, b], 0) is None


def test_prev_moment_operating_on_distance():
    a = cirq.QubitId()

    c = Circuit([
        Moment(),
        Moment([cirq.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
    ])

    assert c.prev_moment_operating_on([a], max_distance=4) is None
    assert c.prev_moment_operating_on([a], 6, max_distance=4) is None
    assert c.prev_moment_operating_on([a], 5, max_distance=3) is None
    assert c.prev_moment_operating_on([a], 4, max_distance=2) is None
    assert c.prev_moment_operating_on([a], 3, max_distance=1) is None
    assert c.prev_moment_operating_on([a], 2, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 1, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 0, max_distance=0) is None

    assert c.prev_moment_operating_on([a], 6, max_distance=5) == 1
    assert c.prev_moment_operating_on([a], 5, max_distance=4) == 1
    assert c.prev_moment_operating_on([a], 4, max_distance=3) == 1
    assert c.prev_moment_operating_on([a], 3, max_distance=2) == 1
    assert c.prev_moment_operating_on([a], 2, max_distance=1) == 1

    assert c.prev_moment_operating_on([a], 6, max_distance=10) == 1
    assert c.prev_moment_operating_on([a], 6, max_distance=100) == 1
    assert c.prev_moment_operating_on([a], 13, max_distance=500) == 1

    # Huge max distances should be handled quickly due to capping.
    assert c.prev_moment_operating_on([a], 1, max_distance=10**100) is None


def test_operation_at():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, -1) is None
    assert c.operation_at(a, 102) is None

    c = Circuit([Moment()])
    assert c.operation_at(a, 0) is None

    c = Circuit([Moment([cirq.X(a)])])
    assert c.operation_at(b, 0) is None
    assert c.operation_at(a, 1) is None
    assert c.operation_at(a, 0) == cirq.X(a)

    c = Circuit([Moment(), Moment([cirq.CZ(a, b)])])
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, 1) == cirq.CZ(a, b)


def test_findall_operations():
    a = cirq.QubitId()
    b = cirq.QubitId()

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    za = cirq.Z.on(a)
    zb = cirq.Z.on(b)

    def is_x(op: cirq.Operation) -> bool:
        return (isinstance(op, cirq.GateOperation) and
                isinstance(op.gate, cirq.RotXGate))

    c = Circuit()
    assert list(c.findall_operations(is_x)) == []

    c = Circuit.from_ops(xa)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = Circuit.from_ops(za)
    assert list(c.findall_operations(is_x)) == []

    c = Circuit.from_ops([za, zb] * 8)
    assert list(c.findall_operations(is_x)) == []

    c = Circuit.from_ops(xa, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (0, xb)]

    c = Circuit.from_ops(xa, zb)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = Circuit.from_ops(xa, za)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = Circuit.from_ops([xa] * 8)
    assert list(c.findall_operations(is_x)) == list(enumerate([xa] * 8))

    c = Circuit.from_ops(za, zb, xa, xb)
    assert list(c.findall_operations(is_x)) == [(1, xa), (1, xb)]

    c = Circuit.from_ops(xa, zb, za, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (1, xb)]


def test_findall_operations_with_gate():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.Z(a), cirq.Z(b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.measure(a), cirq.measure(b)]),
    ])
    assert list(c.findall_operations_with_gate_type(cirq.RotXGate)) == [
        (0, cirq.X(a), cirq.X),
        (2, cirq.X(a), cirq.X),
        (2, cirq.X(b), cirq.X),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.Rot11Gate)) == [
        (3, cirq.CZ(a, b), cirq.CZ),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.MeasurementGate)) == [
        (4, cirq.MeasurementGate(key='a').on(a), cirq.MeasurementGate(key='a')),
        (4, cirq.MeasurementGate(key='b').on(b), cirq.MeasurementGate(key='b')),
    ]


def test_are_all_measurements_terminal():
    a = cirq.QubitId()
    b = cirq.QubitId()

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ma = cirq.MeasurementGate().on(a)
    mb = cirq.MeasurementGate().on(b)

    c = Circuit()
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(xa, xb)
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(ma)
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(ma, mb)
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(xa, ma)
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(xa, ma, xb, mb)
    assert c.are_all_measurements_terminal()

    c = Circuit.from_ops(ma, xa)
    assert not c.are_all_measurements_terminal()

    c = Circuit.from_ops(ma, xa, mb)
    assert not c.are_all_measurements_terminal()

    c = Circuit.from_ops(xa, ma, xb, xa)
    assert not c.are_all_measurements_terminal()

    c = Circuit.from_ops(ma, ma)
    assert not c.are_all_measurements_terminal()

    c = Circuit.from_ops(xa, ma, xa)
    assert not c.are_all_measurements_terminal()


def test_clear_operations_touching():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit()
    c.clear_operations_touching([a, b], range(10))
    assert c == Circuit()

    c = Circuit([
        Moment(),
        Moment([cirq.X(a), cirq.X(b)]),
        Moment([cirq.X(a)]),
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment(),
        Moment([cirq.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment([cirq.X(b)]),
        Moment([cirq.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment([cirq.X(b)]),
        Moment(),
    ])

    c = Circuit([
        Moment(),
        Moment([cirq.X(a), cirq.X(b)]),
        Moment([cirq.X(a)]),
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment(),
        Moment([cirq.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a, b], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment(),
        Moment([cirq.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
    ])


def test_all_qubits():
    a = cirq.QubitId()
    b = cirq.QubitId()

    c = Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(b)]),
    ])
    assert c.all_qubits() == {a, b}

    c = Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(a)]),
    ])
    assert c.all_qubits() == {a}

    c = Circuit([
        Moment([cirq.CZ(a, b)]),
    ])
    assert c.all_qubits() == {a, b}

    c = Circuit([
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(a)])
    ])
    assert c.all_qubits() == {a, b}


def test_all_operations():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(b)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(a)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(a)]

    c = Circuit([
        Moment([cirq.CZ(a, b)]),
    ])
    assert list(c.all_operations()) == [cirq.CZ(a, b)]

    c = Circuit([
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(a)])
    ])
    assert list(c.all_operations()) == [cirq.CZ(a, b), cirq.X(a)]

    c = Circuit([
            Moment([]),
            Moment([cirq.X(a), cirq.Y(b)]),
            Moment([]),
            Moment([cirq.CNOT(a, b)]),
            Moment([cirq.Z(b), cirq.H(a)]),  # Different qubit order
            Moment([])])

    assert list(c.all_operations()) == [
        cirq.X(a),
        cirq.Y(b),
        cirq.CNOT(a, b),
        cirq.Z(b),
        cirq.H(a)
    ]


def test_from_ops():
    a = cirq.QubitId()
    b = cirq.QubitId()

    actual = Circuit.from_ops(
        cirq.X(a),
        [cirq.Y(a), cirq.Z(b)],
        cirq.CZ(a, b),
        cirq.X(a),
        [cirq.Z(b), cirq.Y(a)],
    )

    assert actual == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.Y(a), cirq.Z(b)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(a), cirq.Z(b)]),
        Moment([cirq.Y(a)]),
    ])


def test_to_text_diagram_teleportation_to_diagram():
    ali = cirq.NamedQubit('(0, 0)')
    bob = cirq.NamedQubit('(0, 1)')
    msg = cirq.NamedQubit('(1, 0)')
    tmp = cirq.NamedQubit('(1, 1)')

    c = Circuit([
        Moment([cirq.H(ali)]),
        Moment([cirq.CNOT(ali, bob)]),
        Moment([cirq.X(msg)**0.5]),
        Moment([cirq.CNOT(msg, ali)]),
        Moment([cirq.H(msg)]),
        Moment(
            [cirq.measure(msg),
             cirq.measure(ali)]),
        Moment([cirq.CNOT(ali, bob)]),
        Moment([cirq.CNOT(msg, tmp)]),
        Moment([cirq.CZ(bob, tmp)]),
    ])

    assert c.to_text_diagram().strip() == """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────@───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───@───
    """.strip()
    assert c.to_text_diagram(use_unicode_characters=False).strip() == """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------@---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---@---
        """.strip()

    assert c.to_text_diagram(transpose=True,
                             use_unicode_characters=False).strip() == """
(0, 0) (0, 1) (1, 0) (1, 1)
|      |      |      |
H      |      |      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      X^0.5  |
|      |      |      |
X-------------@      |
|      |      |      |
|      |      H      |
|      |      |      |
M      |      M      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      @------X
|      |      |      |
|      @-------------@
|      |      |      |
        """.strip()


def test_diagram_with_unknown_exponent():
    class WeirdGate(cirq.Gate, cirq.TextDiagrammable):
        def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                              ) -> cirq.TextDiagramInfo:
            return cirq.TextDiagramInfo(wire_symbols=('B',),
                                        exponent='fancy')

    class WeirderGate(cirq.Gate, cirq.TextDiagrammable):
        def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                              ) -> cirq.TextDiagramInfo:
            return cirq.TextDiagramInfo(wire_symbols=('W',),
                                        exponent='fancy-that')

    c = cirq.Circuit.from_ops(
        WeirdGate().on(cirq.NamedQubit('q')),
        WeirderGate().on(cirq.NamedQubit('q')),
    )

    # The hyphen in the exponent should cause parens to appear.
    assert c.to_text_diagram() == 'q: ───B^fancy───W^(fancy-that)───'


def test_to_text_diagram_extended_gate():
    q = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')

    class FGate(cirq.Gate):
        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'

    f = FGate()
    c = Circuit([
        Moment([f.on(q)]),
    ])

    # Fallback to repr without extension.
    diagram = Circuit([
        Moment([f.on(q)]),
    ]).to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
(0, 0): ---python-object-FGate:arbitrary-digits---
        """.strip()

    # When used on multiple qubits, show the qubit order as a digit suffix.
    diagram = Circuit([
        Moment([f.on(q, q3, q2)]),
    ]).to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
(0, 0): ---python-object-FGate:arbitrary-digits:0---
           |
(0, 1): ---python-object-FGate:arbitrary-digits:2---
           |
(0, 2): ---python-object-FGate:arbitrary-digits:1---
            """.strip()

    # Succeeds with extension.
    class FGateAsText(cirq.Gate, cirq.TextDiagrammable):
        def __init__(self, f_gate):
            self.f_gate = f_gate

        def text_diagram_info(self, args: cirq.TextDiagramInfoArgs):
            return cirq.TextDiagramInfo(('F',))

    ext = cirq.Extensions()
    ext.add_cast(cirq.TextDiagrammable, FGate, FGateAsText)
    diagram = c.to_text_diagram(ext, use_unicode_characters=False)

    assert diagram.strip() == """
(0, 0): ---F---
        """.strip()


def test_to_text_diagram_multi_qubit_gate():
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = Circuit.from_ops(cirq.measure(q1, q2, q3, key='msg'))
    assert c.to_text_diagram().strip() == """
(0, 0): ───M───
           │
(0, 1): ───M───
           │
(0, 2): ───M───
    """.strip()
    assert c.to_text_diagram(use_unicode_characters=False).strip() == """
(0, 0): ---M---
           |
(0, 1): ---M---
           |
(0, 2): ---M---
    """.strip()
    assert c.to_text_diagram(transpose=True).strip() == """
(0, 0) (0, 1) (0, 2)
│      │      │
M──────M──────M
│      │      │
    """.strip()


def test_to_text_diagram_many_qubits_gate_but_multiple_wire_symbols():
    class BadGate(cirq.Gate, cirq.TextDiagrammable):
        def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                              ) -> cirq.TextDiagramInfo:
            return cirq.TextDiagramInfo(wire_symbols=('a', 'a'))
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = Circuit([Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()


def test_to_text_diagram_parameterized_value():
    q = cirq.NamedQubit('cube')

    class PGate(cirq.Gate, cirq.TextDiagrammable):
        def __init__(self, val):
            self.val = val

        def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                              ) -> cirq.TextDiagramInfo:
            return cirq.TextDiagramInfo(('P',), self.val)

    c = Circuit.from_ops(
        PGate(1).on(q),
        PGate(2).on(q),
        PGate(cirq.Symbol('a')).on(q),
        PGate(cirq.Symbol('%$&#*(')).on(q),
    )
    assert str(c).strip() == 'cube: ───P───P^2───P^a───P^Symbol("%$&#*(")───'


def test_to_text_diagram_custom_order():
    qa = cirq.NamedQubit('2')
    qb = cirq.NamedQubit('3')
    qc = cirq.NamedQubit('4')

    c = Circuit([Moment([cirq.X(qa), cirq.X(qb), cirq.X(qc)])])
    diagram = c.to_text_diagram(
        qubit_order=cirq.QubitOrder.sorted_by(lambda e: int(str(e)) % 3),
        use_unicode_characters=False)
    assert diagram.strip() == """
3: ---X---

4: ---X---

2: ---X---
    """.strip()


def test_overly_precise_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.12345678])])
    diagram = c.to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
a: ---X^0.123---
    """.strip()


def test_none_precision_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.4921875])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=None)
    assert diagram.strip() == """
a: ---X^0.4921875---
    """.strip()


def test_diagram_custom_precision():
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.12341234])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=5)
    assert diagram.strip() == """
a: ---X^0.12341---
    """.strip()


def test_diagram_wgate():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.google.ExpWGate(
        half_turns=0.12341234, axis_half_turns=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=2)
    assert diagram.strip() == """
a: ---W(0.43)^0.12---
    """.strip()


def test_diagram_wgate_none_precision():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.google.ExpWGate(
        half_turns=0.12341234, axis_half_turns=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=None)
    assert diagram.strip() == """
a: ---W(0.43214321)^0.12341234---
    """.strip()


def test_text_diagram_jupyter():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = Circuit.from_ops((
                cirq.CNOT(a, b),
                cirq.CNOT(b, c),
                cirq.CNOT(c, a)) * 50)
    text_expected = circuit.to_text_diagram()

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''
        def text(self, to_print):
            self.text_pretty += to_print
    p = FakePrinter()
    circuit._repr_pretty_(p, False)
    assert p.text_pretty == text_expected

    # Test cycle handling
    p = FakePrinter()
    circuit._repr_pretty_(p, True)
    assert p.text_pretty == 'Circuit(...)'

    # Test Jupyter notebook html output
    text_html = circuit._repr_html_()
    # Don't enforce specific html surrounding the diagram content
    assert text_expected in text_html


def test_operation_to_unitary_matrix():
    ex = cirq.Extensions()
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    m = _operation_to_unitary_matrix(cirq.X(a),
                                     {a: 0},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([[0, 1],
                  [1, 0]]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(cirq.X(a),
                                     {a: 0, b: 1},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.kron(cirq.X.matrix(), np.eye(2)))
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(cirq.X(a),
                                     {a: 1, b: 0},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(cirq.CNOT(b, a),
                                     {a: 0, b: 1},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(cirq.CNOT(b, a),
                                     {a: 1, b: 0},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]),
        atol=1e-8)


def test_circuit_to_unitary_matrix():
    # Single qubit gates.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit.from_ops(cirq.Z(a), cirq.X(b))
    m = c.to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, -1, 0],
        ]),
        atol=1e-8)

    # Single qubit gates and two qubit gate.
    c = Circuit.from_ops(cirq.Z(a), cirq.X(b), cirq.CNOT(a, b))
    m = c.to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8)

    # Measurement gate has no corresponding matrix.
    c = Circuit.from_ops(cirq.measure(a))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix(ignore_terminal_measurements=False)

    # Ignoring terminal measurements.
    c = Circuit.from_ops(cirq.measure(a))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        np.eye(2))

    # Ignoring terminal measurements with further cirq.
    c = Circuit.from_ops(cirq.Z(a), cirq.measure(a), cirq.Z(b))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]))

    # Optionally don't ignoring terminal measurements.
    c = Circuit.from_ops(cirq.measure(a))
    with pytest.raises(TypeError, match="Terminal"):
        c.to_unitary_matrix(ignore_terminal_measurements=False)

    # Non-terminal measurements are not ignored.
    c = Circuit.from_ops(cirq.measure(a), cirq.X(a))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix()

    # Non-terminal measurements are not ignored (multiple qubits).
    c = Circuit.from_ops(
            cirq.measure(a),
            cirq.measure(b),
            cirq.CNOT(a, b))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix()

    # Gates without matrix or decomposition raise exception
    class MysteryGate(cirq.Gate):
        pass
    c = Circuit.from_ops(MysteryGate()(a, b))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix()


def test_simple_circuits_to_unitary_matrix():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Phase parity.
    c = Circuit.from_ops(cirq.CNOT(a, b), cirq.Z(b), cirq.CNOT(a, b))
    m = c.to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8)

    # 2-qubit matrix matches when qubits in order.
    for expected in [np.diag([1, 1j, -1, -1j]), cirq.CNOT.matrix()]:

        class Passthrough(cirq.Gate, cirq.KnownMatrix):
            def matrix(self):
                return expected

        c = Circuit.from_ops(Passthrough()(a, b))
        m = c.to_unitary_matrix()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected)


def test_composite_gate_to_unitary_matrix():
    class CNOT_composite(cirq.Gate, cirq.CompositeGate):
        def default_decompose(self, qubits):
            q0, q1 = qubits
            return cirq.Y(q1)**-0.5, cirq.CZ(q0, q1), cirq.Y(q1)**0.5

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit.from_ops(
            cirq.X(a),
            CNOT_composite()(a, b),
            cirq.X(a),
            cirq.measure(a),
            cirq.X(b),
            cirq.measure(b))
    mat = c.to_unitary_matrix()
    mat_expected = cirq.CNOT.matrix()

    cirq.testing.assert_allclose_up_to_global_phase(mat, mat_expected)


def test_expanding_gate_symbols():
    class MultiTargetCZ(cirq.Gate, cirq.TextDiagrammable):
        def text_diagram_info(self,
                              args: cirq.TextDiagramInfoArgs
                              ) -> cirq.TextDiagramInfo:
            assert args.known_qubit_count is not None
            return cirq.TextDiagramInfo(
                ('@',) + ('Z',) * (args.known_qubit_count - 1))

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    t0 = cirq.Circuit.from_ops(MultiTargetCZ().on(c))
    t1 = cirq.Circuit.from_ops(MultiTargetCZ().on(c, a))
    t2 = cirq.Circuit.from_ops(MultiTargetCZ().on(c, a, b))

    assert t0.to_text_diagram().strip() == """
c: ───@───
    """.strip()

    assert t1.to_text_diagram().strip() == """
a: ───Z───
      │
c: ───@───
    """.strip()

    assert t2.to_text_diagram().strip() == """
a: ───Z───
      │
b: ───Z───
      │
c: ───@───
    """.strip()


def test_transposed_diagram_exponent_order():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(
        cirq.CZ(a, b)**-0.5,
        cirq.CZ(a, c)**0.5,
        cirq.CZ(b, c)**0.125,
    )
    assert circuit.to_text_diagram(transpose=True).strip() == """
0 1      2
│ │      │
@─@^-0.5 │
│ │      │
@─┼──────@^0.5
│ │      │
│ @──────@^0.125
│ │      │
    """.strip()


def test_insert_moments():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit()

    m0 = cirq.Moment([cirq.X(q)])
    c.append(m0)
    assert list(c) == [m0]
    assert c[0] is m0

    m1 = cirq.Moment([cirq.Y(q)])
    c.append(m1)
    assert list(c) == [m0, m1]
    assert c[1] is m1

    m2 = cirq.Moment([cirq.Z(q)])
    c.insert(0, m2)
    assert list(c) == [m2, m0, m1]
    assert c[0] is m2


def test_items():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit()
    m1 = cirq.Moment([cirq.X(a), cirq.X(b)])
    m2 = cirq.Moment([cirq.X(a)])
    m3 = cirq.Moment([])
    m4 = cirq.Moment([cirq.CZ(a, b)])

    c[:] = [m1, m2]
    assert c == cirq.Circuit([m1, m2])

    assert c[0] == m1
    del c[0]
    assert c == cirq.Circuit([m2])

    c.append(m1)
    c.append(m3)
    assert c == cirq.Circuit([m2, m1, m3])

    assert c[0:2] == Circuit([m2, m1])
    c[0:2] = [m4]
    assert c == cirq.Circuit([m4, m3])

    c[:] = [m1]
    assert c == cirq.Circuit([m1])

    with pytest.raises(TypeError):
        c[:] = [m1, 1]
    with pytest.raises(TypeError):
        c[0] = 1


def test_copy():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit.from_ops(cirq.X(a), cirq.CZ(a, b), cirq.Z(a), cirq.Z(b))
    assert c == c.copy() == c.__copy__() == c.__deepcopy__()
    c2 = c.copy()
    assert c2 == c
    c2[:] = []
    assert c2 != c
