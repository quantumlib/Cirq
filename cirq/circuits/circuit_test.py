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
from typing import Tuple

from collections import defaultdict
from random import randint, random, sample, randrange

import numpy as np
import pytest

import cirq
from cirq.circuits.optimization_pass import (PointOptimizer,
                                             PointOptimizationSummary)
from cirq import Circuit, InsertStrategy, Moment
from cirq.testing import random_circuit
import cirq.google as cg


def test_equality():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(Circuit(),
                          Circuit(device=cirq.UnconstrainedDevice),
                          Circuit([]), Circuit(()))
    eq.add_equality_group(
        Circuit([Moment()]),
        Circuit((Moment(),)))
    eq.add_equality_group(Circuit(device=cg.Foxtail))

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
    a = cirq.NamedQubit('a')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.X(b)])
    ])

    c = Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
    ])

    c = Circuit()
    c.append(cirq.X(a), cirq.InsertStrategy.EARLIEST)
    c.append(cirq.X(b), cirq.InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a), cirq.X(b)]),
    ])


@cirq.testing.only_test_in_python3
def test_repr():
    assert repr(cirq.Circuit()) == 'cirq.Circuit()'

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit([
        Moment([cirq.H(a), cirq.H(b)]),
        Moment(),
        Moment([cirq.CZ(a, b)]),
    ])
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == """cirq.Circuit(moments=[
    cirq.Moment(operations=[
        cirq.H.on(cirq.NamedQubit('a')),
        cirq.H.on(cirq.NamedQubit('b')),
    ]),
    cirq.Moment(),
    cirq.Moment(operations=[
        cirq.CZ.on(cirq.NamedQubit('a'), cirq.NamedQubit('b')),
    ]),
])"""

    c = Circuit(device=cg.Foxtail)
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == 'cirq.Circuit(device=cirq.google.Foxtail)'

    c = Circuit.from_ops(
        cirq.Z(cirq.GridQubit(0, 0)),
        device=cg.Foxtail)
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == """cirq.Circuit(moments=[
    cirq.Moment(operations=[
        cirq.Z.on(cirq.GridQubit(0, 0)),
    ]),
], device=cirq.google.Foxtail)"""


def test_slice():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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


def test_concatenate_with_device():
    fox = cirq.Circuit(device=cg.Foxtail)
    cone = cirq.Circuit(device=cg.Bristlecone)
    unr = cirq.Circuit()

    _ = cone + cone
    _ = cone + unr
    _ = unr + cone
    cone += unr
    with pytest.raises(ValueError):
        _ = cone + fox
    with pytest.raises(ValueError):
        unr += cone
    with pytest.raises(ValueError):
        cone += fox

    unr.append(cirq.X(cirq.NamedQubit('not_allowed')))
    with pytest.raises(ValueError):
        cone += unr
    with pytest.raises(ValueError):
        _ = cone + unr
    assert len(cone) == 0


def test_with_device():
    c = cirq.Circuit.from_ops(cirq.X(cirq.LineQubit(0)))
    c2 = c.with_device(cg.Foxtail,
                                lambda e: cirq.GridQubit(e.x, 0))
    assert c2 == cirq.Circuit.from_ops(
        cirq.X(cirq.GridQubit(0, 0)),
        device=cg.Foxtail)

    # Qubit type must be correct.
    c = cirq.Circuit.from_ops(cirq.X(cirq.LineQubit(0)))
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        _ = c.with_device(cg.Foxtail)

    # Operations must be compatible from the start
    c = cirq.Circuit.from_ops(cirq.X(cirq.GridQubit(0, 0)))
    _ = c.with_device(cg.Foxtail)
    c = cirq.Circuit.from_ops(cirq.H(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError, match='Unsupported gate type'):
        _ = c.with_device(cg.Foxtail)

    # Some qubits exist on multiple devices.
    c = cirq.Circuit.from_ops(cirq.X(cirq.GridQubit(0, 0)), device=cg.Foxtail)
    with pytest.raises(ValueError):
        _ = c.with_device(cg.Bristlecone)
    c = cirq.Circuit.from_ops(cirq.X(cirq.GridQubit(0, 6)), device=cg.Foxtail)
    _ = c.with_device(cg.Bristlecone)


def test_set_device():
    c = cirq.Circuit.from_ops(cirq.X(cirq.LineQubit(0)))
    assert c.device is cirq.UnconstrainedDevice

    with pytest.raises(ValueError):
        c.device = cg.Foxtail
    assert c.device is cirq.UnconstrainedDevice

    c[:] = []
    c.append(cirq.X(cirq.GridQubit(0, 0)))
    c.device = cg.Foxtail
    assert c.device == cg.Foxtail


def test_multiply():
    a = cirq.NamedQubit('a')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit([Moment([cirq.H(a), cirq.H(b)])])
    with pytest.raises(TypeError):
        _ = c['string']

def test_append_strategies():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    stream = [cirq.X(a), cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(a)]

    c = Circuit()
    c.append(stream, cirq.InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(a)]),
    ])

    c = Circuit()
    c.append(stream, cirq.InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b)]),
        Moment([cirq.X(b), cirq.X(a)]),
    ])

    c = Circuit()
    c.append(stream, cirq.InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.X(b), cirq.X(a)]),
        Moment([cirq.X(b)]),
    ])


def test_insert_op_tree_new():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = Circuit()

    op_tree_list = [(-10, 0, cirq.CZ(a, b), a),
                    (-20, 0, cirq.X(a), a),
                    (20, 2, cirq.X(b), b),
                    (2, 2, cirq.H(b), b),
                    (-3, 1, cirq.H(a), a)]

    for given_index, actual_index, operation, qubit in op_tree_list:
        c.insert(given_index, operation, InsertStrategy.NEW)
        assert c.operation_at(qubit, actual_index) == operation

    c.insert(1, (), InsertStrategy.NEW)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.H(a)]),
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.H(b)]),
        Moment([cirq.X(b)]),
    ])


def test_insert_op_tree_newinline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = Circuit()

    op_tree_list = [(-5, 0, [cirq.H(a), cirq.X(b)], [a, b]),
                    (-15, 0, [cirq.CZ(a, b)], [a]),
                    (15, 2, [cirq.H(b), cirq.X(a)], [b, a])]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, InsertStrategy.NEW_THEN_INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]

    c2 = Circuit()
    c2.insert(0, [cirq.CZ(a, b), cirq.H(a), cirq.X(b), cirq.H(b), cirq.X(a)],
              InsertStrategy.NEW_THEN_INLINE)
    assert c == c2


def test_insert_op_tree_inline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = Circuit([Moment([cirq.H(a)])])

    op_tree_list = [(1, 1, [cirq.H(a), cirq.X(b)], [a, b]),
                    (0, 0, [cirq.X(b)], [b]),
                    (4, 3, [cirq.H(b)], [b]),
                    (5, 3, [cirq.H(a)], [a]),
                    (-2, 0, [cirq.X(b)], [b]),
                    (-5, 0, [cirq.CZ(a, b)], [a])]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, InsertStrategy.INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]


def test_insert_op_tree_earliest():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = Circuit([Moment([cirq.H(a)])])

    op_tree_list = [(5, [1, 0], [cirq.X(a), cirq.X(b)], [a, b]),
                    (1, [1], [cirq.H(b)], [b]),
                    (-4, [0], [cirq.X(b)], [b])]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, InsertStrategy.EARLIEST)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index[i]) == op_list[i]


def test_insert_moment():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = Circuit()

    moment_list = [(-10, 0, [cirq.CZ(a, b)], a, InsertStrategy.NEW_THEN_INLINE),
                   (-20, 0, [cirq.X(a)], a, InsertStrategy.NEW),
                   (20, 2, [cirq.X(b)], b, InsertStrategy.INLINE),
                   (2, 2, [cirq.H(b)], b, InsertStrategy.EARLIEST),
                   (-3, 1, [cirq.H(a)], a, InsertStrategy.EARLIEST)]

    for given_index, actual_index, operation, qubit, strat in moment_list:
        c.insert(given_index, Moment(operation), strat)
        assert c.operation_at(qubit, actual_index) == operation[0]


def test_insert_inline_near_start():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = Circuit([
        Moment(),
        Moment(),
    ])

    c.insert(1, cirq.X(a), strategy=cirq.InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.X(a)]),
        Moment(),
    ])

    c.insert(1, cirq.Y(a), strategy=cirq.InsertStrategy.INLINE)
    assert c ==Circuit([
        Moment([cirq.X(a)]),
        Moment([cirq.Y(a)]),
        Moment(),
    ])

    c.insert(0, cirq.Z(b), strategy=cirq.InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([cirq.Z(b)]),
        Moment([cirq.X(a)]),
        Moment([cirq.Y(a)]),
        Moment(),
    ])

def test_insert_at_frontier_init():
    x = cirq.NamedQubit('x')
    op = cirq.X(x)
    circuit = Circuit.from_ops(op)
    actual_frontier = circuit.insert_at_frontier(op, 3)
    expected_circuit = Circuit([Moment([op]), Moment(), Moment(), Moment([op])])
    assert circuit == expected_circuit
    expected_frontier = defaultdict(lambda: 0)
    expected_frontier[x] = 4
    assert actual_frontier == expected_frontier

    with pytest.raises(ValueError):
        circuit = Circuit([Moment(), Moment([op])])
        frontier = {x: 2}
        circuit.insert_at_frontier(op, 0, frontier)


def test_insert_at_frontier():
    class Replacer(PointOptimizer):
        def __init__(self, replacer=(lambda x: x)):
            super().__init__()
            self.replacer = replacer

        def optimization_at(self, circuit, index, op):
            new_ops = self.replacer(op)
            return PointOptimizationSummary(clear_span=1,
                                            clear_qubits=op.qubits,
                                            new_operations=new_ops)

    replacer = lambda op: ((cirq.Z(op.qubits[0]),) * 2 +
                           (op, cirq.Y(op.qubits[0])))
    prepend_two_Xs_append_one_Y = Replacer(replacer)
    qubits = [cirq.NamedQubit(s) for s in 'abcdef']
    a, b, c = qubits[:3]

    circuit = Circuit([
        Moment([cirq.CZ(a, b)]),
        Moment([cirq.CZ(b, c)]),
        Moment([cirq.CZ(a, b)])
    ])

    prepend_two_Xs_append_one_Y.optimize_circuit(circuit)

    cirq.testing.assert_has_diagram(circuit, """
a: ───Z───Z───@───Y───────────────Z───Z───@───Y───
              │                           │
b: ───────────@───Z───Z───@───Y───────────@───────
                          │
c: ───────────────────────@───────────────────────
""")

    prepender = lambda op: (cirq.X(op.qubits[0]),) * 3 + (op,)
    prepend_3_Xs = Replacer(prepender)
    circuit = Circuit([
        Moment([cirq.CNOT(a, b)]),
        Moment([cirq.CNOT(b, c)]),
        Moment([cirq.CNOT(c, b)])
    ])
    prepend_3_Xs.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, """
a: ───X───X───X───@───────────────────────────────────
                  │
b: ───────────────X───X───X───X───@───────────────X───
                                  │               │
c: ───────────────────────────────X───X───X───X───@───
""")

    duplicate = Replacer(lambda op: (op,) * 2)
    circuit = Circuit([
        Moment([cirq.CZ(qubits[j], qubits[j + 1])
                for j in range(i % 2, 5, 2)])
        for i in range(4)])

    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, """
a: ───@───@───────────@───@───────────
      │   │           │   │
b: ───@───@───@───@───@───@───@───@───
              │   │           │   │
c: ───@───@───@───@───@───@───@───@───
      │   │           │   │
d: ───@───@───@───@───@───@───@───@───
              │   │           │   │
e: ───@───@───@───@───@───@───@───@───
      │   │           │   │
f: ───@───@───────────@───@───────────
""")

    circuit = Circuit([
        Moment([cirq.CZ(*qubits[2:4]), cirq.CNOT(*qubits[:2])]),
        Moment([cirq.CNOT(*qubits[1::-1])])
    ])

    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, """
a: ───@───@───X───X───
      │   │   │   │
b: ───X───X───@───@───

c: ───@───────@───────
      │       │
d: ───@───────@───────
""")


def test_insert_into_range():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    c = Circuit([Moment([cirq.X(x)])] * 4)
    c.insert_into_range([cirq.Z(x), cirq.CZ(x, y)], 2, 2)
    cirq.testing.assert_has_diagram(c, """
x: ───X───X───Z───@───X───X───
                  │
y: ───────────────@───────────
""")


def test_next_moment_operating_on():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    za = cirq.Z.on(a)
    zb = cirq.Z.on(b)

    def is_x(op: cirq.Operation) -> bool:
        return (isinstance(op, cirq.GateOperation) and
                isinstance(op.gate, cirq.XPowGate))

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
    assert list(c.findall_operations_with_gate_type(cirq.XPowGate)) == [
        (0, cirq.X(a), cirq.X),
        (2, cirq.X(a), cirq.X),
        (2, cirq.X(b), cirq.X),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.CZPowGate)) == [
        (3, cirq.CZ(a, b), cirq.CZ),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.MeasurementGate)) == [
        (4, cirq.MeasurementGate(key='a').on(a), cirq.MeasurementGate(key='a')),
        (4, cirq.MeasurementGate(key='b').on(b), cirq.MeasurementGate(key='b')),
    ]


def test_are_all_measurements_terminal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

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

    cirq.testing.assert_has_diagram(c, """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────@───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───@───
""")

    cirq.testing.assert_has_diagram(c, """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------@---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---@---
""", use_unicode_characters=False)

    cirq.testing.assert_has_diagram(c, """
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
""", use_unicode_characters=False, transpose=True)


def test_diagram_with_unknown_exponent():
    class WeirdGate(cirq.Gate):
        def _circuit_diagram_info_(self,
                                   args: cirq.CircuitDiagramInfoArgs
                                   ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('B',),
                                           exponent='fancy')

    class WeirderGate(cirq.Gate):
        def _circuit_diagram_info_(self,
                                   args: cirq.CircuitDiagramInfoArgs
                                   ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('W',),
                                           exponent='fancy-that')

    c = cirq.Circuit.from_ops(
        WeirdGate().on(cirq.NamedQubit('q')),
        WeirderGate().on(cirq.NamedQubit('q')),
    )

    # The hyphen in the exponent should cause parens to appear.
    cirq.testing.assert_has_diagram(c, 'q: ───B^fancy───W^(fancy-that)───')


def test_circuit_diagram_on_gate_without_info():
    q = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')

    class FGate(cirq.Gate):
        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'

    # Fallback to repr.
    f = FGate()
    cirq.testing.assert_has_diagram(Circuit([
        Moment([f.on(q)]),
    ]), """
(0, 0): ---python-object-FGate:arbitrary-digits---
""", use_unicode_characters=False)

    # When used on multiple qubits, show the qubit order as a digit suffix.
    cirq.testing.assert_has_diagram(Circuit([
        Moment([f.on(q, q3, q2)]),
    ]), """
(0, 0): ---python-object-FGate:arbitrary-digits---
           |
(0, 1): ---#3-------------------------------------
           |
(0, 2): ---#2-------------------------------------
""", use_unicode_characters=False)


def test_to_text_diagram_multi_qubit_gate():
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = Circuit.from_ops(cirq.measure(q1, q2, q3, key='msg'))
    cirq.testing.assert_has_diagram(c, """
(0, 0): ───M('msg')───
           │
(0, 1): ───M──────────
           │
(0, 2): ───M──────────
""")
    cirq.testing.assert_has_diagram(c, """
(0, 0): ---M('msg')---
           |
(0, 1): ---M----------
           |
(0, 2): ---M----------
""", use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c, """
(0, 0)   (0, 1) (0, 2)
│        │      │
M('msg')─M──────M
│        │      │
""", transpose=True)


def test_to_text_diagram_many_qubits_gate_but_multiple_wire_symbols():
    class BadGate(cirq.Gate):
        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                   ) -> Tuple[str, str]:
            return 'a', 'a'
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = Circuit([Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()


def test_to_text_diagram_parameterized_value():
    q = cirq.NamedQubit('cube')

    class PGate(cirq.Gate):
        def __init__(self, val):
            self.val = val

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                   ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(('P',), self.val)

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
    cirq.testing.assert_has_diagram(c, """
3: ---X---

4: ---X---

2: ---X---
""", qubit_order=cirq.QubitOrder.sorted_by(lambda e: int(str(e)) % 3),
                                    use_unicode_characters=False)


def test_overly_precise_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.12345678])])
    cirq.testing.assert_has_diagram(c, """
a: ---X^0.123---
""", use_unicode_characters=False)


def test_none_precision_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.4921875])])
    cirq.testing.assert_has_diagram(c, """
a: ---X^0.4921875---
""", use_unicode_characters=False, precision=None)


def test_diagram_custom_precision():
    qa = cirq.NamedQubit('a')
    c = Circuit([Moment([cirq.X(qa)**0.12341234])])
    cirq.testing.assert_has_diagram(c, """
a: ---X^0.12341---
""", use_unicode_characters=False, precision=5)


def test_diagram_wgate():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(
        exponent=0.12341234, phase_exponent=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(c, """
a: ---PhasedX(0.43)^(1/8)---
""", use_unicode_characters=False, precision=2)


def test_diagram_wgate_none_precision():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(
        exponent=0.12341234, phase_exponent=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(c, """
a: ---PhasedX(0.43214321)^0.12341234---
""", use_unicode_characters=False, precision=None)


def test_has_unitary():

    class NonUnitary(cirq.Gate):
        pass

    class EventualUnitary(cirq.Gate):
        def _decompose_(self, qubits):
            return cirq.X.on_each(qubits)

    q = cirq.NamedQubit('q')

    # Non-unitary operations cause a non-unitary circuit.
    assert cirq.has_unitary(cirq.Circuit.from_ops(cirq.X(q)))
    assert not cirq.has_unitary(cirq.Circuit.from_ops(NonUnitary().on(q)))

    # Terminal measurements are ignored, though.
    assert cirq.has_unitary(cirq.Circuit.from_ops(cirq.measure(q)))
    assert not cirq.has_unitary(cirq.Circuit.from_ops(cirq.measure(q),
                                                      cirq.measure(q)))

    # Still unitary if operations decompose into unitary operations.
    assert cirq.has_unitary(cirq.Circuit.from_ops(EventualUnitary().on(q)))


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


def test_circuit_to_unitary_matrix():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Single qubit gates.
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit.from_ops(cirq.X(a)**0.5).to_unitary_matrix(),
        np.array([
            [1j, 1],
            [1, 1j],
        ]) * np.sqrt(0.5),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit.from_ops(cirq.Y(a)**0.25).to_unitary_matrix(),
        cirq.unitary(cirq.Y(a)**0.25),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit.from_ops(cirq.Z(a), cirq.X(b)).to_unitary_matrix(),
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, -1, 0],
        ]),
        atol=1e-8)

    # Single qubit gates and two qubit gate.
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit.from_ops(cirq.Z(a),
                         cirq.X(b),
                         cirq.CNOT(a, b)).to_unitary_matrix(),
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        Circuit.from_ops(cirq.H(b),
                         cirq.CNOT(b, a)**0.5,
                         cirq.Y(a)**0.5).to_unitary_matrix(),
        np.array([
            [1, 1, -1, -1],
            [1j, -1j, -1j, 1j],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
        ]) * np.sqrt(0.25),
        atol=1e-8)

    # Measurement gate has no corresponding matrix.
    c = Circuit.from_ops(cirq.measure(a))
    with pytest.raises(ValueError):
        _ = c.to_unitary_matrix(ignore_terminal_measurements=False)

    # Ignoring terminal measurements.
    c = Circuit.from_ops(cirq.measure(a))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        np.eye(2),
        atol=1e-8)

    # Ignoring terminal measurements with further cirq.
    c = Circuit.from_ops(cirq.Z(a), cirq.measure(a), cirq.Z(b))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]),
        atol=1e-8)

    # Optionally don't ignoring terminal measurements.
    c = Circuit.from_ops(cirq.measure(a))
    with pytest.raises(ValueError, match="measurement"):
        _ = c.to_unitary_matrix(ignore_terminal_measurements=False),

    # Non-terminal measurements are not ignored.
    c = Circuit.from_ops(cirq.measure(a), cirq.X(a))
    with pytest.raises(ValueError):
        _ = c.to_unitary_matrix()

    # Non-terminal measurements are not ignored (multiple qubits).
    c = Circuit.from_ops(
        cirq.measure(a),
        cirq.measure(b),
        cirq.CNOT(a, b))
    with pytest.raises(ValueError):
        _ = c.to_unitary_matrix()

    # Gates without matrix or decomposition raise exception
    class MysteryGate(cirq.Gate):
        pass
    c = Circuit.from_ops(MysteryGate()(a, b))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix()

    # Accounts for measurement bit flipping.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cirq.measure(a, invert_mask=(True,))
        ).to_unitary_matrix(),
        cirq.unitary(cirq.X),
        atol=1e-8)


def test_circuit_unitary():
    q = cirq.NamedQubit('q')

    with_inner_measure = cirq.Circuit.from_ops(
        cirq.H(q), cirq.measure(q), cirq.H(q))
    assert not cirq.has_unitary(with_inner_measure)
    assert cirq.unitary(with_inner_measure, None) is None

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(Circuit.from_ops(
            cirq.X(q)**0.5),
            cirq.measure(q),
        ),
        np.array([
            [1j, 1],
            [1, 1j],
        ]) * np.sqrt(0.5),
        atol=1e-8)


def test_simple_circuits_to_unitary_matrix():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Phase parity.
    c = Circuit.from_ops(cirq.CNOT(a, b), cirq.Z(b), cirq.CNOT(a, b))
    assert cirq.has_unitary(c)
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
    for expected in [np.diag([1, 1j, -1, -1j]),
                     cirq.unitary(cirq.CNOT)]:

        class Passthrough(cirq.Gate):
            def _unitary_(self) -> np.ndarray:
                return expected

        c = Circuit.from_ops(Passthrough()(a, b))
        m = c.to_unitary_matrix()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected, atol=1e-8)


def test_composite_gate_to_unitary_matrix():
    class CnotComposite(cirq.Gate):
        def _decompose_(self, qubits):
            q0, q1 = qubits
            return cirq.Y(q1)**-0.5, cirq.CZ(q0, q1), cirq.Y(q1)**0.5

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = Circuit.from_ops(
        cirq.X(a),
        CnotComposite()(a, b),
        cirq.X(a),
        cirq.measure(a),
        cirq.X(b),
        cirq.measure(b))
    assert cirq.has_unitary(c)

    mat = c.to_unitary_matrix()
    mat_expected = cirq.unitary(cirq.CNOT)

    cirq.testing.assert_allclose_up_to_global_phase(mat, mat_expected,
                                                    atol=1e-8)


def test_expanding_gate_symbols():
    class MultiTargetCZ(cirq.Gate):
        def _circuit_diagram_info_(self,
                                   args: cirq.CircuitDiagramInfoArgs
                                   ) -> Tuple[str, ...]:
            assert args.known_qubit_count is not None
            return ('@',) + ('Z',) * (args.known_qubit_count - 1)

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    t0 = cirq.Circuit.from_ops(MultiTargetCZ().on(c))
    t1 = cirq.Circuit.from_ops(MultiTargetCZ().on(c, a))
    t2 = cirq.Circuit.from_ops(MultiTargetCZ().on(c, a, b))

    cirq.testing.assert_has_diagram(t0, """
c: ───@───
""")

    cirq.testing.assert_has_diagram(t1, """
a: ───Z───
      │
c: ───@───
""")

    cirq.testing.assert_has_diagram(t2, """
a: ───Z───
      │
b: ───Z───
      │
c: ───@───
""")


def test_transposed_diagram_exponent_order():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(
        cirq.CZ(a, b)**-0.5,
        cirq.CZ(a, c)**0.5,
        cirq.CZ(b, c)**0.125,
    )
    cirq.testing.assert_has_diagram(circuit, """
0 1      2
│ │      │
@─@^-0.5 │
│ │      │
@─┼──────@^0.5
│ │      │
│ @──────@^(1/8)
│ │      │
""", transpose=True)


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

    assert c._moments == [m2, m0, m1]
    assert c._moments[0] is m2

def test_apply_unitary_effect_to_state():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # State ordering.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(a)**0.5).apply_unitary_effect_to_state(),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(a)**0.5).apply_unitary_effect_to_state(
            initial_state=0),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(a)**0.5).apply_unitary_effect_to_state(
            initial_state=1),
        np.array([1, 1j]) * np.sqrt(0.5),
        atol=1e-8)

    # Vector state.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(a)**0.5).apply_unitary_effect_to_state(
            initial_state=np.array([1j, 1]) * np.sqrt(0.5)),
        np.array([0, 1]),
        atol=1e-8)

    # Qubit ordering.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.CNOT(a, b)).apply_unitary_effect_to_state(
            initial_state=0),
        np.array([1, 0, 0, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.CNOT(a, b)).apply_unitary_effect_to_state(
            initial_state=1),
        np.array([0, 1, 0, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.CNOT(a, b)).apply_unitary_effect_to_state(
            initial_state=2),
        np.array([0, 0, 0, 1]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.CNOT(a, b)).apply_unitary_effect_to_state(
            initial_state=3),
        np.array([0, 0, 1, 0]),
        atol=1e-8)

    # Measurements.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.measure(a)).apply_unitary_effect_to_state(),
        np.array([1, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(a), cirq.measure(a)
                              ).apply_unitary_effect_to_state(),
        np.array([0, 1]),
        atol=1e-8)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.Circuit.from_ops(cirq.measure(a), cirq.X(a)
                                  ).apply_unitary_effect_to_state(),
            np.array([1, 0]),
            atol=1e-8)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.Circuit.from_ops(
                cirq.measure(a)).apply_unitary_effect_to_state(
                    ignore_terminal_measurements=False),
            np.array([1, 0]),
            atol=1e-8)

    # Extra qubits.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops().apply_unitary_effect_to_state(),
        np.array([1]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops().apply_unitary_effect_to_state(
            qubits_that_should_be_present=[a]),
        np.array([1, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(cirq.X(b)).apply_unitary_effect_to_state(
            qubits_that_should_be_present=[a]),
        np.array([0, 1, 0, 0]),
        atol=1e-8)

    # Qubit order.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cirq.Z(a), cirq.X(b)).apply_unitary_effect_to_state(
                qubit_order=[a, b]),
        np.array([0, 1, 0, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cirq.Z(a), cirq.X(b)).apply_unitary_effect_to_state(
                qubit_order=[b, a]),
        np.array([0, 0, 1, 0]),
        atol=1e-8)

    # Dtypes.
    for dt in (np.complex64, np.complex128, np.complex256):
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.Circuit.from_ops(cirq.X(a)**0.5).apply_unitary_effect_to_state(
                initial_state=np.array([1j, 1]) * np.sqrt(0.5), dtype=dt),
            np.array([0, 1]),
            atol=1e-8)


def test_is_parameterized():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        cirq.CZ(a, b)**cirq.Symbol('u'),
        cirq.X(a)**cirq.Symbol('v'),
        cirq.Y(b)**cirq.Symbol('w'),
    )
    assert cirq.is_parameterized(circuit)

    circuit = cirq.resolve_parameters(circuit,
                                      cirq.ParamResolver({'u': 0.1, 'v': 0.3}))
    assert cirq.is_parameterized(circuit)

    circuit = cirq.resolve_parameters(circuit,
                                      cirq.ParamResolver({'w': 0.2}))
    assert not cirq.is_parameterized(circuit)


def test_resolve_parameters():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        cirq.CZ(a, b)**cirq.Symbol('u'),
        cirq.X(a)**cirq.Symbol('v'),
        cirq.Y(b)**cirq.Symbol('w'),
    )
    resolved_circuit = cirq.resolve_parameters(
        circuit,
        cirq.ParamResolver({'u': 0.1, 'v': 0.3, 'w': 0.2}))
    cirq.testing.assert_has_diagram(resolved_circuit, """
0: ───@───────X^0.3───
      │
1: ───@^0.1───Y^0.2───
""")
    q = cirq.NamedQubit('q')
    # no-op parameter resolution
    circuit = cirq.Circuit([
        cirq.Moment(), cirq.Moment([cirq.X(q)])])
    resolved_circuit = cirq.resolve_parameters(
        circuit,
        cirq.ParamResolver({}))
    cirq.testing.assert_same_circuits(circuit, resolved_circuit)
    # actually resolve something
    circuit = cirq.Circuit([
        cirq.Moment(), cirq.Moment([cirq.X(q)**cirq.Symbol('x')])])
    resolved_circuit = cirq.resolve_parameters(
        circuit,
        cirq.ParamResolver({'x': 0.2}))
    expected_circuit = cirq.Circuit([
        cirq.Moment(), cirq.Moment([cirq.X(q)**0.2])])
    cirq.testing.assert_same_circuits(expected_circuit, resolved_circuit)


def test_items():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit()
    m1 = cirq.Moment([cirq.X(a), cirq.X(b)])
    m2 = cirq.Moment([cirq.X(a)])
    m3 = cirq.Moment([])
    m4 = cirq.Moment([cirq.CZ(a, b)])

    c[:] = [m1, m2]
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([m1, m2]))

    assert c[0] == m1
    del c[0]
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([m2]))

    c.append(m1)
    c.append(m3)
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([m2, m1, m3]))

    assert c[0:2] == Circuit([m2, m1])
    c[0:2] = [m4]
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([m4, m3]))

    c[:] = [m1]
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([m1]))

    with pytest.raises(TypeError):
        c[:] = [m1, 1]
    with pytest.raises(TypeError):
        c[0] = 1


def test_copy():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit.from_ops(cirq.X(a), cirq.CZ(a, b), cirq.Z(a), cirq.Z(b))
    assert c == c.copy() == c.__copy__()
    c2 = c.copy()
    assert c2 == c
    c2[:] = []
    assert c2 != c


def test_batch_remove():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Empty case.
    after = original.copy()
    after.batch_remove([])
    assert after == original

    # Delete one.
    after = original.copy()
    after.batch_remove([(0, cirq.X(a))])
    assert after == cirq.Circuit([
        cirq.Moment(),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Out of range.
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_remove([(500, cirq.X(a))])
    assert after == original

    # Delete several.
    after = original.copy()
    after.batch_remove([(0, cirq.X(a)), (2, cirq.CZ(a, b))])
    assert after == cirq.Circuit([
        cirq.Moment(),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment(),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Delete all.
    after = original.copy()
    after.batch_remove([(0, cirq.X(a)),
                        (1, cirq.Z(b)),
                        (2, cirq.CZ(a, b)),
                        (3, cirq.X(a)),
                        (3, cirq.X(b))])
    assert after == cirq.Circuit([
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
    ])

    # Delete moment partially.
    after = original.copy()
    after.batch_remove([(3, cirq.X(a))])
    assert after == cirq.Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(b)]),
    ])

    # Deleting something that's not there.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(b))])
    assert after == original

    # Duplicate delete.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(a)), (0, cirq.X(a))])
    assert after == original


def test_batch_insert_into():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Empty case.
    after = original.copy()
    after.batch_insert_into([])
    assert after == original

    # Add into non-empty moment.
    after = original.copy()
    after.batch_insert_into([(0, cirq.X(b))])
    assert after == cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.X(b)]),
        cirq.Moment(),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Add into empty moment.
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b))])
    assert after == cirq.Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Add into two moments.
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b)), (0, cirq.X(b))])
    assert after == cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.X(b)]),
        cirq.Moment([cirq.Z(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Out of range.
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_insert_into([(500, cirq.X(a))])
    assert after == original

    # Collision.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(0, cirq.X(a))])
    assert after == original

    # Duplicate insertion collision.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(1, cirq.X(a)), (1, cirq.CZ(a, b))])
    assert after == original


def test_batch_insert():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])

    # Empty case.
    after = original.copy()
    after.batch_insert([])
    assert after == original

    # Pushing.
    after = original.copy()
    after.batch_insert([(0, cirq.CZ(a, b)),
                        (0, cirq.CNOT(a, b)),
                        (1, cirq.Z(b))])
    assert after == cirq.Circuit([
        cirq.Moment([cirq.CNOT(a, b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.Z(b)]),
        cirq.Moment(),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])


def test_batch_insert_multiple_same_index():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.batch_insert([(0, cirq.Z(a)),
                    (0, cirq.Z(b)),
                    (0, cirq.Z(a))])
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([
            cirq.Moment([cirq.Z(a), cirq.Z(b)]),
            cirq.Moment([cirq.Z(a)]),
        ]))


def test_batch_insert_reverses_order_for_same_index_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.batch_insert([(0, cirq.Z(a)),
                    (0, cirq.CZ(a, b)),
                    (0, cirq.Z(b))])
    assert c == cirq.Circuit.from_ops(cirq.Z(b),
                                      cirq.CZ(a, b),
                                      cirq.Z(a))


def test_batch_insert_maintains_order_despite_multiple_previous_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit.from_ops(cirq.H(a))
    c.batch_insert([(0, cirq.Z(a)),
                    (0, cirq.Z(a)),
                    (0, cirq.Z(a)),
                    (1, cirq.CZ(a, b))])
    assert c == cirq.Circuit.from_ops([cirq.Z(a)]*3, cirq.H(a), cirq.CZ(a, b))


def test_batch_insert_doesnt_overshift_due_to_previous_shifts():
    a = cirq.NamedQubit('a')
    c = cirq.Circuit().from_ops([cirq.H(a)] * 3)
    c.batch_insert([(0, cirq.Z(a)),
                    (0, cirq.Z(a)),
                    (1, cirq.X(a)),
                    (2, cirq.Y(a))])
    assert c == cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.Z(a),
        cirq.H(a),
        cirq.X(a),
        cirq.H(a),
        cirq.Y(a),
        cirq.H(a),
    )


def test_batch_insert_doesnt_overshift_due_to_inline_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit().from_ops(
        cirq.SWAP(a, b),
        cirq.SWAP(a, b),
        cirq.H(a),
        cirq.SWAP(a, b),
        cirq.SWAP(a, b))
    c.batch_insert([(0, cirq.X(a)),
                    (3, cirq.X(b)),
                    (4, cirq.Y(a))])
    assert c == cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.SWAP(a, b),
        cirq.SWAP(a, b),
        cirq.H(a), cirq.X(b),
        cirq.SWAP(a, b),
        cirq.Y(a),
        cirq.SWAP(a, b)
    )


def test_next_moments_operating_on():
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = random_circuit(randint(1, 20), n_moments, random())
        circuit_qubits = circuit.all_qubits()
        n_key_qubits = randint(int(bool(circuit_qubits)), len(circuit_qubits))
        key_qubits = sample(circuit_qubits, n_key_qubits)
        start = randrange(len(circuit))
        next_moments = circuit.next_moments_operating_on(key_qubits, start)
        for q, m in next_moments.items():
            if m == len(circuit):
                p = circuit.prev_moment_operating_on([q])
            else:
                p = circuit.prev_moment_operating_on([q], m - 1)
            assert (not p) or (p < start)


def test_pick_inserted_ops_moment_indices():
    for _ in range(20):
        n_moments = randint(1, 10)
        n_qubits = randint(1, 20)
        op_density = random()
        circuit = random_circuit(n_qubits, n_moments, op_density)
        start = randrange(n_moments)
        first_half = Circuit(circuit[:start])
        second_half = Circuit(circuit[start:])
        operations = tuple(op for moment in second_half
                for op in moment.operations)
        squeezed_second_half = Circuit.from_ops(operations,
                strategy=InsertStrategy.EARLIEST)
        expected_circuit = Circuit(first_half._moments +
                                   squeezed_second_half._moments)
        expected_circuit._moments += [Moment() for _ in
                range(len(circuit) - len(expected_circuit))]
        insert_indices, _ = circuit._pick_inserted_ops_moment_indices(
            operations, start)
        actual_circuit = Circuit(first_half._moments +
            [Moment() for _ in range(n_moments - start)])
        for op, insert_index in zip(operations, insert_indices):
            actual_circuit._moments[insert_index] = (
                actual_circuit._moments[insert_index].with_operation(op))
        assert actual_circuit == expected_circuit


def test_push_frontier_new_moments():
    operation = cirq.X(cirq.NamedQubit('q'))
    insertion_index = 3
    circuit = Circuit()
    circuit._insert_operations([operation], [insertion_index])
    assert circuit == Circuit([Moment() for _ in range(insertion_index)] +
                              [Moment([operation])])


def test_push_frontier_random_circuit():
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = random_circuit(randint(1, 20), n_moments, random())
        qubits = circuit.all_qubits()
        early_frontier = {q: randint(0, n_moments) for q in
                          sample(qubits, randint(0, len(qubits)))}
        late_frontier = {q: randint(0, n_moments) for q in
                         sample(qubits, randint(0, len(qubits)))}
        update_qubits = sample(qubits, randint(0, len(qubits)))

        orig_early_frontier = {q: f for q, f in early_frontier.items()}
        orig_moments = [m for m in circuit._moments]
        insert_index, n_new_moments = circuit._push_frontier(
            early_frontier, late_frontier, update_qubits)

        assert set(early_frontier.keys()) == set(orig_early_frontier.keys())
        for q in set(early_frontier).difference(update_qubits):
            assert early_frontier[q] == orig_early_frontier[q]
        for q, f in late_frontier.items():
            assert (orig_early_frontier.get(q, 0) <=
                    late_frontier[q] + n_new_moments)
            if f != len(orig_moments):
                assert orig_moments[f] == circuit[f + n_new_moments]
        for q in set(update_qubits).intersection(early_frontier):
            if orig_early_frontier[q] == insert_index:
                assert orig_early_frontier[q] == early_frontier[q]
                assert (not n_new_moments) or (
                        circuit._moments[early_frontier[q]] == Moment())
            elif orig_early_frontier[q] == len(orig_moments):
                assert early_frontier[q] == len(circuit)
            else:
                assert (orig_moments[orig_early_frontier[q]] ==
                        circuit._moments[early_frontier[q]])


@pytest.mark.parametrize('circuit',
    [random_circuit(cirq.LineQubit.range(10), 10, 0.5)
     for _ in range(20)])
def test_insert_operations_random_circuits(circuit):
    n_moments = len(circuit)
    operations, insert_indices = [], []
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            operations.append(op)
            insert_indices.append(moment_index)
    other_circuit = Circuit([Moment() for _ in range(n_moments)])
    other_circuit._insert_operations(operations, insert_indices)
    assert circuit == other_circuit


def test_insert_operations_errors():
    a, b, c = (cirq.NamedQubit(s) for s in 'abc')
    with pytest.raises(ValueError):
        circuit = cirq.Circuit([Moment([cirq.Z(c)])])
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = [0, 0]
        circuit._insert_operations(operations, insertion_indices)

    with pytest.raises(ValueError):
        circuit = cirq.Circuit.from_ops(cirq.X(a))
        operations = [cirq.CZ(a, b)]
        insertion_indices = [0]
        circuit._insert_operations(operations, insertion_indices)

    with pytest.raises(ValueError):
        circuit = cirq.Circuit()
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = []
        circuit._insert_operations(operations, insertion_indices)


def test_validates_while_editing():
    c = cirq.Circuit(device=cg.Foxtail)

    with pytest.raises(ValueError, match='Unsupported qubit type'):
        # Wrong type of qubit.
        c.append(cirq.Z(cirq.NamedQubit('q')))
    with pytest.raises(ValueError, match='Qubit not on device'):
        # A qubit that's not on the device.
        c[:] = [cirq.Moment([
            cirq.Z(cirq.GridQubit(-5, 100))])]
    c.append(cirq.Z(cirq.GridQubit(0, 0)))

    with pytest.raises(ValueError, match='Non-local interaction'):
        # Non-adjacent CZ.
        c[0] = cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0),
                                    cirq.GridQubit(1, 2))])

    c.insert(0, cirq.CZ(cirq.GridQubit(0, 0),
                        cirq.GridQubit(1, 0)))


def test_respects_additional_adjacency_constraints():
    c = cirq.Circuit(device=cg.Foxtail)
    c.append(cirq.CZ(cirq.GridQubit(0, 0),
                     cirq.GridQubit(0, 1)))
    c.append(cirq.CZ(cirq.GridQubit(1, 0),
                     cirq.GridQubit(1, 1)),
             strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([
            cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0),
                                 cirq.GridQubit(0, 1))]),
            cirq.Moment([cirq.CZ(cirq.GridQubit(1, 0),
                                 cirq.GridQubit(1, 1))]),
        ], device=cg.Foxtail))


def test_commutes_past_adjacency_constraints():
    c = cirq.Circuit([
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment([
            cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))])
    ],
        device=cg.Foxtail)
    c.append(cirq.CZ(cirq.GridQubit(1, 0),
                     cirq.GridQubit(1, 1)),
             strategy=cirq.InsertStrategy.EARLIEST)
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([
            cirq.Moment([cirq.CZ(cirq.GridQubit(1, 0),
                                 cirq.GridQubit(1, 1))]),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0),
                                 cirq.GridQubit(0, 1))]),
        ], device=cg.Foxtail))


def test_decomposes_while_appending():
    c = cirq.Circuit(device=cg.Foxtail)
    c.append(cirq.TOFFOLI(cirq.GridQubit(0, 0),
                          cirq.GridQubit(0, 1),
                          cirq.GridQubit(0, 2)))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        cirq.unitary(cirq.TOFFOLI),
        atol=1e-8)

    # But you still have to respect adjacency constraints!
    with pytest.raises(ValueError):
        c.append(cirq.TOFFOLI(cirq.GridQubit(0, 0),
                              cirq.GridQubit(0, 2),
                              cirq.GridQubit(0, 4)))


def test_to_qasm():
    q0 = cirq.NamedQubit('q0')
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
    )
    assert circuit.to_qasm() == cirq.qasm(circuit)
    assert (circuit.to_qasm() ==
"""// Generated from Cirq v{}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
""".format(cirq.__version__))


def test_save_qasm():
    q0 = cirq.NamedQubit('q0')
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
    )
    with cirq.testing.TempFilePath() as file_path:
        circuit.save_qasm(file_path)
        with open(file_path, 'r') as f:
            file_content = f.read()
    assert (file_content ==
"""// Generated from Cirq v{}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
""".format(cirq.__version__))


def test_findall_operations_between():
    a, b, c, d = cirq.LineQubit.range(4)

    #    0: ───H───@───────────────────────────────────────@───H───
    #              │                                       │
    #    1: ───────@───H───@───────────────────────@───H───@───────
    #                      │                       │
    #    2: ───────────────@───H───@───────@───H───@───────────────
    #                              │       │
    #    3: ───────────────────────@───H───@───────────────────────
    #
    # moments: 0   1   2   3   4   5   6   7   8   9   10  11  12
    circuit = cirq.Circuit.from_ops(
        cirq.H(a),
        cirq.CZ(a, b),
        cirq.H(b),
        cirq.CZ(b, c),
        cirq.H(c),
        cirq.CZ(c, d),
        cirq.H(d),
        cirq.CZ(c, d),
        cirq.H(c),
        cirq.CZ(b, c),
        cirq.H(b),
        cirq.CZ(a, b),
        cirq.H(a))

    # Empty frontiers means no results.
    actual = circuit.findall_operations_between(
        start_frontier={},
        end_frontier={})
    assert actual == []

    # Empty range is empty.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5},
        end_frontier={a: 5})
    assert actual == []

    # Default end_frontier value is len(circuit.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5},
        end_frontier={})
    assert actual == [
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Default start_frontier value is 0.
    actual = circuit.findall_operations_between(
        start_frontier={},
        end_frontier={a: 5})
    assert actual == [
        (0, cirq.H(a)),
        (1, cirq.CZ(a, b))
    ]

    # omit_crossing_operations omits crossing operations.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5},
        end_frontier={},
        omit_crossing_operations=True)
    assert actual == [
        (12, cirq.H(a)),
    ]

    # omit_crossing_operations keeps operations across included regions.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5, b: 5},
        end_frontier={},
        omit_crossing_operations=True)
    assert actual == [
        (10, cirq.H(b)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5},
        end_frontier={b: 5})
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together (2).
    actual = circuit.findall_operations_between(
        start_frontier={a: 5},
        end_frontier={a: 5, b: 5})
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
    ]

    # Inclusive start, exclusive end.
    actual = circuit.findall_operations_between(
        start_frontier={c: 4},
        end_frontier={c: 8})
    assert actual == [
        (4, cirq.H(c)),
        (5, cirq.CZ(c, d)),
        (7, cirq.CZ(c, d)),
    ]

    # Out of range is clamped.
    actual = circuit.findall_operations_between(
        start_frontier={a: -100},
        end_frontier={a: +100})
    assert actual == [
        (0, cirq.H(a)),
        (1, cirq.CZ(a, b)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]


def test_reachable_frontier_from():
    a, b, c, d = cirq.LineQubit.range(4)

    #    0: ───H───@───────────────────────────────────────@───H───
    #              │                                       │
    #    1: ───────@───H───@───────────────────────@───H───@───────
    #                      │                       │
    #    2: ───────────────@───H───@───────@───H───@───────────────
    #                              │       │
    #    3: ───────────────────────@───H───@───────────────────────
    #
    # moments: 0   1   2   3   4   5   6   7   8   9   10  11  12
    circuit = cirq.Circuit.from_ops(
        cirq.H(a),
        cirq.CZ(a, b),
        cirq.H(b),
        cirq.CZ(b, c),
        cirq.H(c),
        cirq.CZ(c, d),
        cirq.H(d),
        cirq.CZ(c, d),
        cirq.H(c),
        cirq.CZ(b, c),
        cirq.H(b),
        cirq.CZ(a, b),
        cirq.H(a))

    # Empty cases.
    assert cirq.Circuit().reachable_frontier_from(start_frontier={}) == {}
    assert circuit.reachable_frontier_from(start_frontier={}) == {}

    # Clamped input cases.
    assert cirq.Circuit().reachable_frontier_from(
        start_frontier={a: 5}) == {a: 5}
    assert cirq.Circuit().reachable_frontier_from(
        start_frontier={a: -100}) == {a: 0}
    assert circuit.reachable_frontier_from(
        start_frontier={a: 100}) == {a: 100}

    # Stopped by crossing outside case.
    assert circuit.reachable_frontier_from({a: -1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 0}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 2}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 5}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 10}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 11}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 12}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 13}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 14}) == {a: 14}

    # Inside crossing works only before blocked case.
    assert circuit.reachable_frontier_from({a: 0, b: 0}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 2, b: 2}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 0, b: 4}) == {a: 1, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 4}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 9}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 10}) == {a: 13, b: 13}

    # Travelling shadow.
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0}) == {a: 11,
                                                                   b: 9,
                                                                   c: 5}

    # Full circuit
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0, d: 0}) == {
        a: 13,
        b: 13,
        c: 13,
        d: 13
    }

    # Blocker.
    assert circuit.reachable_frontier_from(
        {a: 0, b: 0, c: 0, d: 0},
        is_blocker=lambda op: op == cirq.CZ(b, c)) == {
            a: 11,
            b: 3,
            c: 3,
            d: 5
        }


def test_submoments():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.Circuit.from_ops(
        cirq.H.on(a),
        cirq.H.on(d),
        cirq.CZ.on(a, d),
        cirq.CZ.on(b, c),
        (cirq.CNOT**0.5).on(a, d),
        (cirq.CNOT**0.5).on(b, e),
        (cirq.CNOT**0.5).on(c, f),
        cirq.H.on(c),
        cirq.H.on(e),
    )

    cirq.testing.assert_has_diagram(circuit, """
         ┌──┐  ┌───────────────┐
0: ───H───@─────@─────────────────────
          │     │
1: ───────┼@────┼────@────────────────
          ││    │    │
2: ───────┼@────┼────┼────@───────H───
          │     │    │    │
3: ───H───@─────X^0.5┼────┼───────────
                     │    │
4: ──────────────────X^0.5┼───────H───
                          │
5: ───────────────────────X^0.5───────
         └──┘  └───────────────┘
""")

    cirq.testing.assert_has_diagram(circuit, """
  0 1 2 3     4     5
  │ │ │ │     │     │
  H │ │ H     │     │
┌ │ │ │ │     │     │     ┐
│ @─┼─┼─@     │     │     │
│ │ @─@ │     │     │     │
└ │ │ │ │     │     │     ┘
┌ │ │ │ │     │     │     ┐
│ @─┼─┼─X^0.5 │     │     │
│ │ @─┼─┼─────X^0.5 │     │
│ │ │ @─┼─────┼─────X^0.5 │
└ │ │ │ │     │     │     ┘
  │ │ H │     H     │
  │ │ │ │     │     │
""", transpose=True)

    cirq.testing.assert_has_diagram(circuit, r"""
         /--\  /---------------\
0: ---H---@-----@---------------------
          |     |
1: -------|@----|----@----------------
          ||    |    |
2: -------|@----|----|----@-------H---
          |     |    |    |
3: ---H---@-----X^0.5|----|-----------
                     |    |
4: ------------------X^0.5|-------H---
                          |
5: -----------------------X^0.5-------
         \--/  \---------------/
""", use_unicode_characters=False)

    cirq.testing.assert_has_diagram(circuit, r"""
  0 1 2 3     4     5
  | | | |     |     |
  H | | H     |     |
/ | | | |     |     |     \
| @-----@     |     |     |
| | @-@ |     |     |     |
\ | | | |     |     |     /
/ | | | |     |     |     \
| @-----X^0.5 |     |     |
| | @---------X^0.5 |     |
| | | @-------------X^0.5 |
\ | | | |     |     |     /
  | | H |     H     |
  | | | |     |     |
""", use_unicode_characters=False, transpose=True)


def test_decompose():
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose(
        cirq.Circuit.from_ops(cirq.X(a), cirq.Y(b), cirq.CZ(a, b))
    ) == [cirq.X(a), cirq.Y(b), cirq.CZ(a, b)]
