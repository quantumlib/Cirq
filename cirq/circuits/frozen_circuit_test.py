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

from random import randint, random, sample, randrange
import os
import numpy as np
import pytest
import sympy

import cirq
import cirq.google as cg
import cirq.testing


def test_freeze_and_unfreeze():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(b))

    f = c.freeze()
    assert f.moments == tuple(c.moments)

    ff = f.freeze()
    assert ff is f

    unf = f.unfreeze()
    assert unf.moments == c.moments
    assert unf is not c

    cc = c.unfreeze()
    assert cc is c

    fcc = cc.freeze()
    assert fcc.moments == f.moments
    assert fcc is not f


def test_equality():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(cirq.FrozenCircuit(),
                          cirq.FrozenCircuit(device=cirq.UNCONSTRAINED_DEVICE),
                          cirq.FrozenCircuit([]), cirq.FrozenCircuit(()))
    eq.add_equality_group(cirq.FrozenCircuit([cirq.Moment()]),
                          cirq.FrozenCircuit((cirq.Moment(),)))
    eq.add_equality_group(cirq.FrozenCircuit(device=cg.Foxtail))

    # Equality depends on structure and contents.
    eq.add_equality_group(cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]))
    eq.add_equality_group(cirq.FrozenCircuit([cirq.Moment([cirq.X(b)])]))
    eq.add_equality_group(
        cirq.FrozenCircuit([cirq.Moment([cirq.X(a)]),
                            cirq.Moment([cirq.X(b)])]))
    eq.add_equality_group(
        cirq.FrozenCircuit([cirq.Moment([cirq.X(a), cirq.X(b)])]))

    # Big case.
    eq.add_equality_group(
        cirq.FrozenCircuit([
            cirq.Moment([cirq.H(a), cirq.H(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.H(b)]),
        ]))
    eq.add_equality_group(
        cirq.FrozenCircuit([
            cirq.Moment([cirq.H(a)]),
            cirq.Moment([cirq.CNOT(a, b)]),
        ]))


def test_approx_eq():

    class TestDevice(cirq.Device):

        def validate_operation(self, operation: cirq.Operation) -> None:
            pass

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert not cirq.approx_eq(cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]),
                              cirq.Moment([cirq.X(a)]))

    assert cirq.approx_eq(cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]),
                          cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]))
    assert not cirq.approx_eq(cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]),
                              cirq.FrozenCircuit([cirq.Moment([cirq.X(b)])]))

    assert cirq.approx_eq(
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=1e-9)(a)])]))

    assert not cirq.approx_eq(
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)])]))
    assert cirq.approx_eq(
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        cirq.FrozenCircuit([cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)])]),
        atol=1e-6)

    assert not cirq.approx_eq(
        cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])]),
        cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])], device=TestDevice()))


def test_add_op_tree():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.FrozenCircuit()
    assert c + [cirq.X(a), cirq.Y(b)] == cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
    ])

    assert c + cirq.X(a) == cirq.FrozenCircuit(cirq.X(a))
    assert c + [cirq.X(a)] == cirq.FrozenCircuit(cirq.X(a))
    assert c + [[[cirq.X(a)], []]] == cirq.FrozenCircuit(cirq.X(a))
    assert c + (cirq.X(a),) == cirq.FrozenCircuit(cirq.X(a))
    assert c + (cirq.X(a) for _ in range(1)) == cirq.FrozenCircuit(cirq.X(a))
    with pytest.raises(TypeError):
        _ = c + cirq.X


def test_radd_op_tree():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.FrozenCircuit()
    assert [cirq.X(a), cirq.Y(b)] + c == cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
    ])

    assert cirq.X(a) + c == cirq.FrozenCircuit(cirq.X(a))
    assert [cirq.X(a)] + c == cirq.FrozenCircuit(cirq.X(a))
    assert [[[cirq.X(a)], []]] + c == cirq.FrozenCircuit(cirq.X(a))
    assert (cirq.X(a),) + c == cirq.FrozenCircuit(cirq.X(a))
    assert (cirq.X(a) for _ in range(1)) + c == cirq.FrozenCircuit(cirq.X(a))
    with pytest.raises(AttributeError):
        _ = cirq.X + c
    with pytest.raises(TypeError):
        _ = 0 + c

    # non-empty circuit addition
    d = cirq.FrozenCircuit(cirq.Y(b))
    assert [cirq.X(a)] + d == cirq.FrozenCircuit(
        [cirq.Moment([cirq.X(a)]),
         cirq.Moment([cirq.Y(b)])])
    assert cirq.Moment([cirq.X(a)]) + d == cirq.FrozenCircuit(
        [cirq.Moment([cirq.X(a)]),
         cirq.Moment([cirq.Y(b)])])

    # Preserves device.
    c = cirq.FrozenCircuit(device=cirq.google.Bristlecone)
    c2 = [] + c
    assert c2.device is cirq.google.Bristlecone
    assert c2 == c

    # Validates versus device.
    c = cirq.FrozenCircuit(device=cirq.google.Bristlecone)
    with pytest.raises(ValueError, match='Unsupported qubit'):
        _ = [cirq.X(cirq.NamedQubit('a'))] + c


def test_bool():
    assert not cirq.FrozenCircuit()
    assert cirq.FrozenCircuit(cirq.X(cirq.NamedQubit('a')))


def test_repr():
    assert repr(cirq.FrozenCircuit()) == 'cirq.FrozenCircuit()'

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.H(a), cirq.H(b)]),
        cirq.Moment(),
        cirq.Moment([cirq.CZ(a, b)]),
    ])
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == """cirq.FrozenCircuit([
    cirq.Moment(
        cirq.H(cirq.NamedQubit('a')),
        cirq.H(cirq.NamedQubit('b')),
    ),
    cirq.Moment(),
    cirq.Moment(
        cirq.CZ(cirq.NamedQubit('a'), cirq.NamedQubit('b')),
    ),
])"""

    c = cirq.FrozenCircuit(device=cg.Foxtail)
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == 'cirq.FrozenCircuit(device=cirq.google.Foxtail)'

    c = cirq.FrozenCircuit(cirq.Z(cirq.GridQubit(0, 0)), device=cg.Foxtail)
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == """cirq.FrozenCircuit([
    cirq.Moment(
        cirq.Z(cirq.GridQubit(0, 0)),
    ),
], device=cirq.google.Foxtail)"""


def test_empty_moments():
    # 1-qubit test
    op = cirq.X(cirq.NamedQubit('a'))
    op_moment = cirq.Moment([op])
    circuit = cirq.FrozenCircuit(
        [op_moment, op_moment, cirq.Moment(), op_moment])

    cirq.testing.assert_has_diagram(circuit,
                                    "a: ───X───X───────X───",
                                    use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit,
                                    """
a
│
X
│
X
│
│
│
X
│
""",
                                    use_unicode_characters=True,
                                    transpose=True)

    # 1-qubit ascii-only test
    cirq.testing.assert_has_diagram(circuit,
                                    "a: ---X---X-------X---",
                                    use_unicode_characters=False)
    cirq.testing.assert_has_diagram(circuit,
                                    """
a
|
X
|
X
|
|
|
X
|
""",
                                    use_unicode_characters=False,
                                    transpose=True)

    # 2-qubit test
    op = cirq.CNOT(cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    op_moment = cirq.Moment([op])
    circuit = cirq.FrozenCircuit(
        [op_moment, op_moment, cirq.Moment(), op_moment])

    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ───@───@───────@───
      │   │       │
b: ───X───X───────X───""",
                                    use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit,
                                    """
a b
│ │
@─X
│ │
@─X
│ │
│ │
│ │
@─X
│ │
""",
                                    use_unicode_characters=True,
                                    transpose=True)

    # 2-qubit ascii-only test
    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ---@---@-------@---
      |   |       |
b: ---X---X-------X---""",
                                    use_unicode_characters=False)
    cirq.testing.assert_has_diagram(circuit,
                                    """
a b
| |
@-X
| |
@-X
| |
| |
| |
@-X
| |
""",
                                    use_unicode_characters=False,
                                    transpose=True)


def test_symbol_addition_in_gate_exponent():
    # 1-qubit test
    qubit = cirq.NamedQubit('a')
    circuit = cirq.FrozenCircuit(
        cirq.X(qubit)**0.5,
        cirq.YPowGate(exponent=sympy.Symbol('a') + sympy.Symbol('b')).on(qubit))
    cirq.testing.assert_has_diagram(circuit,
                                    'a: ───X^0.5───Y^(a + b)───',
                                    use_unicode_characters=True)

    cirq.testing.assert_has_diagram(circuit,
                                    """
a
│
X^0.5
│
Y^(a + b)
│
""",
                                    use_unicode_characters=True,
                                    transpose=True)

    cirq.testing.assert_has_diagram(circuit,
                                    'a: ---X^0.5---Y^(a + b)---',
                                    use_unicode_characters=False)

    cirq.testing.assert_has_diagram(circuit,
                                    """
a
|
X^0.5
|
Y^(a + b)
|

 """,
                                    use_unicode_characters=False,
                                    transpose=True)


def test_slice():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.H(a), cirq.H(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.H(b)]),
    ])
    assert c[0:1] == cirq.FrozenCircuit([cirq.Moment([cirq.H(a), cirq.H(b)])])
    assert c[::2] == cirq.FrozenCircuit(
        [cirq.Moment([cirq.H(a), cirq.H(b)]),
         cirq.Moment([cirq.H(b)])])
    assert c[0:1:2] == cirq.FrozenCircuit([cirq.Moment([cirq.H(a), cirq.H(b)])])
    assert c[1:3:] == cirq.FrozenCircuit(
        [cirq.Moment([cirq.CZ(a, b)]),
         cirq.Moment([cirq.H(b)])])
    assert c[::-1] == cirq.FrozenCircuit([
        cirq.Moment([cirq.H(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.H(a), cirq.H(b)])
    ])
    assert c[3:0:-1] == cirq.FrozenCircuit(
        [cirq.Moment([cirq.H(b)]),
         cirq.Moment([cirq.CZ(a, b)])])
    assert c[0:2:-1] == cirq.FrozenCircuit()


def test_with_device():
    c = cirq.FrozenCircuit(cirq.X(cirq.LineQubit(0)))
    c2 = c.with_device(cg.Foxtail, lambda e: cirq.GridQubit(e.x, 0))
    assert c2 == cirq.FrozenCircuit(cirq.X(cirq.GridQubit(0, 0)),
                                    device=cg.Foxtail)

    # Qubit type must be correct.
    c = cirq.FrozenCircuit(cirq.X(cirq.LineQubit(0)))
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        _ = c.with_device(cg.Foxtail)

    # Operations must be compatible from the start
    c = cirq.FrozenCircuit(cirq.X(cirq.GridQubit(0, 0)))
    _ = c.with_device(cg.Foxtail)
    c = cirq.FrozenCircuit(cirq.H(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError, match='Unsupported gate type'):
        _ = c.with_device(cg.Foxtail)

    # Some qubits exist on multiple devices.
    c = cirq.FrozenCircuit(cirq.X(cirq.GridQubit(0, 0)), device=cg.Foxtail)
    with pytest.raises(ValueError):
        _ = c.with_device(cg.Bristlecone)
    c = cirq.FrozenCircuit(cirq.X(cirq.GridQubit(0, 6)), device=cg.Foxtail)
    _ = c.with_device(cg.Bristlecone)


def test_multiply():
    a = cirq.NamedQubit('a')

    c = cirq.FrozenCircuit()
    d = cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])])

    assert c * 0 == cirq.FrozenCircuit()
    assert d * 0 == cirq.FrozenCircuit()
    assert d * 2 == cirq.FrozenCircuit(
        [cirq.Moment([cirq.X(a)]),
         cirq.Moment([cirq.X(a)])])
    assert 1 * c == cirq.FrozenCircuit()
    assert -1 * d == cirq.FrozenCircuit()
    assert 1 * d == cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])])

    d *= 3
    assert d == cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(a)])
    ])

    with pytest.raises(TypeError):
        _ = c * 'a'
    with pytest.raises(TypeError):
        _ = 'a' * c
    with pytest.raises(TypeError):
        c *= 'a'


def test_container_methods():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.H(a), cirq.H(b)]),
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment([cirq.H(b)]),
    ])
    assert list(c) == list(c._moments)
    # __iter__
    assert list(iter(c)) == list(c._moments)
    # __reversed__ for free.
    assert list(reversed(c)) == list(reversed(c._moments))
    # __contains__ for free.
    assert cirq.Moment([cirq.H(b)]) in c

    assert len(c) == 3


def test_bad_index():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit([cirq.Moment([cirq.H(a), cirq.H(b)])])
    with pytest.raises(TypeError):
        _ = c['string']


def test_next_moment_operating_on():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.FrozenCircuit()
    assert c.next_moment_operating_on([a]) is None
    assert c.next_moment_operating_on([a], 0) is None
    assert c.next_moment_operating_on([a], 102) is None

    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])])
    assert c.next_moment_operating_on([a]) == 0
    assert c.next_moment_operating_on([a], 0) == 0
    assert c.next_moment_operating_on([a, b]) == 0
    assert c.next_moment_operating_on([a], 1) is None
    assert c.next_moment_operating_on([b]) is None

    c = cirq.FrozenCircuit([
        cirq.Moment(),
        cirq.Moment([cirq.X(a)]),
        cirq.Moment(),
        cirq.Moment([cirq.CZ(a, b)])
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

    c = cirq.FrozenCircuit([
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment([cirq.X(a)]),
        cirq.Moment(),
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

    c = cirq.FrozenCircuit()
    assert c.prev_moment_operating_on([a]) is None
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([a], 102) is None

    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])])
    assert c.prev_moment_operating_on([a]) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a, b]) == 0
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([b]) is None

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.CZ(a, b)]),
        cirq.Moment(),
        cirq.Moment([cirq.X(a)]),
        cirq.Moment(),
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

    c = cirq.FrozenCircuit([
        cirq.Moment(),
        cirq.Moment([cirq.X(a)]),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
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

    c = cirq.FrozenCircuit()
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, -1) is None
    assert c.operation_at(a, 102) is None

    c = cirq.FrozenCircuit([cirq.Moment()])
    assert c.operation_at(a, 0) is None

    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(a)])])
    assert c.operation_at(b, 0) is None
    assert c.operation_at(a, 1) is None
    assert c.operation_at(a, 0) == cirq.X(a)

    c = cirq.FrozenCircuit([cirq.Moment(), cirq.Moment([cirq.CZ(a, b)])])
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

    c = cirq.FrozenCircuit()
    assert list(c.findall_operations(is_x)) == []

    c = cirq.FrozenCircuit(xa)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = cirq.FrozenCircuit(za)
    assert list(c.findall_operations(is_x)) == []

    c = cirq.FrozenCircuit([za, zb] * 8)
    assert list(c.findall_operations(is_x)) == []

    c = cirq.FrozenCircuit(xa, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (0, xb)]

    c = cirq.FrozenCircuit(xa, zb)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = cirq.FrozenCircuit(xa, za)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = cirq.FrozenCircuit([xa] * 8)
    assert list(c.findall_operations(is_x)) == list(enumerate([xa] * 8))

    c = cirq.FrozenCircuit(za, zb, xa, xb)
    assert list(c.findall_operations(is_x)) == [(1, xa), (1, xb)]

    c = cirq.FrozenCircuit(xa, zb, za, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (1, xb)]


def test_findall_operations_with_gate():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit([
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
        (4, cirq.MeasurementGate(1,
                                 key='a').on(a), cirq.MeasurementGate(1,
                                                                      key='a')),
        (4, cirq.MeasurementGate(1,
                                 key='b').on(b), cirq.MeasurementGate(1,
                                                                      key='b')),
    ]


def assert_findall_operations_until_blocked_as_expected(circuit=None,
                                                        start_frontier=None,
                                                        is_blocker=None,
                                                        expected_ops=None):
    if circuit is None:
        circuit = cirq.FrozenCircuit()
    if start_frontier is None:
        start_frontier = {}
    kwargs = {} if is_blocker is None else {'is_blocker': is_blocker}
    found_ops = circuit.findall_operations_until_blocked(
        start_frontier, **kwargs)

    for i, op in found_ops:
        assert i >= min(
            (start_frontier[q] for q in op.qubits if q in start_frontier),
            default=0)
        assert set(op.qubits).intersection(start_frontier)

    if expected_ops is None:
        return
    assert sorted(found_ops) == sorted(expected_ops)


def test_findall_operations_until_blocked():
    a, b, c, d = cirq.LineQubit.range(4)

    assert_findall_operations_until_blocked_as_expected()

    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CZ(a, b), cirq.H(b),
                                 cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d),
                                 cirq.H(d), cirq.CZ(c, d), cirq.H(c),
                                 cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b),
                                 cirq.H(a))
    expected_diagram = """
0: ───H───@───────────────────────────────────────@───H───
          │                                       │
1: ───────@───H───@───────────────────────@───H───@───────
                  │                       │
2: ───────────────@───H───@───────@───H───@───────────────
                          │       │
3: ───────────────────────@───H───@───────────────────────
""".strip()
    #     0   1   2   3   4   5   6   7   8   9   10  11  12
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    # Always return true to test basic features
    go_to_end = lambda op: False
    stop_if_op = lambda op: True
    stop_if_h_on_a = lambda op: op.gate == cirq.H and a in op.qubits

    # Empty cases.
    assert_findall_operations_until_blocked_as_expected(is_blocker=go_to_end,
                                                        expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(circuit=circuit,
                                                        is_blocker=go_to_end,
                                                        expected_ops=[])

    # Clamped input cases. (out of bounds)
    assert_findall_operations_until_blocked_as_expected(start_frontier={a: 5},
                                                        is_blocker=stop_if_op,
                                                        expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(
        start_frontier={a: -100}, is_blocker=stop_if_op, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(circuit=circuit,
                                                        start_frontier={a: 100},
                                                        is_blocker=stop_if_op,
                                                        expected_ops=[])

    # Test if all operations are blocked
    for idx in range(0, 15):
        for q in (a, b, c, d):
            assert_findall_operations_until_blocked_as_expected(
                circuit=circuit,
                start_frontier={q: idx},
                is_blocker=stop_if_op,
                expected_ops=[])
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={
                a: idx,
                b: idx,
                c: idx,
                d: idx
            },
            is_blocker=stop_if_op,
            expected_ops=[])

    # Cases where nothing is blocked, it goes to the end
    a_ending_ops = [(11, cirq.CZ.on(a, b)), (12, cirq.H.on(a))]
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={a: idx},
            is_blocker=go_to_end,
            expected_ops=a_ending_ops)

    # Block on H, but pick up the CZ
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={a: idx},
            is_blocker=stop_if_h_on_a,
            expected_ops=[(11, cirq.CZ.on(a, b))])

    circuit = cirq.FrozenCircuit([cirq.CZ(a, b), cirq.CZ(a, b), cirq.CZ(b, c)])
    expected_diagram = """
0: ───@───@───────
      │   │
1: ───@───@───@───
              │
2: ───────────@───
""".strip()
    #     0   1   2
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0}
    is_blocker = lambda next_op: sorted(next_op.qubits) != [a, b]
    expected_ops = [(0, cirq.CZ(a, b)), (1, cirq.CZ(a, b))]
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=expected_ops)

    circuit = cirq.FrozenCircuit([cirq.ZZ(a, b), cirq.ZZ(b, c)])
    expected_diagram = """
0: ───ZZ────────
      │
1: ───ZZ───ZZ───
           │
2: ────────ZZ───
""".strip()
    #     0    1
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0, c: 0}
    is_blocker = lambda op: a in op.qubits
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=[])

    circuit = cirq.FrozenCircuit(
        [cirq.ZZ(a, b), cirq.XX(c, d),
         cirq.ZZ(b, c), cirq.Z(b)])
    expected_diagram = """
0: ───ZZ────────────
      │
1: ───ZZ───ZZ───Z───
           │
2: ───XX───ZZ───────
      │
3: ───XX────────────
""".strip()
    #     0    1    2
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0, c: 0, d: 0}
    is_blocker = lambda op: isinstance(op.gate, cirq.XXPowGate)
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=[(0, cirq.ZZ(a, b))])

    circuit = cirq.FrozenCircuit(
        [cirq.XX(a, b),
         cirq.Z(a),
         cirq.ZZ(b, c),
         cirq.ZZ(c, d),
         cirq.Z(d)])
    expected_diagram = """
0: ───XX───Z─────────────
      │
1: ───XX───ZZ────────────
           │
2: ────────ZZ───ZZ───────
                │
3: ─────────────ZZ───Z───
""".strip()
    #     0    1    2    3
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, d: 0}
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=[])


@pytest.mark.parametrize('seed', [randint(0, 2**31)])
def test_findall_operations_until_blocked_docstring_examples(seed):
    prng = np.random.RandomState(seed)

    class ExampleGate(cirq.Gate):

        def __init__(self, n_qubits, label):
            self.n_qubits = n_qubits
            self.label = label

        def num_qubits(self):
            return self.n_qubits

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(wire_symbols=[self.label] *
                                           self.n_qubits)

    def is_blocker(op):
        if op.gate.label == 'F':
            return False
        if op.gate.label == 'T':
            return True
        return prng.rand() < 0.5

    F2 = ExampleGate(2, 'F')
    T2 = ExampleGate(2, 'T')
    M2 = ExampleGate(2, 'M')
    a, b, c, d = cirq.LineQubit.range(4)

    circuit = cirq.FrozenCircuit([F2(a, b), F2(a, b), T2(b, c)])
    start = {a: 0, b: 0}
    expected_diagram = """
0: ───F───F───────
      │   │
1: ───F───F───T───
              │
2: ───────────T───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    expected_ops = [(0, F2(a, b)), (1, F2(a, b))]
    new_circuit = cirq.FrozenCircuit(op for _, op in expected_ops)
    expected_diagram = """
0: ───F───F───
      │   │
1: ───F───F───
    """
    cirq.testing.assert_has_diagram(new_circuit, expected_diagram)
    assert (circuit.findall_operations_until_blocked(
        start, is_blocker) == expected_ops)

    circuit = cirq.FrozenCircuit([M2(a, b), M2(b, c), F2(a, b), M2(c, d)])
    start = {a: 2, b: 2}
    expected_diagram = """
0: ───M───────F───
      │       │
1: ───M───M───F───
          │
2: ───────M───M───
              │
3: ───────────M───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    expected_ops = [(2, F2(a, b))]
    new_circuit = cirq.FrozenCircuit(op for _, op in expected_ops)
    expected_diagram = """
0: ───F───
      │
1: ───F───
    """
    cirq.testing.assert_has_diagram(new_circuit, expected_diagram)
    assert (circuit.findall_operations_until_blocked(
        start, is_blocker) == expected_ops)

    circuit = cirq.FrozenCircuit([M2(a, b), T2(b, c), M2(a, b), M2(c, d)])
    start = {a: 1, b: 1}
    expected_diagram = """
0: ───M───────M───
      │       │
1: ───M───T───M───
          │
2: ───────T───M───
              │
3: ───────────M───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == []

    ops = [(0, F2(a, b)), (1, F2(a, b))]
    circuit = cirq.FrozenCircuit(op for _, op in ops)
    start = {a: 0, b: 1}
    expected_diagram = """
0: ───F───F───
      │   │
1: ───F───F───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == ops

    ops = [F2(a, b), F2(b, c), F2(c, d)]
    circuit = cirq.FrozenCircuit(ops)
    start = {a: 0, d: 0}
    expected_diagram = """
0: ───F───────────
      │
1: ───F───F───────
          │
2: ───────F───F───
              │
3: ───────────F───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert (circuit.findall_operations_until_blocked(start, is_blocker) == [
        (0, F2(a, b)), (2, F2(c, d))
    ])


def test_has_measurements():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ma = cirq.measure(a)
    mb = cirq.measure(b)

    c = cirq.FrozenCircuit()
    assert not c.has_measurements()

    c = cirq.FrozenCircuit(xa, xb)
    assert not c.has_measurements()

    c = cirq.FrozenCircuit(ma)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(ma, mb)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(xa, ma)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(xa, ma, xb, mb)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(ma, xa)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(ma, xa, mb)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(xa, ma, xb, xa)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(ma, ma)
    assert c.has_measurements()

    c = cirq.FrozenCircuit(xa, ma, xa)
    assert c.has_measurements()


def test_are_all_measurements_terminal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ma = cirq.measure(a)
    mb = cirq.measure(b)

    c = cirq.FrozenCircuit()
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(xa, xb)
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(ma)
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(ma, mb)
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(xa, ma)
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(xa, ma, xb, mb)
    assert c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(ma, xa)
    assert not c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(ma, xa, mb)
    assert not c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(xa, ma, xb, xa)
    assert not c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(ma, ma)
    assert not c.are_all_measurements_terminal()

    c = cirq.FrozenCircuit(xa, ma, xa)
    assert not c.are_all_measurements_terminal()


def test_all_terminal():

    def is_x_pow_gate(op):
        return isinstance(op.gate, cirq.XPowGate)

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ya = cirq.Y.on(a)
    yb = cirq.Y.on(b)

    c = cirq.FrozenCircuit()
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(ya)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(ya, yb)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(ya, yb, xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(ya, yb, xa, xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xa, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xa, ya)
    assert not c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xb, ya, yb)
    assert not c.are_all_matches_terminal(is_x_pow_gate)

    c = cirq.FrozenCircuit(xa, ya, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)


def test_all_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(b)]),
    ])
    assert c.all_qubits() == {a, b}

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(a)]),
    ])
    assert c.all_qubits() == {a}

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.CZ(a, b)]),
    ])
    assert c.all_qubits() == {a, b}

    c = cirq.FrozenCircuit(
        [cirq.Moment([cirq.CZ(a, b)]),
         cirq.Moment([cirq.X(a)])])
    assert c.all_qubits() == {a, b}


def test_all_operations():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(b)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a), cirq.X(b)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(a)]),
    ])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(a)]

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.CZ(a, b)]),
    ])
    assert list(c.all_operations()) == [cirq.CZ(a, b)]

    c = cirq.FrozenCircuit(
        [cirq.Moment([cirq.CZ(a, b)]),
         cirq.Moment([cirq.X(a)])])
    assert list(c.all_operations()) == [cirq.CZ(a, b), cirq.X(a)]

    c = cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([]),
        cirq.Moment([cirq.CNOT(a, b)]),
        cirq.Moment([cirq.Z(b), cirq.H(a)]),  # Different qubit order
        cirq.Moment([])
    ])

    assert list(c.all_operations()) == [
        cirq.X(a), cirq.Y(b),
        cirq.CNOT(a, b),
        cirq.Z(b), cirq.H(a)
    ]


def test_qid_shape_qubit():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.FrozenCircuit([
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.X(b)]),
    ])

    assert cirq.qid_shape(circuit) == (2, 2)
    assert cirq.num_qubits(circuit) == 2
    assert circuit.qid_shape() == (2, 2)
    assert circuit.qid_shape(qubit_order=[c, a, b]) == (2, 2, 2)
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[a])


def test_qid_shape_qudit():

    class PlusOneMod3Gate(cirq.SingleQubitGate):

        def _qid_shape_(self):
            return (3,)

    class C2NotGate(cirq.Gate):

        def _qid_shape_(self):
            return (3, 2)

    class IdentityGate(cirq.SingleQubitGate):

        def _qid_shape_(self):
            return (1,)

    a, b, c = cirq.LineQid.for_qid_shape((3, 2, 1))

    circuit = cirq.FrozenCircuit(
        PlusOneMod3Gate().on(a),
        C2NotGate().on(a, b),
        IdentityGate().on_each(c),
    )

    assert cirq.num_qubits(circuit) == 3
    assert cirq.qid_shape(circuit) == (3, 2, 1)
    assert circuit.qid_shape() == (3, 2, 1)
    assert circuit.qid_shape()
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[b, c])


def test_to_text_diagram_teleportation_to_diagram():
    ali = cirq.NamedQubit('(0, 0)')
    bob = cirq.NamedQubit('(0, 1)')
    msg = cirq.NamedQubit('(1, 0)')
    tmp = cirq.NamedQubit('(1, 1)')

    c = cirq.FrozenCircuit([
        cirq.Moment([cirq.H(ali)]),
        cirq.Moment([cirq.CNOT(ali, bob)]),
        cirq.Moment([cirq.X(msg)**0.5]),
        cirq.Moment([cirq.CNOT(msg, ali)]),
        cirq.Moment([cirq.H(msg)]),
        cirq.Moment([cirq.measure(msg), cirq.measure(ali)]),
        cirq.Moment([cirq.CNOT(ali, bob)]),
        cirq.Moment([cirq.CNOT(msg, tmp)]),
        cirq.Moment([cirq.CZ(bob, tmp)]),
    ])

    cirq.testing.assert_has_diagram(
        c, """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────@───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───@───
""")

    cirq.testing.assert_has_diagram(c,
                                    """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------@---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---@---
""",
                                    use_unicode_characters=False)

    cirq.testing.assert_has_diagram(c,
                                    """
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
""",
                                    use_unicode_characters=False,
                                    transpose=True)


def test_diagram_with_unknown_exponent():

    class WeirdGate(cirq.SingleQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('B',),
                                           exponent='fancy')

    class WeirderGate(cirq.SingleQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('W',),
                                           exponent='fancy-that')

    c = cirq.FrozenCircuit(
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

        def __init__(self, num_qubits=1):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'

    # Fallback to repr.
    f = FGate()
    cirq.testing.assert_has_diagram(cirq.FrozenCircuit([
        cirq.Moment([f.on(q)]),
    ]),
                                    """
(0, 0): ---python-object-FGate:arbitrary-digits---
""",
                                    use_unicode_characters=False)

    f3 = FGate(3)
    # When used on multiple qubits, show the qubit order as a digit suffix.
    cirq.testing.assert_has_diagram(cirq.FrozenCircuit([
        cirq.Moment([f3.on(q, q3, q2)]),
    ]),
                                    """
(0, 0): ---python-object-FGate:arbitrary-digits---
           |
(0, 1): ---#3-------------------------------------
           |
(0, 2): ---#2-------------------------------------
""",
                                    use_unicode_characters=False)


def test_to_text_diagram_multi_qubit_gate():
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = cirq.FrozenCircuit(cirq.measure(q1, q2, q3, key='msg'))
    cirq.testing.assert_has_diagram(
        c, """
(0, 0): ───M('msg')───
           │
(0, 1): ───M──────────
           │
(0, 2): ───M──────────
""")
    cirq.testing.assert_has_diagram(c,
                                    """
(0, 0): ---M('msg')---
           |
(0, 1): ---M----------
           |
(0, 2): ---M----------
""",
                                    use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c,
                                    """
(0, 0)   (0, 1) (0, 2)
│        │      │
M('msg')─M──────M
│        │      │
""",
                                    transpose=True)


def test_to_text_diagram_many_qubits_gate_but_multiple_wire_symbols():

    class BadGate(cirq.ThreeQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> Tuple[str, str]:
            return 'a', 'a'

    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = cirq.FrozenCircuit([cirq.Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()


def test_to_text_diagram_parameterized_value():
    q = cirq.NamedQubit('cube')

    class PGate(cirq.SingleQubitGate):

        def __init__(self, val):
            self.val = val

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(('P',), self.val)

    c = cirq.FrozenCircuit(
        PGate(1).on(q),
        PGate(2).on(q),
        PGate(sympy.Symbol('a')).on(q),
        PGate(sympy.Symbol('%$&#*(')).on(q),
    )
    assert str(c).strip() == 'cube: ───P───P^2───P^a───P^(%$&#*()───'


def test_to_text_diagram_custom_order():
    qa = cirq.NamedQubit('2')
    qb = cirq.NamedQubit('3')
    qc = cirq.NamedQubit('4')

    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(qa), cirq.X(qb), cirq.X(qc)])])
    cirq.testing.assert_has_diagram(
        c,
        """
3: ---X---

4: ---X---

2: ---X---
""",
        qubit_order=cirq.QubitOrder.sorted_by(lambda e: int(str(e)) % 3),
        use_unicode_characters=False)


def test_overly_precise_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(qa)**0.12345678])])
    cirq.testing.assert_has_diagram(c,
                                    """
a: ---X^0.123---
""",
                                    use_unicode_characters=False)


def test_none_precision_diagram():
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(qa)**0.4921875])])
    cirq.testing.assert_has_diagram(c,
                                    """
a: ---X^0.4921875---
""",
                                    use_unicode_characters=False,
                                    precision=None)


def test_diagram_custom_precision():
    qa = cirq.NamedQubit('a')
    c = cirq.FrozenCircuit([cirq.Moment([cirq.X(qa)**0.12341234])])
    cirq.testing.assert_has_diagram(c,
                                    """
a: ---X^0.12341---
""",
                                    use_unicode_characters=False,
                                    precision=5)


def test_diagram_wgate():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(exponent=0.12341234,
                                     phase_exponent=0.43214321)
    c = cirq.FrozenCircuit([cirq.Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(c,
                                    """
a: ---PhX(0.43)^(1/8)---
""",
                                    use_unicode_characters=False,
                                    precision=2)


def test_diagram_wgate_none_precision():
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(exponent=0.12341234,
                                     phase_exponent=0.43214321)
    c = cirq.FrozenCircuit([cirq.Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(c,
                                    """
a: ---PhX(0.43214321)^0.12341234---
""",
                                    use_unicode_characters=False,
                                    precision=None)


def test_diagram_global_phase():
    qa = cirq.NamedQubit('a')
    global_phase = cirq.GlobalPhaseOperation(coefficient=1j)
    c = cirq.FrozenCircuit([global_phase])
    cirq.testing.assert_has_diagram(c,
                                    "\n\nglobal phase:   0.5pi",
                                    use_unicode_characters=False,
                                    precision=2)
    cirq.testing.assert_has_diagram(c,
                                    "\n\nglobal phase:   0.5π",
                                    use_unicode_characters=True,
                                    precision=2)

    c = cirq.FrozenCircuit([cirq.X(qa), global_phase, global_phase])
    cirq.testing.assert_has_diagram(c,
                                    """\
a: ─────────────X───

global phase:   π""",
                                    use_unicode_characters=True,
                                    precision=2)
    c = cirq.FrozenCircuit([cirq.X(qa), global_phase],
                           cirq.Moment([cirq.X(qa), global_phase]))
    cirq.testing.assert_has_diagram(c,
                                    """\
a: ─────────────X──────X──────

global phase:   0.5π   0.5π
""",
                                    use_unicode_characters=True,
                                    precision=2)


def test_has_unitary():

    class NonUnitary(cirq.SingleQubitGate):
        pass

    class EventualUnitary(cirq.SingleQubitGate):

        def _decompose_(self, qubits):
            return cirq.X.on_each(*qubits)

    q = cirq.NamedQubit('q')

    # Non-unitary operations cause a non-unitary circuit.
    assert cirq.has_unitary(cirq.FrozenCircuit(cirq.X(q)))
    assert not cirq.has_unitary(cirq.FrozenCircuit(NonUnitary().on(q)))

    # Terminal measurements are ignored, though.
    assert cirq.has_unitary(cirq.FrozenCircuit(cirq.measure(q)))
    assert not cirq.has_unitary(
        cirq.FrozenCircuit(cirq.measure(q), cirq.measure(q)))

    # Still unitary if operations decompose into unitary operations.
    assert cirq.has_unitary(cirq.FrozenCircuit(EventualUnitary().on(q)))


def test_text_diagram_jupyter():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = cirq.FrozenCircuit(
        (cirq.CNOT(a, b), cirq.CNOT(b, c), cirq.CNOT(c, a)) * 50)
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
    assert p.text_pretty == 'FrozenCircuit(...)'

    # Test Jupyter notebook html output
    text_html = circuit._repr_html_()
    # Don't enforce specific html surrounding the diagram content
    assert text_expected in text_html


def test_circuit_to_unitary_matrix():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Single qubit gates.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.X(a)**0.5).unitary(),
                                                    np.array([
                                                        [1j, 1],
                                                        [1, 1j],
                                                    ]) * np.sqrt(0.5),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit(cirq.Y(a)**0.25).unitary(),
        cirq.unitary(cirq.Y(a)**0.25),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.Z(a), cirq.X(b)).unitary(),
                                                    np.array([
                                                        [0, 1, 0, 0],
                                                        [1, 0, 0, 0],
                                                        [0, 0, 0, -1],
                                                        [0, 0, -1, 0],
                                                    ]),
                                                    atol=1e-8)

    # Single qubit gates and two qubit gate.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.Z(a), cirq.X(b), cirq.CNOT(a, b)).unitary(),
                                                    np.array([
                                                        [0, 1, 0, 0],
                                                        [1, 0, 0, 0],
                                                        [0, 0, -1, 0],
                                                        [0, 0, 0, -1],
                                                    ]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.H(b),
        cirq.CNOT(b, a)**0.5,
        cirq.Y(a)**0.5).unitary(),
                                                    np.array([
                                                        [1, 1, -1, -1],
                                                        [1j, -1j, -1j, 1j],
                                                        [1, 1, 1, 1],
                                                        [1, -1, 1, -1],
                                                    ]) * np.sqrt(0.25),
                                                    atol=1e-8)

    # Measurement gate has no corresponding matrix.
    c = cirq.FrozenCircuit(cirq.measure(a))
    with pytest.raises(ValueError):
        _ = c.unitary(ignore_terminal_measurements=False)

    # Ignoring terminal measurements.
    c = cirq.FrozenCircuit(cirq.measure(a))
    cirq.testing.assert_allclose_up_to_global_phase(c.unitary(),
                                                    np.eye(2),
                                                    atol=1e-8)

    # Ignoring terminal measurements with further cirq.
    c = cirq.FrozenCircuit(cirq.Z(a), cirq.measure(a), cirq.Z(b))
    cirq.testing.assert_allclose_up_to_global_phase(c.unitary(),
                                                    np.array([[1, 0, 0, 0],
                                                              [0, -1, 0, 0],
                                                              [0, 0, -1, 0],
                                                              [0, 0, 0, 1]]),
                                                    atol=1e-8)

    # Optionally don't ignoring terminal measurements.
    c = cirq.FrozenCircuit(cirq.measure(a))
    with pytest.raises(ValueError, match="measurement"):
        _ = c.unitary(ignore_terminal_measurements=False),

    # Non-terminal measurements are not ignored.
    c = cirq.FrozenCircuit(cirq.measure(a), cirq.X(a))
    with pytest.raises(ValueError):
        _ = c.unitary()

    # Non-terminal measurements are not ignored (multiple qubits).
    c = cirq.FrozenCircuit(cirq.measure(a), cirq.measure(b), cirq.CNOT(a, b))
    with pytest.raises(ValueError):
        _ = c.unitary()

    # Gates without matrix or decomposition raise exception
    class MysteryGate(cirq.TwoQubitGate):
        pass

    c = cirq.FrozenCircuit(MysteryGate()(a, b))
    with pytest.raises(TypeError):
        _ = c.unitary()

    # Accounts for measurement bit flipping.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.measure(a, invert_mask=(True,))).unitary(),
                                                    cirq.unitary(cirq.X),
                                                    atol=1e-8)

    # dtype
    c = cirq.FrozenCircuit(cirq.X(a))
    assert c.unitary(dtype=np.complex64).dtype == np.complex64
    assert c.unitary(dtype=np.complex128).dtype == np.complex128
    assert c.unitary(dtype=np.float64).dtype == np.float64


def test_circuit_unitary():
    q = cirq.NamedQubit('q')

    with_inner_measure = cirq.FrozenCircuit(cirq.H(q), cirq.measure(q),
                                            cirq.H(q))
    assert not cirq.has_unitary(with_inner_measure)
    assert cirq.unitary(with_inner_measure, None) is None

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(
        cirq.FrozenCircuit(cirq.X(q)**0.5),
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
    c = cirq.FrozenCircuit(cirq.CNOT(a, b), cirq.Z(b), cirq.CNOT(a, b))
    assert cirq.has_unitary(c)
    m = c.unitary()
    cirq.testing.assert_allclose_up_to_global_phase(m,
                                                    np.array([
                                                        [1, 0, 0, 0],
                                                        [0, -1, 0, 0],
                                                        [0, 0, -1, 0],
                                                        [0, 0, 0, 1],
                                                    ]),
                                                    atol=1e-8)

    # 2-qubit matrix matches when qubits in order.
    for expected in [np.diag([1, 1j, -1, -1j]), cirq.unitary(cirq.CNOT)]:

        class Passthrough(cirq.TwoQubitGate):

            def _unitary_(self) -> np.ndarray:
                return expected

        c = cirq.FrozenCircuit(Passthrough()(a, b))
        m = c.unitary()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected, atol=1e-8)


def test_composite_gate_to_unitary_matrix():

    class CnotComposite(cirq.TwoQubitGate):

        def _decompose_(self, qubits):
            q0, q1 = qubits
            return cirq.Y(q1)**-0.5, cirq.CZ(q0, q1), cirq.Y(q1)**0.5

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.FrozenCircuit(cirq.X(a),
                           CnotComposite()(a, b), cirq.X(a), cirq.measure(a),
                           cirq.X(b), cirq.measure(b))
    assert cirq.has_unitary(c)

    mat = c.unitary()
    mat_expected = cirq.unitary(cirq.CNOT)

    cirq.testing.assert_allclose_up_to_global_phase(mat,
                                                    mat_expected,
                                                    atol=1e-8)


def test_expanding_gate_symbols():

    class MultiTargetCZ(cirq.Gate):

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> Tuple[str, ...]:
            assert args.known_qubit_count is not None
            return ('@',) + ('Z',) * (args.known_qubit_count - 1)

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    t0 = cirq.FrozenCircuit(MultiTargetCZ(1).on(c))
    t1 = cirq.FrozenCircuit(MultiTargetCZ(2).on(c, a))
    t2 = cirq.FrozenCircuit(MultiTargetCZ(3).on(c, a, b))

    cirq.testing.assert_has_diagram(t0, """
c: ───@───
""")

    cirq.testing.assert_has_diagram(t1, """
a: ───Z───
      │
c: ───@───
""")

    cirq.testing.assert_has_diagram(
        t2, """
a: ───Z───
      │
b: ───Z───
      │
c: ───@───
""")


def test_transposed_diagram_exponent_order():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.CZ(a, b)**-0.5,
        cirq.CZ(a, c)**0.5,
        cirq.CZ(b, c)**0.125,
    )
    cirq.testing.assert_has_diagram(circuit,
                                    """
0 1      2
│ │      │
@─@^-0.5 │
│ │      │
@─┼──────@^0.5
│ │      │
│ @──────@^(1/8)
│ │      │
""",
                                    transpose=True)


def test_apply_unitary_effect_to_state():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # State ordering.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit(cirq.X(a)**0.5).final_state_vector(),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit(cirq.X(a)**0.5).final_state_vector(initial_state=0),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit(cirq.X(a)**0.5).final_state_vector(initial_state=1),
        np.array([1, 1j]) * np.sqrt(0.5),
        atol=1e-8)

    # Vector state.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.X(a)**0.5).final_state_vector(initial_state=np.array([1j, 1]) *
                                           np.sqrt(0.5)),
                                                    np.array([0, 1]),
                                                    atol=1e-8)

    # Qubit ordering.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=0),
                                                    np.array([1, 0, 0, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=1),
                                                    np.array([0, 1, 0, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=2),
                                                    np.array([0, 0, 0, 1]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=3),
                                                    np.array([0, 0, 1, 0]),
                                                    atol=1e-8)

    # Product state
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ZERO(a) *
                                            cirq.KET_ZERO(b)),
                                                    np.array([1, 0, 0, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ZERO(a) *
                                            cirq.KET_ONE(b)),
                                                    np.array([0, 1, 0, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ONE(a) *
                                            cirq.KET_ZERO(b)),
                                                    np.array([0, 0, 0, 1]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ONE(a) *
                                            cirq.KET_ONE(b)),
                                                    np.array([0, 0, 1, 0]),
                                                    atol=1e-8)

    # Measurements.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.measure(a)).final_state_vector(),
                                                    np.array([1, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.X(a), cirq.measure(a)).final_state_vector(),
                                                    np.array([0, 1]),
                                                    atol=1e-8)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
            cirq.measure(a), cirq.X(a)).final_state_vector(),
                                                        np.array([1, 0]),
                                                        atol=1e-8)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
            cirq.measure(a)).final_state_vector(
                ignore_terminal_measurements=False),
                                                        np.array([1, 0]),
                                                        atol=1e-8)

    # Extra qubits.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit().final_state_vector(), np.array([1]), atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.FrozenCircuit().final_state_vector(
            qubits_that_should_be_present=[a]),
        np.array([1, 0]),
        atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.X(b)).final_state_vector(qubits_that_should_be_present=[a]),
                                                    np.array([0, 1, 0, 0]),
                                                    atol=1e-8)

    # Qubit order.
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[a, b]),
                                                    np.array([0, 1, 0, 0]),
                                                    atol=1e-8)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
        cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[b, a]),
                                                    np.array([0, 0, 1, 0]),
                                                    atol=1e-8)

    # Dtypes.
    dtypes = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):  # Some systems don't support 128 bit floats.
        dtypes.append(np.complex256)
    for dt in dtypes:
        cirq.testing.assert_allclose_up_to_global_phase(cirq.FrozenCircuit(
            cirq.X(a)**0.5).final_state_vector(initial_state=np.array([1j, 1]) *
                                               np.sqrt(0.5),
                                               dtype=dt),
                                                        np.array([0, 1]),
                                                        atol=1e-8)


def test_is_parameterized():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.CZ(a, b)**sympy.Symbol('u'),
        cirq.X(a)**sympy.Symbol('v'),
        cirq.Y(b)**sympy.Symbol('w'),
    )
    assert cirq.is_parameterized(circuit)

    circuit = cirq.resolve_parameters(circuit,
                                      cirq.ParamResolver({
                                          'u': 0.1,
                                          'v': 0.3
                                      }))
    assert cirq.is_parameterized(circuit)

    circuit = cirq.resolve_parameters(circuit, cirq.ParamResolver({'w': 0.2}))
    assert not cirq.is_parameterized(circuit)


def test_resolve_parameters():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.CZ(a, b)**sympy.Symbol('u'),
        cirq.X(a)**sympy.Symbol('v'),
        cirq.Y(b)**sympy.Symbol('w'),
    )
    resolved_circuit = cirq.resolve_parameters(
        circuit, cirq.ParamResolver({
            'u': 0.1,
            'v': 0.3,
            'w': 0.2
        }))
    cirq.testing.assert_has_diagram(
        resolved_circuit, """
0: ───@───────X^0.3───
      │
1: ───@^0.1───Y^0.2───
""")
    q = cirq.NamedQubit('q')
    # no-op parameter resolution
    circuit = cirq.FrozenCircuit([cirq.Moment(), cirq.Moment([cirq.X(q)])])
    resolved_circuit = cirq.resolve_parameters(circuit, cirq.ParamResolver({}))
    cirq.testing.assert_same_circuits(circuit, resolved_circuit)
    # actually resolve something
    circuit = cirq.FrozenCircuit(
        [cirq.Moment(),
         cirq.Moment([cirq.X(q)**sympy.Symbol('x')])])
    resolved_circuit = cirq.resolve_parameters(circuit,
                                               cirq.ParamResolver({'x': 0.2}))
    expected_circuit = cirq.FrozenCircuit(
        [cirq.Moment(), cirq.Moment([cirq.X(q)**0.2])])
    cirq.testing.assert_same_circuits(expected_circuit, resolved_circuit)


def test_parameter_names():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.CZ(a, b)**sympy.Symbol('u'),
        cirq.X(a)**sympy.Symbol('v'),
        cirq.Y(b)**sympy.Symbol('w'),
    )
    resolved_circuit = cirq.resolve_parameters(
        circuit, cirq.ParamResolver({
            'u': 0.1,
            'v': 0.3,
            'w': 0.2
        }))
    assert cirq.parameter_names(circuit) == {'u', 'v', 'w'}
    assert cirq.parameter_names(resolved_circuit) == set()


def test_next_moments_operating_on():
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = cirq.testing.random_circuit(randint(1, 20), n_moments,
                                              random())
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


def test_to_qasm():
    q0 = cirq.NamedQubit('q0')
    circuit = cirq.FrozenCircuit(cirq.X(q0),)
    assert circuit.to_qasm() == cirq.qasm(circuit)
    assert (circuit.to_qasm() == """// Generated from Cirq v{}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
""".format(cirq.__version__))


def test_save_qasm(tmpdir):
    file_path = os.path.join(tmpdir, 'test.qasm')
    q0 = cirq.NamedQubit('q0')
    circuit = cirq.FrozenCircuit(cirq.X(q0),)

    circuit.save_qasm(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
    assert (file_content == """// Generated from Cirq v{}

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
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CZ(a, b), cirq.H(b),
                                 cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d),
                                 cirq.H(d), cirq.CZ(c, d), cirq.H(c),
                                 cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b),
                                 cirq.H(a))

    # Empty frontiers means no results.
    actual = circuit.findall_operations_between(start_frontier={},
                                                end_frontier={})
    assert actual == []

    # Empty range is empty.
    actual = circuit.findall_operations_between(start_frontier={a: 5},
                                                end_frontier={a: 5})
    assert actual == []

    # Default end_frontier value is len(circuit.
    actual = circuit.findall_operations_between(start_frontier={a: 5},
                                                end_frontier={})
    assert actual == [
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Default start_frontier value is 0.
    actual = circuit.findall_operations_between(start_frontier={},
                                                end_frontier={a: 5})
    assert actual == [(0, cirq.H(a)), (1, cirq.CZ(a, b))]

    # omit_crossing_operations omits crossing operations.
    actual = circuit.findall_operations_between(start_frontier={a: 5},
                                                end_frontier={},
                                                omit_crossing_operations=True)
    assert actual == [
        (12, cirq.H(a)),
    ]

    # omit_crossing_operations keeps operations across included regions.
    actual = circuit.findall_operations_between(start_frontier={
        a: 5,
        b: 5
    },
                                                end_frontier={},
                                                omit_crossing_operations=True)
    assert actual == [
        (10, cirq.H(b)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together.
    actual = circuit.findall_operations_between(start_frontier={a: 5},
                                                end_frontier={b: 5})
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together (2).
    actual = circuit.findall_operations_between(start_frontier={a: 5},
                                                end_frontier={
                                                    a: 5,
                                                    b: 5
                                                })
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
    ]

    # Inclusive start, exclusive end.
    actual = circuit.findall_operations_between(start_frontier={c: 4},
                                                end_frontier={c: 8})
    assert actual == [
        (4, cirq.H(c)),
        (5, cirq.CZ(c, d)),
        (7, cirq.CZ(c, d)),
    ]

    # Out of range is clamped.
    actual = circuit.findall_operations_between(start_frontier={a: -100},
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
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CZ(a, b), cirq.H(b),
                                 cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d),
                                 cirq.H(d), cirq.CZ(c, d), cirq.H(c),
                                 cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b),
                                 cirq.H(a))

    # Empty cases.
    assert cirq.FrozenCircuit().reachable_frontier_from(start_frontier={}) == {}
    assert circuit.reachable_frontier_from(start_frontier={}) == {}

    # Clamped input cases.
    assert cirq.FrozenCircuit().reachable_frontier_from(
        start_frontier={a: 5}) == {
            a: 5
        }
    assert cirq.FrozenCircuit().reachable_frontier_from(
        start_frontier={a: -100}) == {
            a: 0
        }
    assert circuit.reachable_frontier_from(start_frontier={a: 100}) == {a: 100}

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
    assert circuit.reachable_frontier_from({
        a: 0,
        b: 0,
        c: 0
    }) == {
        a: 11,
        b: 9,
        c: 5
    }

    # Full circuit
    assert circuit.reachable_frontier_from({
        a: 0,
        b: 0,
        c: 0,
        d: 0
    }) == {
        a: 13,
        b: 13,
        c: 13,
        d: 13
    }

    # Blocker.
    assert circuit.reachable_frontier_from(
        {
            a: 0,
            b: 0,
            c: 0,
            d: 0
        }, is_blocker=lambda op: op == cirq.CZ(b, c)) == {
            a: 11,
            b: 3,
            c: 3,
            d: 5
        }


def test_submoments():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.FrozenCircuit(
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

    cirq.testing.assert_has_diagram(
        circuit, """
          ┌───────────┐   ┌──────┐
0: ───H────@───────────────@─────────
           │               │
1: ───@────┼@──────────────┼─────────
      │    ││              │
2: ───@────┼┼────@─────────┼────H────
           ││    │         │
3: ───H────@┼────┼─────────X^0.5─────
            │    │
4: ─────────X^0.5┼─────────H─────────
                 │
5: ──────────────X^0.5───────────────
          └───────────┘   └──────┘
""")

    cirq.testing.assert_has_diagram(circuit,
                                    """
  0 1 2 3     4     5
  │ │ │ │     │     │
  H @─@ H     │     │
  │ │ │ │     │     │
┌╴│ │ │ │     │     │    ╶┐
│ @─┼─┼─@     │     │     │
│ │ @─┼─┼─────X^0.5 │     │
│ │ │ @─┼─────┼─────X^0.5 │
└╴│ │ │ │     │     │    ╶┘
  │ │ │ │     │     │
┌╴│ │ │ │     │     │    ╶┐
│ @─┼─┼─X^0.5 H     │     │
│ │ │ H │     │     │     │
└╴│ │ │ │     │     │    ╶┘
  │ │ │ │     │     │
""",
                                    transpose=True)

    cirq.testing.assert_has_diagram(circuit,
                                    r"""
          /-----------\   /------\
0: ---H----@---------------@---------
           |               |
1: ---@----|@--------------|---------
      |    ||              |
2: ---@----||----@---------|----H----
           ||    |         |
3: ---H----@|----|---------X^0.5-----
            |    |
4: ---------X^0.5|---------H---------
                 |
5: --------------X^0.5---------------
          \-----------/   \------/
""",
                                    use_unicode_characters=False)

    cirq.testing.assert_has_diagram(circuit,
                                    r"""
  0 1 2 3     4     5
  | | | |     |     |
  H @-@ H     |     |
  | | | |     |     |
/ | | | |     |     |     \
| @-----@     |     |     |
| | @---------X^0.5 |     |
| | | @-------------X^0.5 |
\ | | | |     |     |     /
  | | | |     |     |
/ | | | |     |     |     \
| @-----X^0.5 H     |     |
| | | H |     |     |     |
\ | | | |     |     |     /
  | | | |     |     |
""",
                                    use_unicode_characters=False,
                                    transpose=True)


def test_decompose():
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose(
        cirq.FrozenCircuit(cirq.X(a), cirq.Y(b), cirq.CZ(
            a, b))) == [cirq.X(a), cirq.Y(b),
                        cirq.CZ(a, b)]


def test_inverse():
    a, b = cirq.LineQubit.range(2)
    forward = cirq.FrozenCircuit((cirq.X**0.5)(a), (cirq.Y**-0.2)(b),
                                 cirq.CZ(a, b))
    backward = cirq.FrozenCircuit((cirq.CZ**(-1.0))(a, b), (cirq.X**(-0.5))(a),
                                  (cirq.Y**(0.2))(b))
    cirq.testing.assert_same_circuits(cirq.inverse(forward), backward)

    cirq.testing.assert_same_circuits(cirq.inverse(cirq.FrozenCircuit()),
                                      cirq.FrozenCircuit())

    no_inverse = cirq.FrozenCircuit(cirq.measure(a, b))
    with pytest.raises(TypeError, match='__pow__'):
        cirq.inverse(no_inverse)

    # Default when there is no inverse for an op.
    default = cirq.FrozenCircuit((cirq.X**0.5)(a), (cirq.Y**-0.2)(b))
    cirq.testing.assert_same_circuits(cirq.inverse(no_inverse, default),
                                      default)
    assert cirq.inverse(no_inverse, None) is None


def test_pow_valid_only_for_minus_1():
    a, b = cirq.LineQubit.range(2)
    forward = cirq.FrozenCircuit((cirq.X**0.5)(a), (cirq.Y**-0.2)(b),
                                 cirq.CZ(a, b))

    backward = cirq.FrozenCircuit((cirq.CZ**(-1.0))(a, b), (cirq.X**(-0.5))(a),
                                  (cirq.Y**(0.2))(b))

    cirq.testing.assert_same_circuits(cirq.pow(forward, -1), backward)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 1)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 0)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, -2.5)


def test_device_propagates():
    c = cirq.FrozenCircuit(device=cg.Foxtail)
    assert c[:].device is cg.Foxtail


def test_moment_groups():
    qubits = [cirq.GridQubit(x, y) for x in range(8) for y in range(8)]
    c0 = cirq.H(qubits[0])
    c7 = cirq.H(qubits[7])
    cz14 = cirq.CZ(qubits[1], qubits[4])
    cz25 = cirq.CZ(qubits[2], qubits[5])
    cz36 = cirq.CZ(qubits[3], qubits[6])
    moment1 = cirq.Moment([c0, cz14, cz25, c7])
    moment2 = cirq.Moment([c0, cz14, cz25, cz36, c7])
    moment3 = cirq.Moment([cz14, cz25, cz36])
    moment4 = cirq.Moment([cz25, cz36])
    circuit = cirq.FrozenCircuit((moment1, moment2, moment3, moment4))
    cirq.testing.assert_has_diagram(circuit,
                                    r"""
           ┌──┐   ┌───┐   ┌───┐   ┌──┐
(0, 0): ────H──────H─────────────────────

(0, 1): ────@──────@───────@─────────────
            │      │       │
(0, 2): ────┼@─────┼@──────┼@──────@─────
            ││     ││      ││      │
(0, 3): ────┼┼─────┼┼@─────┼┼@─────┼@────
            ││     │││     │││     ││
(0, 4): ────@┼─────@┼┼─────@┼┼─────┼┼────
             │      ││      ││     ││
(0, 5): ─────@──────@┼──────@┼─────@┼────
                     │       │      │
(0, 6): ─────────────@───────@──────@────

(0, 7): ────H──────H─────────────────────
           └──┘   └───┘   └───┘   └──┘
""",
                                    use_unicode_characters=True)


def test_moments_property():
    q = cirq.NamedQubit('q')
    c = cirq.FrozenCircuit(cirq.X(q), cirq.Y(q))
    assert c.moments[0] == cirq.Moment([cirq.X(q)])
    assert c.moments[1] == cirq.Moment([cirq.Y(q)])


def test_operation_shape_validation():

    class BadOperation1(cirq.Operation):

        def _qid_shape_(self):
            return (1,)

        @property
        def qubits(self):
            return cirq.LineQid.for_qid_shape((1, 2, 3))

        def with_qubits(self, *qubits):
            raise NotImplementedError

    class BadOperation2(cirq.Operation):

        def _qid_shape_(self):
            return (1, 2, 3, 9)

        @property
        def qubits(self):
            return cirq.LineQid.for_qid_shape((1, 2, 3))

        def with_qubits(self, *qubits):
            raise NotImplementedError

    _ = cirq.FrozenCircuit(cirq.X(cirq.LineQid(0, 2)))  # Valid
    with pytest.raises(ValueError, match='Invalid operation'):
        _ = cirq.FrozenCircuit(BadOperation1())
    with pytest.raises(ValueError, match='Invalid operation'):
        _ = cirq.FrozenCircuit(BadOperation2())


def test_json_dict():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.FrozenCircuit(cirq.CNOT(q0, q1))
    # TODO: JSON dict is missing moments + device?
    assert c._json_dict_() == {
        'cirq_type': 'FrozenCircuit',
        'moments': (cirq.Moment([cirq.CNOT(q0, q1)]),),
        'device': cirq.UNCONSTRAINED_DEVICE,
    }


def test_init_contents():
    a, b = cirq.LineQubit.range(2)

    # Moments are not subject to insertion rules.
    c = cirq.FrozenCircuit(
        cirq.Moment([cirq.H(a)]),
        cirq.Moment([cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )
    assert len(c.moments) == 3

    # Earliest packing by default.
    c = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.X(b),
        cirq.CNOT(a, b),
    )
    assert c == cirq.FrozenCircuit(
        cirq.Moment([cirq.H(a), cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )

    # Packing can be controlled.
    c = cirq.FrozenCircuit(cirq.H(a),
                           cirq.X(b),
                           cirq.CNOT(a, b),
                           strategy=cirq.InsertStrategy.NEW)
    assert c == cirq.FrozenCircuit(
        cirq.Moment([cirq.H(a)]),
        cirq.Moment([cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )

    cirq.FrozenCircuit()


def test_indexing_by_pair():
    # 0: ───H───@───X───@───
    #           │       │
    # 1: ───────H───@───@───
    #               │   │
    # 2: ───────────H───X───
    q = cirq.LineQubit.range(3)
    c = cirq.FrozenCircuit([
        cirq.H(q[0]),
        cirq.H(q[1]).controlled_by(q[0]),
        cirq.H(q[2]).controlled_by(q[1]),
        cirq.X(q[0]),
        cirq.CCNOT(*q),
    ])

    # Indexing by single moment and qubit.
    assert c[0, q[0]] == c[0][q[0]] == cirq.H(q[0])
    assert c[1, q[0]] == c[1, q[1]] == cirq.H(q[1]).controlled_by(q[0])
    assert c[2, q[0]] == c[2][q[0]] == cirq.X(q[0])
    assert c[2, q[1]] == c[2, q[2]] == cirq.H(q[2]).controlled_by(q[1])
    assert c[3, q[0]] == c[3, q[1]] == c[3, q[2]] == cirq.CCNOT(*q)

    # Indexing by moment and qubit - throws if there is no operation.
    with pytest.raises(KeyError, match="Moment doesn't act on given qubit"):
        _ = c[0, q[1]]

    # Indexing by single moment and multiple qubits.
    assert c[0, q] == c[0]
    assert c[1, q] == c[1]
    assert c[2, q] == c[2]
    assert c[3, q] == c[3]
    assert c[0, q[0:2]] == c[0]
    assert c[0, q[1:3]] == cirq.Moment([])
    assert c[1, q[1:2]] == c[1]
    assert c[2, [q[0]]] == cirq.Moment([cirq.X(q[0])])
    assert c[2, q[1:3]] == cirq.Moment([cirq.H(q[2]).controlled_by(q[1])])
    assert c[np.int64(2), q[0:2]] == c[2]

    # Indexing by single qubit.
    assert c[:, q[0]] == cirq.FrozenCircuit([
        cirq.Moment([cirq.H(q[0])]),
        cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
        cirq.Moment([cirq.X(q[0])]),
        cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
    ])
    assert c[:, q[1]] == cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
        cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
        cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
    ])
    assert c[:, q[2]] == cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([]),
        cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
        cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
    ])

    # Indexing by several qubits.
    assert c[:, q] == c[:, q[0:2]] == c[:, [q[0], q[2]]] == c
    assert c[:, q[1:3]] == cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
        cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
        cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
    ])

    # Indexing by several moments and one qubit.
    assert c[1:3, q[0]] == cirq.FrozenCircuit([
        cirq.H(q[1]).controlled_by(q[0]),
        cirq.X(q[0]),
    ])
    assert c[1::2, q[2]] == cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([cirq.CCNOT(*q)]),
    ])

    # Indexing by several moments and several qubits.
    assert c[0:2, q[1:3]] == cirq.FrozenCircuit([
        cirq.Moment([]),
        cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
    ])
    assert c[::2, q[0:2]] == cirq.FrozenCircuit([
        cirq.Moment([cirq.H(q[0])]),
        cirq.Moment([cirq.H(q[2]).controlled_by(q[1]),
                     cirq.X(q[0])]),
    ])

    # Equivalent ways of indexing.
    assert c[0:2, q[1:3]] == c[0:2][:, q[1:3]] == c[:, q[1:3]][0:2]

    # Passing more than 2 items is forbidden.
    with pytest.raises(ValueError, match='If key is tuple, it must be a pair.'):
        _ = c[0, q[1], 0]

    # Can't swap indices.
    with pytest.raises(TypeError,
                       match='tuple indices must be integers or slices'):
        _ = c[q[1], 0]


def test_indexing_by_numpy_integer():
    q = cirq.NamedQubit('q')
    c = cirq.FrozenCircuit(cirq.X(q), cirq.Y(q))

    assert c[np.int32(1)] == cirq.Moment([cirq.Y(q)])
    assert c[np.int64(1)] == cirq.Moment([cirq.Y(q)])


def test_all_measurement_keys():

    class Unknown(cirq.SingleQubitGate):

        def _measurement_key_(self):
            return 'test'

    a, b = cirq.LineQubit.range(2)
    c = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.CNOT(a, b),
        cirq.measure(a, key='x'),
        cirq.measure(b, key='y'),
        cirq.reset(a),
        cirq.measure(a, b, key='xy'),
        Unknown().on(a),
    )

    # Big case.
    assert c.all_measurement_keys() == ('x', 'y', 'xy', 'test')

    # Empty case.
    assert cirq.FrozenCircuit().all_measurement_keys() == ()

    # Output order matches insertion order, not qubit order.
    assert cirq.FrozenCircuit(
        cirq.Moment([
            cirq.measure(a, key='x'),
            cirq.measure(b, key='y'),
        ])).all_measurement_keys() == ('x', 'y')
    assert cirq.FrozenCircuit(
        cirq.Moment([
            cirq.measure(b, key='y'),
            cirq.measure(a, key='x'),
        ])).all_measurement_keys() == ('y', 'x')


def test_deprecated():
    q = cirq.NamedQubit('q')
    circuit = cirq.FrozenCircuit([cirq.H(q)])
    with cirq.testing.assert_logs('final_state_vector', 'deprecated'):
        _ = circuit.final_wavefunction()


def test_repr_html_escaping():

    class TestGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(
                wire_symbols=["< ' F ' >", "< ' F ' >"])

    F2 = TestGate()
    a = cirq.LineQubit(1)
    c = cirq.NamedQubit("|c>")

    circuit = cirq.FrozenCircuit([F2(a, c)])

    # Escaping Special Characters in Gate names.
    assert '&lt; &#x27; F &#x27; &gt;' in circuit._repr_html_()

    # Escaping Special Characters in Qubit names.
    assert '|c&gt;' in circuit._repr_html_()
