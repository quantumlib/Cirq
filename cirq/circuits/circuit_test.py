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
from cirq import ops, Symbol
from cirq.circuits.circuit import Circuit, _operation_to_unitary_matrix
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.google import ExpWGate
from cirq.testing import EqualsTester
from cirq.extension import Extensions


def test_equality():
    a = ops.QubitId()
    b = ops.QubitId()

    eq = EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(Circuit(),
                          Circuit([]), Circuit(()))
    eq.add_equality_group(
        Circuit([Moment()]),
        Circuit((Moment(),)))

    # Equality depends on structure and contents.
    eq.add_equality_group(Circuit([Moment([ops.X(a)])]))
    eq.add_equality_group(Circuit([Moment([ops.X(b)])]))
    eq.add_equality_group(
        Circuit(
            [Moment([ops.X(a)]),
             Moment([ops.X(b)])]))
    eq.add_equality_group(
        Circuit([Moment([ops.X(a), ops.X(b)])]))

    # Big case.
    eq.add_equality_group(
        Circuit([
            Moment([ops.H(a), ops.H(b)]),
            Moment([ops.CZ(a, b)]),
            Moment([ops.H(b)]),
        ]))
    eq.add_equality_group(
        Circuit([
            Moment([ops.H(a)]),
            Moment([ops.CNOT(a, b)]),
        ]))


def test_append_single():
    a = ops.QubitId()

    c = Circuit()
    c.append(())
    assert c == Circuit()

    c = Circuit()
    c.append(ops.X(a))
    assert c == Circuit([Moment([ops.X(a)])])

    c = Circuit()
    c.append([ops.X(a)])
    assert c == Circuit([Moment([ops.X(a)])])


def test_append_multiple():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    c.append([ops.X(a), ops.X(b)], InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(b)])
    ])

    c = Circuit()
    c.append([ops.X(a), ops.X(b)], InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.X(b)]),
    ])

    c = Circuit()
    c.append(ops.X(a), InsertStrategy.EARLIEST)
    c.append(ops.X(b), InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.X(b)]),
    ])


def test_slice():
    a = ops.QubitId()
    b = ops.QubitId()
    c = Circuit([
        Moment([ops.H(a), ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.H(b)]),
    ])
    assert c[0:1] == Circuit([Moment([ops.H(a), ops.H(b)])])
    assert c[::2] == Circuit([
        Moment([ops.H(a), ops.H(b)]), Moment([ops.H(b)])
    ])
    assert c[0:1:2] == Circuit([Moment([ops.H(a), ops.H(b)])])
    assert c[1:3:] == Circuit([Moment([ops.CZ(a, b)]), Moment([ops.H(b)])])
    assert c[::-1] == Circuit([Moment([ops.H(b)]), Moment([ops.CZ(a, b)]),
                               Moment([ops.H(a), ops.H(b)])])
    assert c[3:0:-1] == Circuit([Moment([ops.H(b)]), Moment([ops.CZ(a, b)])])
    assert c[0:2:-1] == Circuit()


def test_concatenate():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    d = Circuit([Moment([ops.X(b)])])
    e = Circuit([Moment([ops.X(a), ops.X(b)])])

    assert c + d == Circuit([Moment([ops.X(b)])])
    assert d + c == Circuit([Moment([ops.X(b)])])
    assert e + d == Circuit([
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(b)])
    ])

    d += c
    assert d == Circuit([Moment([ops.X(b)])])

    c += d
    assert c == Circuit([Moment([ops.X(b)])])

    f = e + d
    f += e
    assert f == Circuit([
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(a), ops.X(b)])
    ])

    with pytest.raises(TypeError):
        _ = c + 'a'
    with pytest.raises(TypeError):
        c += 'a'


def test_multiply():
    a = ops.QubitId()

    c = Circuit()
    d = Circuit([Moment([ops.X(a)])])

    assert c * 0 == Circuit()
    assert d * 0 == Circuit()
    assert d * 2 == Circuit([Moment([ops.X(a)]),
                             Moment([ops.X(a)])])
    assert 1 * c == Circuit()
    assert -1 * d == Circuit()
    assert 1 * d == Circuit([Moment([ops.X(a)])])

    d *= 3
    assert d == Circuit([Moment([ops.X(a)]),
                         Moment([ops.X(a)]),
                         Moment([ops.X(a)])])

    with pytest.raises(TypeError):
        _ = c * 'a'
    with pytest.raises(TypeError):
        _ = 'a' * c
    with pytest.raises(TypeError):
        c *= 'a'


def test_container_methods():
    a = ops.QubitId()
    b = ops.QubitId()
    c = Circuit([
        Moment([ops.H(a), ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.H(b)]),
    ])
    assert list(c) == list(c.moments)
    # __iter__
    assert list(iter(c)) == list(c.moments)
    # __reversed__ for free.
    assert list(reversed(c)) == list(reversed(c.moments))
    # __contains__ for free.
    assert Moment([ops.H(b)]) in c

    assert len(c) == 3


def test_bad_index():
    a = ops.QubitId()
    b = ops.QubitId()
    c = Circuit([Moment([ops.H(a), ops.H(b)])])
    with pytest.raises(TypeError):
        _ = c['string']

def test_append_strategies():
    a = ops.QubitId()
    b = ops.QubitId()
    stream = [ops.X(a), ops.CZ(a, b), ops.X(b), ops.X(b), ops.X(a)]

    c = Circuit()
    c.append(stream, InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(b), ops.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b), ops.X(a)]),
        Moment([ops.X(b)]),
    ])


def test_insert():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()

    c.insert(0, ())
    assert c == Circuit()

    with pytest.raises(IndexError):
        c.insert(-1, ())
    with pytest.raises(IndexError):
        c.insert(1, ())

    c.insert(0, [ops.X(a), ops.CZ(a, b), ops.X(b)])
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])

    with pytest.raises(IndexError):
        c.insert(550, ())

    c.insert(1, ops.H(b), strategy=InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])

    c.insert(0, ops.H(b), strategy=InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.H(b)]),
        Moment([ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])


def test_insert_inline_near_start():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
    ])

    c.insert(1, ops.X(a), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment(),
    ])

    c.insert(1, ops.Y(a), strategy=InsertStrategy.INLINE)
    assert c ==Circuit([
        Moment([ops.X(a)]),
        Moment([ops.Y(a)]),
        Moment(),
    ])

    c.insert(0, ops.Z(b), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.Z(b)]),
        Moment([ops.X(a)]),
        Moment([ops.Y(a)]),
        Moment(),
    ])


def test_operation_at():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, -1) is None
    assert c.operation_at(a, 102) is None

    c = Circuit([Moment()])
    assert c.operation_at(a, 0) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.operation_at(b, 0) is None
    assert c.operation_at(a, 1) is None
    assert c.operation_at(a, 0) == ops.X(a)

    c = Circuit([Moment(), Moment([ops.CZ(a, b)])])
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, 1) == ops.CZ(a, b)


def test_next_moment_operating_on():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.next_moment_operating_on([a]) is None
    assert c.next_moment_operating_on([a], 0) is None
    assert c.next_moment_operating_on([a], 102) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.next_moment_operating_on([a]) == 0
    assert c.next_moment_operating_on([a], 0) == 0
    assert c.next_moment_operating_on([a, b]) == 0
    assert c.next_moment_operating_on([a], 1) is None
    assert c.next_moment_operating_on([b]) is None

    c = Circuit([
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
        Moment([ops.CZ(a, b)])
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
    a = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment([ops.X(a)]),
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
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.prev_moment_operating_on([a]) is None
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([a], 102) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.prev_moment_operating_on([a]) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a, b]) == 0
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([b]) is None

    c = Circuit([
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(a)]),
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
    a = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment([ops.X(a)]),
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


def test_clear_operations_touching():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    c.clear_operations_touching([a, b], range(10))
    assert c == Circuit()

    c = Circuit([
        Moment(),
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment([ops.X(b)]),
        Moment([ops.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])

    c = Circuit([
        Moment(),
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a, b], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
    ])


def test_qubits():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(b)]),
    ])
    assert c.qubits() == {a, b}

    c = Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
    ])
    assert c.qubits() == {a}

    c = Circuit([
        Moment([ops.CZ(a, b)]),
    ])
    assert c.qubits() == {a, b}

    c = Circuit([
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(a)])
    ])
    assert c.qubits() == {a, b}


def test_from_ops():
    a = ops.QubitId()
    b = ops.QubitId()

    actual = Circuit.from_ops(
        ops.X(a),
        [ops.Y(a), ops.Z(b)],
        ops.CZ(a, b),
        ops.X(a),
        [ops.Z(b), ops.Y(a)],
    )

    assert actual == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.Y(a), ops.Z(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(a), ops.Z(b)]),
        Moment([ops.Y(a)]),
    ])


def test_to_text_diagram_teleportation_to_diagram():
    ali = ops.NamedQubit('(0, 0)')
    bob = ops.NamedQubit('(0, 1)')
    msg = ops.NamedQubit('(1, 0)')
    tmp = ops.NamedQubit('(1, 1)')

    c = Circuit([
        Moment([ops.H(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([ops.X(msg)**0.5]),
        Moment([ops.CNOT(msg, ali)]),
        Moment([ops.H(msg)]),
        Moment(
            [ops.MeasurementGate()(msg),
             ops.MeasurementGate()(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([ops.CNOT(msg, tmp)]),
        Moment([ops.CZ(bob, tmp)]),
    ])

    assert c.to_text_diagram().strip() == """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────@───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───Z───
    """.strip()
    assert c.to_text_diagram(use_unicode_characters=False).strip() == """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------@---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---Z---
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
|      @-------------Z
|      |      |      |
        """.strip()


def test_to_text_diagram_extended_gate():
    q = ops.NamedQubit('(0, 0)')
    q2 = ops.NamedQubit('(0, 1)')
    q3 = ops.NamedQubit('(0, 2)')

    class FGate(ops.Gate):
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
    class FGateAsText(ops.TextDiagrammableGate):
        def __init__(self, f_gate):
            self.f_gate = f_gate

        def text_diagram_wire_symbols(self,
                                      qubit_count=None,
                                      use_unicode_characters=True,
                                      precision=3):
            return 'F'

    diagram = c.to_text_diagram(Extensions({
        ops.TextDiagrammableGate: {
           FGate: FGateAsText
        }
    }), use_unicode_characters=False)

    assert diagram.strip() == """
(0, 0): ---F---
        """.strip()


def test_to_text_diagram_multi_qubit_gate():
    q1 = ops.NamedQubit('(0, 0)')
    q2 = ops.NamedQubit('(0, 1)')
    q3 = ops.NamedQubit('(0, 2)')
    c = Circuit([Moment([ops.MeasurementGate('msg').on(q1, q2, q3)])])
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
    class BadGate(ops.TextDiagrammableGate):
        def text_diagram_wire_symbols(self,
                                      qubit_count=None,
                                      use_unicode_characters=True,
                                      precision=3):
            return 'a', 'a'
    q1 = ops.NamedQubit('(0, 0)')
    q2 = ops.NamedQubit('(0, 1)')
    q3 = ops.NamedQubit('(0, 2)')
    c = Circuit([Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()


def test_to_text_diagram_parameterized_value():
    q = ops.NamedQubit('cube')

    class PGate(ops.TextDiagrammableGate):
        def __init__(self, val):
            self.val = val

        def text_diagram_wire_symbols(self,
                                      qubit_count=None,
                                      use_unicode_characters=True,
                                      precision=3):
            return 'P',

        def text_diagram_exponent(self):
            return self.val

    c = Circuit.from_ops(
        PGate(1).on(q),
        PGate(2).on(q),
        PGate(Symbol('a')).on(q),
        PGate(Symbol('%$&#*(')).on(q),
    )
    assert str(c).strip() == 'cube: ───P───P^2───P^a───P^Symbol("%$&#*(")───'


def test_to_text_diagram_custom_order():
    qa = ops.NamedQubit('2')
    qb = ops.NamedQubit('3')
    qc = ops.NamedQubit('4')

    c = Circuit([Moment([ops.X(qa), ops.X(qb), ops.X(qc)])])
    diagram = c.to_text_diagram(
        qubit_order=ops.QubitOrder.sorted_by(lambda e: int(str(e)) % 3),
        use_unicode_characters=False)
    assert diagram.strip() == """
3: ---X---

4: ---X---

2: ---X---
    """.strip()


def test_overly_precise_diagram():
    # Test default precision of 3
    qa = ops.NamedQubit('a')
    c = Circuit([Moment([ops.X(qa)**0.12345678])])
    diagram = c.to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
a: ---X^0.123---
    """.strip()


def test_none_precision_diagram():
    # Test default precision of 3
    qa = ops.NamedQubit('a')
    c = Circuit([Moment([ops.X(qa)**0.4921875])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=None)
    assert diagram.strip() == """
a: ---X^0.4921875---
    """.strip()


def test_diagram_custom_precision():
    qa = ops.NamedQubit('a')
    c = Circuit([Moment([ops.X(qa)**0.12341234])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=5)
    assert diagram.strip() == """
a: ---X^0.12341---
    """.strip()


def test_diagram_wgate():
    qa = ops.NamedQubit('a')
    test_wgate = ExpWGate(half_turns=0.12341234, axis_half_turns=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=2)
    assert diagram.strip() == """
a: ---W(0.43)^0.12---
    """.strip()


def test_diagram_wgate_none_precision():
    qa = ops.NamedQubit('a')
    test_wgate = ExpWGate(half_turns=0.12341234, axis_half_turns=0.43214321)
    c = Circuit([Moment([test_wgate.on(qa)])])
    diagram = c.to_text_diagram(use_unicode_characters=False, precision=None)
    assert diagram.strip() == """
a: ---W(0.43214321)^0.12341234---
    """.strip()


def test_operation_to_unitary_matrix():
    ex = Extensions()
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')

    m = _operation_to_unitary_matrix(ops.X(a),
                                     {a: 0},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([[0, 1],
                  [1, 0]]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(ops.X(a),
                                     {a: 0, b: 1},
                                     ex)
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.kron(ops.X.matrix(), np.eye(2)))
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]),
        atol=1e-8)

    m = _operation_to_unitary_matrix(ops.X(a),
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

    m = _operation_to_unitary_matrix(ops.CNOT(b, a),
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

    m = _operation_to_unitary_matrix(ops.CNOT(b, a),
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
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')
    c = Circuit.from_ops(ops.Z(a), ops.X(b))
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
    c = Circuit.from_ops(ops.Z(a), ops.X(b), ops.CNOT(a, b))
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
    c = Circuit.from_ops(ops.MeasurementGate()(a))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix(ignore_terminal_measurements=False)

    # Ignoring terminal measurements.
    c = Circuit.from_ops(ops.MeasurementGate()(a))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(),
        np.eye(2))

    # Non-terminal measurements are not ignored.
    c = Circuit.from_ops(ops.MeasurementGate()(a), ops.X(a))
    with pytest.raises(TypeError):
        _ = c.to_unitary_matrix()

    # Ignore measurements by turning them into identities.
    class IdentityGate(ops.KnownMatrixGate):
        def matrix(self):
            return np.eye(2)
    ex = Extensions()
    ex.add_cast(desired_type=ops.KnownMatrixGate,
                actual_type=ops.MeasurementGate,
                conversion=lambda _: IdentityGate())
    c = Circuit.from_ops(ops.MeasurementGate()(a))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.to_unitary_matrix(ext=ex, ignore_terminal_measurements=False),
        np.eye(2))


def test_simple_circuits_to_unitary_matrix():
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')

    # Phase parity.
    c = Circuit.from_ops(ops.CNOT(a, b), ops.Z(b), ops.CNOT(a, b))
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
    for expected in [np.diag([1, 1j, -1, -1j]), ops.CNOT.matrix()]:

        class Passthrough(ops.KnownMatrixGate):
            def matrix(self):
                return expected

        c = Circuit.from_ops(Passthrough()(a, b))
        m = c.to_unitary_matrix()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected)
