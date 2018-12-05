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
from cirq import Moment


def test_validation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    _ = Moment([])
    _ = Moment([cirq.X(a)])
    _ = Moment([cirq.CZ(a, b)])
    _ = Moment([cirq.CZ(b, d)])
    _ = Moment([cirq.CZ(a, b), cirq.CZ(c, d)])
    _ = Moment([cirq.CZ(a, c), cirq.CZ(b, d)])
    _ = Moment([cirq.CZ(a, c), cirq.X(b)])

    with pytest.raises(ValueError):
        _ = Moment([cirq.X(a), cirq.X(a)])
    with pytest.raises(ValueError):
        _ = Moment([cirq.CZ(a, c), cirq.X(c)])
    with pytest.raises(ValueError):
        _ = Moment([cirq.CZ(a, c), cirq.CZ(c, d)])


def test_equality():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get frozen into tuples.
    eq.add_equality_group(Moment(),
                          Moment([]), Moment(()))
    eq.add_equality_group(
        Moment([cirq.X(d)]), Moment((cirq.X(d),)))

    # Equality depends on gate and qubits.
    eq.add_equality_group(Moment([cirq.X(a)]))
    eq.add_equality_group(Moment([cirq.X(b)]))
    eq.add_equality_group(Moment([cirq.Y(a)]))

    # Equality depends on order.
    eq.add_equality_group(Moment([cirq.X(a), cirq.X(b)]))
    eq.add_equality_group(Moment([cirq.X(b), cirq.X(a)]))

    # Two qubit gates.
    eq.make_equality_group(lambda: Moment([cirq.CZ(c, d)]))
    eq.make_equality_group(lambda: Moment([cirq.CZ(a, c)]))
    eq.make_equality_group(lambda: Moment([cirq.CZ(a, b), cirq.CZ(c, d)]))
    eq.make_equality_group(lambda: Moment([cirq.CZ(a, c), cirq.CZ(b, d)]))


def test_operates_on():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Empty case.
    assert not Moment().operates_on([])
    assert not Moment().operates_on([a])
    assert not Moment().operates_on([b])
    assert not Moment().operates_on([a, b])

    # One-qubit operation case.
    assert not Moment([cirq.X(a)]).operates_on([])
    assert Moment([cirq.X(a)]).operates_on([a])
    assert not Moment([cirq.X(a)]).operates_on([b])
    assert Moment([cirq.X(a)]).operates_on([a, b])

    # Two-qubit operation case.
    assert not Moment([cirq.CZ(a, b)]).operates_on([])
    assert Moment([cirq.CZ(a, b)]).operates_on([a])
    assert Moment([cirq.CZ(a, b)]).operates_on([b])
    assert Moment([cirq.CZ(a, b)]).operates_on([a, b])
    assert not Moment([cirq.CZ(a, b)]).operates_on([c])
    assert Moment([cirq.CZ(a, b)]).operates_on([a, c])
    assert Moment([cirq.CZ(a, b)]).operates_on([a, b, c])

    # Multiple operations case.
    assert not Moment([cirq.X(a), cirq.X(b)]).operates_on([])
    assert Moment([cirq.X(a), cirq.X(b)]).operates_on([a])
    assert Moment([cirq.X(a), cirq.X(b)]).operates_on([b])
    assert Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b])
    assert not Moment([cirq.X(a), cirq.X(b)]).operates_on([c])
    assert Moment([cirq.X(a), cirq.X(b)]).operates_on([a, c])
    assert Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b, c])


def test_with_operation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert Moment().with_operation(cirq.X(a)) == Moment([cirq.X(a)])

    assert (Moment([cirq.X(a)]).with_operation(cirq.X(b)) ==
            Moment([cirq.X(a), cirq.X(b)]))

    with pytest.raises(ValueError):
        _ = Moment([cirq.X(a)]).with_operation(cirq.X(a))


def test_without_operations_touching():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Empty case.
    assert Moment().without_operations_touching([]) == Moment()
    assert Moment().without_operations_touching([a]) == Moment()
    assert Moment().without_operations_touching([a, b]) == Moment()

    # One-qubit operation case.
    assert (Moment([cirq.X(a)]).without_operations_touching([]) ==
            Moment([cirq.X(a)]))
    assert (Moment([cirq.X(a)]).without_operations_touching([a]) ==
            Moment())
    assert (Moment([cirq.X(a)]).without_operations_touching([b]) ==
            Moment([cirq.X(a)]))

    # Two-qubit operation case.
    assert (Moment([cirq.CZ(a, b)]).without_operations_touching([]) ==
            Moment([cirq.CZ(a, b)]))
    assert (Moment([cirq.CZ(a, b)]).without_operations_touching([a]) ==
            Moment())
    assert (Moment([cirq.CZ(a, b)]).without_operations_touching([b]) ==
            Moment())
    assert (Moment([cirq.CZ(a, b)]).without_operations_touching([c]) ==
            Moment([cirq.CZ(a, b)]))

    # Multiple operation case.
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([]) ==
            Moment([cirq.CZ(a, b), cirq.X(c)]))
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([a]) ==
            Moment([cirq.X(c)]))
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([b]) ==
            Moment([cirq.X(c)]))
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([c]) ==
            Moment([cirq.CZ(a, b)]))
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([a, b]) ==
            Moment([cirq.X(c)]))
    assert (Moment([cirq.CZ(a, b),
                    cirq.X(c)]).without_operations_touching([a, c]) ==
            Moment())


def test_copy():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = Moment([cirq.CZ(a, b)])
    copy = original.__copy__()
    assert original == copy
    assert id(original) != id(copy)


def test_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert Moment([cirq.X(a), cirq.X(b)]).qubits == {a , b}
    assert Moment([cirq.X(a)]).qubits == {a}
    assert Moment([cirq.CZ(a, b)]).qubits == {a, b}
