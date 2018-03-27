# Copyright 2018 Google LLC
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
from cirq.circuits.moment import Moment
from cirq.testing import EqualsTester


def test_validation():
    a = ops.QubitId()
    b = ops.QubitId()
    c = ops.QubitId()
    d = ops.QubitId()

    _ = Moment([])
    _ = Moment([ops.X(a)])
    _ = Moment([ops.CZ(a, b)])
    _ = Moment([ops.CZ(b, d)])
    _ = Moment([ops.CZ(a, b), ops.CZ(c, d)])
    _ = Moment([ops.CZ(a, c), ops.CZ(b, d)])
    _ = Moment([ops.CZ(a, c), ops.X(b)])

    with pytest.raises(ValueError):
        _ = Moment([ops.X(a), ops.X(a)])
    with pytest.raises(ValueError):
        _ = Moment([ops.CZ(a, c), ops.X(c)])
    with pytest.raises(ValueError):
        _ = Moment([ops.CZ(a, c), ops.CZ(c, d)])


def test_equality():
    a = ops.QubitId()
    b = ops.QubitId()
    c = ops.QubitId()
    d = ops.QubitId()

    eq = EqualsTester()

    # Default is empty. Iterables get frozen into tuples.
    eq.add_equality_group(Moment(),
                          Moment([]), Moment(()))
    eq.add_equality_group(
        Moment([ops.X(d)]), Moment((ops.X(d),)))

    # Equality depends on gate and qubits.
    eq.add_equality_group(Moment([ops.X(a)]))
    eq.add_equality_group(Moment([ops.X(b)]))
    eq.add_equality_group(Moment([ops.Y(a)]))

    # Equality depends on order.
    eq.add_equality_group(Moment([ops.X(a), ops.X(b)]))
    eq.add_equality_group(Moment([ops.X(b), ops.X(a)]))

    # Two qubit gates.
    eq.make_equality_pair(lambda: Moment([ops.CZ(c, d)]))
    eq.make_equality_pair(lambda: Moment([ops.CZ(a, c)]))
    eq.make_equality_pair(lambda: Moment([ops.CZ(a, b), ops.CZ(c, d)]))
    eq.make_equality_pair(lambda: Moment([ops.CZ(a, c), ops.CZ(b, d)]))


def test_operates_on():
    a = ops.QubitId()
    b = ops.QubitId()
    c = ops.QubitId()

    # Empty case.
    assert not Moment().operates_on([])
    assert not Moment().operates_on([a])
    assert not Moment().operates_on([b])
    assert not Moment().operates_on([a, b])

    # One-qubit operation case.
    assert not Moment([ops.X(a)]).operates_on([])
    assert Moment([ops.X(a)]).operates_on([a])
    assert not Moment([ops.X(a)]).operates_on([b])
    assert Moment([ops.X(a)]).operates_on([a, b])

    # Two-qubit operation case.
    assert not Moment([ops.CZ(a, b)]).operates_on([])
    assert Moment([ops.CZ(a, b)]).operates_on([a])
    assert Moment([ops.CZ(a, b)]).operates_on([b])
    assert Moment([ops.CZ(a, b)]).operates_on([a, b])
    assert not Moment([ops.CZ(a, b)]).operates_on([c])
    assert Moment([ops.CZ(a, b)]).operates_on([a, c])
    assert Moment([ops.CZ(a, b)]).operates_on([a, b, c])

    # Multiple operations case.
    assert not Moment([ops.X(a), ops.X(b)]).operates_on([])
    assert Moment([ops.X(a), ops.X(b)]).operates_on([a])
    assert Moment([ops.X(a), ops.X(b)]).operates_on([b])
    assert Moment([ops.X(a), ops.X(b)]).operates_on([a, b])
    assert not Moment([ops.X(a), ops.X(b)]).operates_on([c])
    assert Moment([ops.X(a), ops.X(b)]).operates_on([a, c])
    assert Moment([ops.X(a), ops.X(b)]).operates_on([a, b, c])


def test_with_operation():
    a = ops.QubitId()
    b = ops.QubitId()

    assert Moment().with_operation(ops.X(a)) == Moment([ops.X(a)])

    assert (Moment([ops.X(a)]).with_operation(ops.X(b)) ==
            Moment([ops.X(a), ops.X(b)]))

    with pytest.raises(ValueError):
        _ = Moment([ops.X(a)]).with_operation(ops.X(a))


def test_without_operations_touching():
    a = ops.QubitId()
    b = ops.QubitId()
    c = ops.QubitId()

    # Empty case.
    assert Moment().without_operations_touching([]) == Moment()
    assert Moment().without_operations_touching([a]) == Moment()
    assert Moment().without_operations_touching([a, b]) == Moment()

    # One-qubit operation case.
    assert (Moment([ops.X(a)]).without_operations_touching([]) ==
            Moment([ops.X(a)]))
    assert (Moment([ops.X(a)]).without_operations_touching([a]) ==
            Moment())
    assert (Moment([ops.X(a)]).without_operations_touching([b]) ==
            Moment([ops.X(a)]))

    # Two-qubit operation case.
    assert (Moment([ops.CZ(a, b)]).without_operations_touching([]) ==
            Moment([ops.CZ(a, b)]))
    assert (Moment([ops.CZ(a, b)]).without_operations_touching([a]) ==
            Moment())
    assert (Moment([ops.CZ(a, b)]).without_operations_touching([b]) ==
            Moment())
    assert (Moment([ops.CZ(a, b)]).without_operations_touching([c]) ==
            Moment([ops.CZ(a, b)]))

    # Multiple operation case.
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([]) ==
            Moment([ops.CZ(a, b), ops.X(c)]))
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([a]) ==
            Moment([ops.X(c)]))
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([b]) ==
            Moment([ops.X(c)]))
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([c]) ==
            Moment([ops.CZ(a, b)]))
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([a, b]) ==
            Moment([ops.X(c)]))
    assert (Moment([ops.CZ(a, b),
                    ops.X(c)]).without_operations_touching([a, c]) ==
            Moment())


def test_qubits():
    a = ops.QubitId()
    b = ops.QubitId()

    assert Moment([ops.X(a), ops.X(b)]).qubits == {a , b}
    assert Moment([ops.X(a)]).qubits == {a}
    assert Moment([ops.CZ(a, b)]).qubits == {a, b}
