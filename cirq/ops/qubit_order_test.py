# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" qubit_order,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cirq


def test_default():
    a2 = cirq.NamedQubit('a2')
    a10 = cirq.NamedQubit('a10')
    b = cirq.NamedQubit('b')
    q4 = cirq.LineQubit(4)
    q5 = cirq.LineQubit(5)
    assert cirq.QubitOrder.DEFAULT.order_for([]) == ()
    assert cirq.QubitOrder.DEFAULT.order_for([a10, a2, b]) == (a2, a10, b)
    assert sorted([]) == []
    assert sorted([a10, a2, b]) == [a2, a10, b]
    assert sorted([q5, a10, a2, b, q4]) == [q4, q5, a2, a10, b]


def test_default_grouping():
    presorted = (
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(999, 999),
        cirq.LineQubit(0),
        cirq.LineQubit(1),
        cirq.LineQubit(999),
        cirq.NamedQubit(''),
        cirq.NamedQubit('0'),
        cirq.NamedQubit('1'),
        cirq.NamedQubit('999'),
        cirq.NamedQubit('a'),
    )
    assert cirq.QubitOrder.DEFAULT.order_for(presorted) == presorted
    assert cirq.QubitOrder.DEFAULT.order_for(reversed(presorted)) == presorted


def test_explicit():
    a2 = cirq.NamedQubit('a2')
    a10 = cirq.NamedQubit('a10')
    b = cirq.NamedQubit('b')
    with pytest.raises(ValueError):
        _ = cirq.QubitOrder.explicit([b, b])
    q = cirq.QubitOrder.explicit([a10, a2, b])
    assert q.order_for([b]) == (a10, a2, b)
    assert q.order_for([a2]) == (a10, a2, b)
    assert q.order_for([]) == (a10, a2, b)
    with pytest.raises(ValueError):
        _ = q.order_for([cirq.NamedQubit('c')])


def test_explicit_with_fallback():
    a2 = cirq.NamedQubit('a2')
    a10 = cirq.NamedQubit('a10')
    b = cirq.NamedQubit('b')
    q = cirq.QubitOrder.explicit([b], fallback=cirq.QubitOrder.DEFAULT)
    assert q.order_for([]) == (b,)
    assert q.order_for([b]) == (b,)
    assert q.order_for([b, a2]) == (b, a2)
    assert q.order_for([a2]) == (b, a2)
    assert q.order_for([a10, a2]) == (b, a2, a10)


def test_sorted_by():
    a = cirq.NamedQubit('2')
    b = cirq.NamedQubit('10')
    c = cirq.NamedQubit('-5')

    q = cirq.QubitOrder.sorted_by(lambda e: -int(str(e)))
    assert q.order_for([]) == ()
    assert q.order_for([a]) == (a,)
    assert q.order_for([a, b]) == (b, a)
    assert q.order_for([a, b, c]) == (b, a, c)


def test_map():
    b = cirq.NamedQubit('b!')
    q = cirq.QubitOrder.explicit([cirq.NamedQubit('b')]).map(
        internalize=lambda e: cirq.NamedQubit(e.name[:-1]),
        externalize=lambda e: cirq.NamedQubit(e.name + '!'))

    assert q.order_for([]) == (b,)
    assert q.order_for([b]) == (b,)


def test_qubit_order_or_list():
    b = cirq.NamedQubit('b')

    implied_by_list = cirq.QubitOrder.as_qubit_order([b])
    assert implied_by_list.order_for([]) == (b,)

    implied_by_generator = cirq.QubitOrder.as_qubit_order(
        cirq.NamedQubit(e.name + '!') for e in [b])
    assert implied_by_generator.order_for([]) == (cirq.NamedQubit('b!'),)
    assert implied_by_generator.order_for([]) == (cirq.NamedQubit('b!'),)

    ordered = cirq.QubitOrder.sorted_by(repr)
    passed_through = cirq.QubitOrder.as_qubit_order(ordered)
    assert ordered is passed_through
