# Copyright 2018 Google LLC
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

from cirq.ops.qubit_order import default_sorting_key
from cirq.ops import QubitOrder, NamedQubit


def test_default_sorting_key():
    assert default_sorting_key('') == ''
    assert default_sorting_key('a') == 'a'
    assert default_sorting_key('a0') == 'a00000000:1'
    assert default_sorting_key('a00') == 'a00000000:2'
    assert default_sorting_key('a1bc23') == 'a00000001:1bc00000023:2'
    assert default_sorting_key('a9') == 'a00000009:1'
    assert default_sorting_key('a09') == 'a00000009:2'
    assert default_sorting_key('a00000000:8') == 'a00000000:8:00000008:1'


def test_sorted_by_default_sorting_key():
    actual = [
        '',
        '1',
        'a',
        'a00000000',
        'a00000000:8',
        'a9',
        'a09',
        'a10',
        'a11',
    ]
    assert sorted(actual, key=default_sorting_key) == actual
    assert sorted(reversed(actual), key=default_sorting_key) == actual


def test_default_basis():
    a2 = NamedQubit('a2')
    a10 = NamedQubit('a10')
    b = NamedQubit('b')
    assert QubitOrder.DEFAULT.order_for([]) == ()
    assert QubitOrder.DEFAULT.order_for([a10, a2, b]) == (a2, a10, b)


def test_explicit_basis():
    a2 = NamedQubit('a2')
    a10 = NamedQubit('a10')
    b = NamedQubit('b')
    with pytest.raises(ValueError):
        _ = QubitOrder.explicit([b, b])
    q = QubitOrder.explicit([a10, a2, b])
    assert q.order_for([b]) == (a10, a2, b)
    assert q.order_for([a2]) == (a10, a2, b)
    assert q.order_for([]) == (a10, a2, b)
    with pytest.raises(ValueError):
        _ = q.order_for([NamedQubit('c')])


def test_explicit_basis_with_fallback():
    a2 = NamedQubit('a2')
    a10 = NamedQubit('a10')
    b = NamedQubit('b')
    q = QubitOrder.explicit([b], fallback=QubitOrder.DEFAULT)
    assert q.order_for([]) == (b,)
    assert q.order_for([b]) == (b,)
    assert q.order_for([b, a2]) == (b, a2)
    assert q.order_for([a2]) == (b, a2)
    assert q.order_for([a10, a2]) == (b, a2, a10)


def test_sorted_by_basis():
    a = NamedQubit('2')
    b = NamedQubit('10')
    c = NamedQubit('-5')

    q = QubitOrder.sorted_by(lambda e: -int(str(e)))
    assert q.order_for([]) == ()
    assert q.order_for([a]) == (a,)
    assert q.order_for([a, b]) == (b, a)
    assert q.order_for([a, b, c]) == (b, a, c)


def test_map():
    b = NamedQubit('b!')
    q = QubitOrder.explicit([NamedQubit('b')]).map(
        internalize=lambda e: NamedQubit(e.name[:-1]),
        externalize=lambda e: NamedQubit(e.name + '!'))

    assert q.order_for([]) == (b,)
    assert q.order_for([b]) == (b,)


def test_qubit_order_or_list():
    b = NamedQubit('b')

    implied_by_list = QubitOrder.as_qubit_order([b])
    assert implied_by_list.order_for([]) == (b,)

    implied_by_generator = QubitOrder.as_qubit_order(
        NamedQubit(e.name + '!') for e in [b])
    assert implied_by_generator.order_for([]) == (NamedQubit('b!'),)
    assert implied_by_generator.order_for([]) == (NamedQubit('b!'),)

    ordered = QubitOrder.sorted_by(repr)
    passed_through = QubitOrder.as_qubit_order(ordered)
    assert ordered is passed_through
