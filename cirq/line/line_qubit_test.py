# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cirq


def test_init():
    q = cirq.LineQubit(1)
    assert q.x == 1


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.LineQubit(1))
    eq.add_equality_group(cirq.LineQubit(2))
    eq.add_equality_group(cirq.LineQubit(0))


def test_str():
    assert str(cirq.LineQubit(5)) == '5'


def test_repr():
    assert repr(cirq.LineQubit(5)) == 'cirq.LineQubit(5)'


def test_cmp():
    assert cirq.LineQubit(0) == cirq.LineQubit(0)
    assert cirq.LineQubit(0) != cirq.LineQubit(1)
    assert cirq.LineQubit(0) < cirq.LineQubit(1)
    assert cirq.LineQubit(1) > cirq.LineQubit(0)
    assert cirq.LineQubit(0) <= cirq.LineQubit(0)
    assert cirq.LineQubit(0) <= cirq.LineQubit(1)
    assert cirq.LineQubit(0) >= cirq.LineQubit(0)
    assert cirq.LineQubit(1) >= cirq.LineQubit(0)


def test_cmp_failure():
    with pytest.raises(TypeError):
        _ = 0 < cirq.LineQubit(1)
    with pytest.raises(TypeError):
        _ = cirq.LineQubit(1) < 0


def test_is_adjacent():
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(2))
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(0))
    assert cirq.LineQubit(2).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(1).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(2).is_adjacent(cirq.LineQubit(0))


def test_range():
    assert cirq.LineQubit.range(0) == []
    assert cirq.LineQubit.range(1) == [cirq.LineQubit(0)]
    assert cirq.LineQubit.range(2) == [cirq.LineQubit(0), cirq.LineQubit(1)]
    assert cirq.LineQubit.range(5) == [
        cirq.LineQubit(0),
        cirq.LineQubit(1),
        cirq.LineQubit(2),
        cirq.LineQubit(3),
        cirq.LineQubit(4),
    ]

    assert cirq.LineQubit.range(0, 0) == []
    assert cirq.LineQubit.range(0, 1) == [cirq.LineQubit(0)]
    assert cirq.LineQubit.range(
        1, 4) == [cirq.LineQubit(1),
                  cirq.LineQubit(2),
                  cirq.LineQubit(3)]

    assert cirq.LineQubit.range(3, 1,
                                -1) == [cirq.LineQubit(3),
                                        cirq.LineQubit(2)]
    assert cirq.LineQubit.range(3, 5, -1) == []
    assert cirq.LineQubit.range(1, 5,
                                2) == [cirq.LineQubit(1),
                                       cirq.LineQubit(3)]


def test_addition_subtraction():
    assert cirq.LineQubit(1) + 2 == cirq.LineQubit(3)
    assert cirq.LineQubit(3) - 1 == cirq.LineQubit(2)
    assert 1 + cirq.LineQubit(4) == cirq.LineQubit(5)
    assert 5 - cirq.LineQubit(3) == cirq.LineQubit(2)


def test_addition_subtraction_type_error():
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) + 'dave'
    with pytest.raises(TypeError, match='dave'):
        _ = cirq.LineQubit(1) - 'dave'


def test_neg():
    assert -cirq.LineQubit(1) == cirq.LineQubit(-1)
