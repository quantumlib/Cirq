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
from cirq import Timestamp, Duration


def test_init():
    assert Timestamp().raw_picos() == 0
    assert Timestamp(picos=513).raw_picos() == 513
    assert Timestamp(picos=-5).raw_picos() == -5
    assert Timestamp(nanos=211).raw_picos() == 211000
    assert Timestamp(nanos=211, picos=1051).raw_picos() == 212051

    assert isinstance(Timestamp(picos=1).raw_picos(), int)
    assert isinstance(Timestamp(nanos=1).raw_picos(), int)
    assert isinstance(Timestamp(picos=1.0).raw_picos(), float)
    assert isinstance(Timestamp(nanos=1.0).raw_picos(), float)


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(Timestamp(),
                          Timestamp(picos=0),
                          Timestamp(nanos=0.0))
    eq.add_equality_group(Timestamp(picos=1000), Timestamp(nanos=1))
    eq.make_equality_group(lambda: Timestamp(picos=-1))


def test_cmp():
    ordered_groups = [
        Timestamp(picos=-1),
        Timestamp(),
        Timestamp(picos=1),
        Timestamp(nanos=1),
        Timestamp(picos=2000),
    ]

    for i in range(len(ordered_groups)):
        for j in range(len(ordered_groups)):
            a = ordered_groups[i]
            b = ordered_groups[j]
            assert (i < j) == (a < b)
            assert (i <= j) == (a <= b)
            assert (i == j) == (a == b)
            assert (i != j) == (a != b)
            assert (i >= j) == (a >= b)
            assert (i > j) == (a > b)

    assert not (Timestamp() == 0)
    assert Timestamp() != 0
    assert not (Timestamp() == Duration())
    assert Timestamp() != Duration()


# In python 2, comparisons fallback to __cmp__ and don't fail.
# But a custom __cmp__ that does fail would result in == failing.
# So we throw up our hands and let it be.
@cirq.testing.only_test_in_python3
def test_cmp_vs_other_type():
    with pytest.raises(TypeError):
        _ = Timestamp() < Duration()
    with pytest.raises(TypeError):
        _ = Timestamp() < 0
    with pytest.raises(TypeError):
        _ = Timestamp() <= 0
    with pytest.raises(TypeError):
        _ = Timestamp() >= 0
    with pytest.raises(TypeError):
        _ = Timestamp() > 0


def test_add():
    assert Timestamp(picos=1) + Duration(picos=2) == Timestamp(picos=3)
    assert Duration(picos=3) + Timestamp(picos=-5) == Timestamp(picos=-2)

    with pytest.raises(TypeError):
        assert Timestamp() + Timestamp() == Timestamp()
    with pytest.raises(TypeError):
        _ = 1 + Timestamp()
    with pytest.raises(TypeError):
        _ = Timestamp() + 1


def test_sub():
    assert Timestamp() - Timestamp() == Duration()
    assert Timestamp(picos=1) - Timestamp(picos=2) == Duration(picos=-1)
    assert Timestamp(picos=5) - Duration(picos=2) == Timestamp(picos=3)

    with pytest.raises(TypeError):
        _ = Duration() - Timestamp()
    with pytest.raises(TypeError):
        _ = 1 - Timestamp()
    with pytest.raises(TypeError):
        _ = Timestamp() - 1
