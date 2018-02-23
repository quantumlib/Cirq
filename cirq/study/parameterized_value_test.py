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

from cirq.study import ParameterizedValue
from cirq.testing import EqualsTester


def test_parameterized_value_init():
    r = ParameterizedValue('', 5)
    assert isinstance(r, int)
    assert r == 5

    s = ParameterizedValue('a', 6)
    assert isinstance(s, ParameterizedValue)
    assert s.val == 6
    assert s.key == 'a'


def test_parameterized_value_eq():
    eq = EqualsTester()
    eq.add_equality_group(5,
                          ParameterizedValue(val=5),
                          ParameterizedValue(val=5, factor=2))
    eq.add_equality_group(ParameterizedValue('', 2.5), 2.5)
    eq.make_equality_pair(lambda: ParameterizedValue('rr', -1))
    eq.make_equality_pair(lambda: ParameterizedValue('rr', 0))
    eq.make_equality_pair(lambda: ParameterizedValue('ra', 0))
    eq.add_equality_group(ParameterizedValue('rr', -1, factor=2))


def test_parameterized_value_shift():
    assert (ParameterizedValue('e', 0.5) + 1.5 ==
            ParameterizedValue('e', 2))

    assert (ParameterizedValue('e', 0.5, factor=3) + 1.5 ==
            ParameterizedValue('e', 2, factor=3))

    assert (5 + ParameterizedValue('a', 0.5) ==
            ParameterizedValue('a', 5.5))

    assert (ParameterizedValue('b', 0.5) - 1.5 ==
            ParameterizedValue('b', -1))

    assert (ParameterizedValue('c', 0.5) - 1 ==
            ParameterizedValue('c', -0.5))


def test_parameterized_value_scale():
    assert (-ParameterizedValue('e', 2, factor=3) ==
            ParameterizedValue('e', -2, factor=-3))

    assert (5 * ParameterizedValue('e', 2, factor=3) ==
            ParameterizedValue('e', 10, factor=15))

    assert (ParameterizedValue('e', 2, factor=3) * 5 ==
            ParameterizedValue('e', 10, factor=15))

    assert (ParameterizedValue('e', 2, factor=3) / 2 ==
            ParameterizedValue('e', 1, factor=1.5))


def test_parameterized_value_of():
    assert ParameterizedValue.val_of(5) == 5
    assert ParameterizedValue.key_of(5) == ''

    e = ParameterizedValue('rr', -1)
    assert ParameterizedValue.val_of(e) == -1
    assert ParameterizedValue.key_of(e) == 'rr'


def test_parameterized_value_repr():
    assert repr(ParameterizedValue('1',
                                   2,
                                   3)) == "ParameterizedValue('1', 2, 3)"


def test_parameterized_value_str():
    assert str(ParameterizedValue()) == '0'
    assert str(ParameterizedValue(val=2)) == '2'

    assert str(ParameterizedValue(key='a', factor=0)) == '0'
    assert str(ParameterizedValue(key='a', factor=1)) == 'a'
    assert str(ParameterizedValue(key='a', factor=3)) == 'a*3'
    assert str(ParameterizedValue(key='a', factor=0, val=2)) == '2'
    assert str(ParameterizedValue(key='a', factor=1, val=2)) == '2+a'
    assert str(ParameterizedValue(key='a', factor=3, val=2)) == '2 + a*3'

    assert str(ParameterizedValue(key='+', factor=0)) == '0'
    assert str(ParameterizedValue(key='+', factor=1)) == "param('+')"
    assert str(ParameterizedValue(key='+', factor=3)) == "param('+')*3"
    assert str(ParameterizedValue(key='+', factor=0, val=2)) == '2'
    assert str(ParameterizedValue(key='+', factor=1, val=2)) == "2+param('+')"
    assert str(ParameterizedValue(key='+',
                                  factor=3,
                                  val=2)) == "2 + param('+')*3"
