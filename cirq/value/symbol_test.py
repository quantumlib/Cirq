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


def test_parameterized_value_init():
    assert cirq.Symbol('a').name == 'a'
    assert cirq.Symbol('b').name == 'b'


def test_string_representation():
    assert str(cirq.Symbol('a1')) == 'a1'
    assert str(cirq.Symbol('_b23_')) == '_b23_'
    assert str(cirq.Symbol('1a')) == 'Symbol("1a")'
    assert str(cirq.Symbol('&%#')) == 'Symbol("&%#")'
    assert str(cirq.Symbol('')) == 'Symbol("")'


@cirq.testing.only_test_in_python3
def test_repr():
    assert repr(cirq.Symbol('a1')) == "cirq.Symbol('a1')"
    assert repr(cirq.Symbol('_b23_')) == "cirq.Symbol('_b23_')"
    assert repr(cirq.Symbol('1a')) == "cirq.Symbol('1a')"
    assert repr(cirq.Symbol('&%#')) == "cirq.Symbol('&%#')"
    assert repr(cirq.Symbol('')) == "cirq.Symbol('')"


def test_parameterized_value_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.Symbol('a'))
    eq.make_equality_group(lambda: cirq.Symbol('rr'))


def test_identity_operations():
    s = cirq.Symbol('s')
    assert s == s * 1 == 1 * s == 1.0 * s * 1.0
    assert s == s + 0 == 0 + s == 0.0 + s + 0.0

    with pytest.raises(TypeError):
        _ = s + s
    with pytest.raises(TypeError):
        _ = s + 1
    with pytest.raises(TypeError):
        _ = 1 + s
    with pytest.raises(TypeError):
        _ = s * s
    with pytest.raises(TypeError):
        _ = s * 2
    with pytest.raises(TypeError):
        _ = 2 * s
