# Copyright 2021 The Cirq Developers
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

import re

import pytest

import cirq

key_a = cirq.MeasurementKey('a')
key_b = cirq.MeasurementKey('b')
init_key_condition = cirq.KeyCondition(key_a)
init_sympy_condition = cirq.parse_sympy_condition('{a} >= 1')


def test_key_condition_with_keys():
    c = init_key_condition.with_keys(key_b)
    assert c.key is key_b


def test_key_condition_str():
    assert str(init_key_condition) == 'a'


def test_key_condition_resolve():
    assert init_key_condition.resolve({'a': [1]})
    assert init_key_condition.resolve({'a': [2]})
    assert init_key_condition.resolve({'a': [0, 1]})
    assert init_key_condition.resolve({'a': [1, 0]})
    assert not init_key_condition.resolve({'a': [0]})
    assert not init_key_condition.resolve({'a': [0, 0]})
    assert not init_key_condition.resolve({'a': []})
    assert not init_key_condition.resolve({'a': [0], 'b': [1]})
    with pytest.raises(
        ValueError, match='Measurement key a missing when testing classical control'
    ):
        _ = init_key_condition.resolve({})
    with pytest.raises(
        ValueError, match='Measurement key a missing when testing classical control'
    ):
        _ = init_key_condition.resolve({'b': [1]})


def test_key_condition_qasm():
    assert init_key_condition.qasm == 'm_a!=0'


def test_sympy_condition_with_keys():
    c = init_sympy_condition.with_keys(key_b)
    assert c.keys == (key_b,)


def test_sympy_condition_str():
    assert str(init_sympy_condition) == "a >= 1"


def test_sympy_condition_resolve():
    assert init_sympy_condition.resolve({'a': [1]})
    assert init_sympy_condition.resolve({'a': [2]})
    assert init_sympy_condition.resolve({'a': [0, 1]})
    assert init_sympy_condition.resolve({'a': [1, 0]})
    assert not init_sympy_condition.resolve({'a': [0]})
    assert not init_sympy_condition.resolve({'a': [0, 0]})
    assert not init_sympy_condition.resolve({'a': []})
    assert not init_sympy_condition.resolve({'a': [0], 'b': [1]})
    with pytest.raises(
        ValueError, match=re.escape("Measurement keys ['a'] missing when testing classical control")
    ):
        _ = init_sympy_condition.resolve({})
    with pytest.raises(
        ValueError, match=re.escape("Measurement keys ['a'] missing when testing classical control")
    ):
        _ = init_sympy_condition.resolve({'b': [1]})


def test_sympy_condition_qasm():
    with pytest.raises(NotImplementedError):
        _ = init_sympy_condition.qasm
