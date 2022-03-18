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
import sympy

import cirq

key_a = cirq.MeasurementKey.parse_serialized('0:a')
key_b = cirq.MeasurementKey.parse_serialized('0:b')
key_c = cirq.MeasurementKey.parse_serialized('0:c')
init_key_condition = cirq.KeyCondition(key_a)
init_sympy_condition = cirq.SympyCondition(sympy.Symbol('0:a') >= 1)


def test_key_condition_with_keys():
    c = init_key_condition.replace_key(key_a, key_b)
    assert c.key is key_b
    c = init_key_condition.replace_key(key_b, key_c)
    assert c.key is key_a


def test_key_condition_str():
    assert str(init_key_condition) == '0:a'
    assert str(cirq.KeyCondition(key_a, index=-2)) == '0:a[-2]'


def test_key_condition_repr():
    cirq.testing.assert_equivalent_repr(init_key_condition)
    cirq.testing.assert_equivalent_repr(cirq.KeyCondition(key_a, index=-2))


def test_key_condition_resolve():
    def resolve(records):
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return init_key_condition.resolve(classical_data)

    assert resolve({'0:a': [[1]]})
    assert resolve({'0:a': [[2]]})
    assert resolve({'0:a': [[0, 1]]})
    assert resolve({'0:a': [[1, 0]]})
    assert not resolve({'0:a': [[0]]})
    assert not resolve({'0:a': [[0, 0]]})
    assert not resolve({'0:a': [[]]})
    assert not resolve({'0:a': [[0]], 'b': [[1]]})
    with pytest.raises(
        ValueError, match='Measurement key 0:a missing when testing classical control'
    ):
        _ = resolve({})
    with pytest.raises(
        ValueError, match='Measurement key 0:a missing when testing classical control'
    ):
        _ = resolve({'0:b': [[1]]})


def test_key_condition_qasm():
    assert cirq.KeyCondition(cirq.MeasurementKey('a')).qasm == 'm_a!=0'


def test_sympy_condition_with_keys():
    c = init_sympy_condition.replace_key(key_a, key_b)
    assert c.keys == (key_b,)
    c = init_sympy_condition.replace_key(key_b, key_c)
    assert c.keys == (key_a,)


def test_sympy_condition_str():
    assert str(init_sympy_condition) == '0:a >= 1'


def test_sympy_condition_repr():
    cirq.testing.assert_equivalent_repr(init_sympy_condition)


def test_sympy_condition_resolve():
    def resolve(records):
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return init_sympy_condition.resolve(classical_data)

    assert resolve({'0:a': [[1]]})
    assert resolve({'0:a': [[2]]})
    assert resolve({'0:a': [[0, 1]]})
    assert resolve({'0:a': [[1, 0]]})
    assert not resolve({'0:a': [[0]]})
    assert not resolve({'0:a': [[0, 0]]})
    assert not resolve({'0:a': [[]]})
    assert not resolve({'0:a': [[0]], 'b': [[1]]})
    with pytest.raises(
        ValueError,
        match=re.escape("Measurement keys ['0:a'] missing when testing classical control"),
    ):
        _ = resolve({})
    with pytest.raises(
        ValueError,
        match=re.escape("Measurement keys ['0:a'] missing when testing classical control"),
    ):
        _ = resolve({'0:b': [[1]]})


def test_sympy_condition_qasm():
    with pytest.raises(NotImplementedError):
        _ = init_sympy_condition.qasm
