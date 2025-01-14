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
    with pytest.raises(ValueError, match='QASM is defined only for SympyConditions'):
        _ = cirq.KeyCondition(cirq.MeasurementKey('a')).qasm


def test_key_condition_qasm_protocol():
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'}, meas_key_bitcount={'m_a': 1})
    qasm = cirq.qasm(cond, args=args)
    assert qasm == 'm_a==1'


def test_key_condition_qasm_protocol_v3():
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'}, version='3.0')
    qasm = cirq.qasm(cond, args=args)
    assert qasm == 'm_a!=0'


def test_key_condition_qasm_protocol_invalid_args():
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    args = cirq.QasmArgs()
    with pytest.raises(ValueError, match='Key "a" not in QasmArgs.meas_key_id_map.'):
        _ = cirq.qasm(cond, args=args)
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'})
    with pytest.raises(ValueError, match='Key "m_a" not in QasmArgs.meas_key_bitcount.'):
        _ = cirq.qasm(cond, args=args)
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'}, meas_key_bitcount={'m_a': 2})
    with pytest.raises(
        ValueError, match='QASM is defined only for single-bit classical conditions.'
    ):
        _ = cirq.qasm(cond, args=args)


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


def test_sympy_indexed_condition():
    a = sympy.IndexedBase('a')
    cond = cirq.SympyCondition(sympy.Xor(a[0], a[1]))
    assert cond.keys == (cirq.MeasurementKey('a'),)
    assert str(cond) == 'a[0] ^ a[1]'

    def resolve(records):
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return cond.resolve(classical_data)

    assert not resolve({'a': [(0, 0)]})
    assert resolve({'a': [(1, 0)]})
    assert resolve({'a': [(0, 1)]})
    assert not resolve({'a': [(1, 1)]})
    assert resolve({'a': [(0, 1, 0)]})
    assert resolve({'a': [(0, 1, 1)]})
    assert not resolve({'a': [(1, 1, 0)]})
    assert not resolve({'a': [(1, 1, 1)]})
    with pytest.raises(IndexError):
        assert resolve({'a': [()]})
    with pytest.raises(IndexError):
        assert resolve({'a': [(0,)]})
    with pytest.raises(IndexError):
        assert resolve({'a': [(1,)]})


def test_sympy_indexed_condition_qudits():
    a = sympy.IndexedBase('a')
    cond = cirq.SympyCondition(sympy.And(a[1] >= 2, a[2] <= 3))
    assert cond.keys == (cirq.MeasurementKey('a'),)
    assert str(cond) == '(a[1] >= 2) & (a[2] <= 3)'

    def resolve(records):
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return cond.resolve(classical_data)

    assert not resolve({'a': [(0, 0, 0)]})
    assert not resolve({'a': [(0, 1, 0)]})
    assert resolve({'a': [(0, 2, 0)]})
    assert resolve({'a': [(0, 3, 0)]})
    assert not resolve({'a': [(0, 0, 4)]})
    assert not resolve({'a': [(0, 1, 4)]})
    assert not resolve({'a': [(0, 2, 4)]})
    assert not resolve({'a': [(0, 3, 4)]})


def test_sympy_condition_qasm():
    # Measurements get prepended with "m_", so the condition needs to be too.
    assert cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), 2)).qasm == 'm_a==2'
    with pytest.raises(
        ValueError, match='QASM is defined only for SympyConditions of type key == constant'
    ):
        _ = cirq.SympyCondition(sympy.Symbol('a') != 2).qasm
