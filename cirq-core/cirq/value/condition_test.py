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

from __future__ import annotations

import re

import pytest
import sympy

import cirq

key_a = cirq.MeasurementKey.parse_serialized('0:a')
key_b = cirq.MeasurementKey.parse_serialized('0:b')
key_c = cirq.MeasurementKey.parse_serialized('0:c')
init_key_condition = cirq.KeyCondition(key_a)
init_sympy_condition = cirq.SympyCondition(sympy.Symbol('0:a') >= 1)


def test_key_condition_with_keys() -> None:
    c = init_key_condition.replace_key(key_a, key_b)
    assert c.key is key_b
    c = init_key_condition.replace_key(key_b, key_c)
    assert c.key is key_a

    indexed = cirq.KeyCondition(key_a, index=0)
    assert indexed.replace_key(key_a, key_b) == cirq.KeyCondition(key_b, index=0)
    assert indexed.replace_key(key_b, key_c) is indexed
    assert cirq.with_measurement_key_mapping(indexed, {'a': 'b'}) == cirq.KeyCondition(
        key_b, index=0
    )
    assert cirq.with_key_path_prefix(indexed, ('p',)) == cirq.KeyCondition(
        key_a.with_key_path_prefix('p'), index=0
    )
    assert cirq.with_rescoped_keys(indexed, ('p',), frozenset([key_a])) == indexed


def test_key_condition_json_roundtrip() -> None:
    cirq.testing.assert_json_roundtrip_works(cirq.KeyCondition(key_a, index=0))
    cirq.testing.assert_json_roundtrip_works(cirq.KeyCondition(key_a, index=-2))

    # Legacy JSON payloads without an "index" field default to -1.
    legacy_json = (
        '{"cirq_type": "KeyCondition", '
        '"key": {"cirq_type": "MeasurementKey", "name": "a", "path": []}}'
    )
    assert cirq.read_json(json_text=legacy_json) == cirq.KeyCondition(cirq.MeasurementKey('a'))


def test_key_condition_str() -> None:
    assert str(init_key_condition) == '0:a'
    assert str(cirq.KeyCondition(key_a, index=-2)) == '0:a[-2]'


def test_key_condition_repr() -> None:
    cirq.testing.assert_equivalent_repr(init_key_condition)
    cirq.testing.assert_equivalent_repr(cirq.KeyCondition(key_a, index=-2))


def test_key_condition_resolve() -> None:
    def resolve(records: dict[cirq.MeasurementKey, list[tuple[int, ...]]]) -> bool:
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return init_key_condition.resolve(classical_data)

    mkey_0a = cirq.MeasurementKey.parse_serialized('0:a')
    mkey_0b = cirq.MeasurementKey.parse_serialized('0:b')
    assert resolve({mkey_0a: [(1,)]})
    assert resolve({mkey_0a: [(2,)]})
    assert resolve({mkey_0a: [(0, 1)]})
    assert resolve({mkey_0a: [(1, 0)]})
    assert not resolve({mkey_0a: [(0,)]})
    assert not resolve({mkey_0a: [(0, 0)]})
    assert not resolve({mkey_0a: [()]})
    assert not resolve({mkey_0a: [(0,)], mkey_0b: [(1,)]})
    with pytest.raises(
        ValueError, match='Measurement key 0:a missing when testing classical control'
    ):
        _ = resolve({})
    with pytest.raises(
        ValueError, match='Measurement key 0:a missing when testing classical control'
    ):
        _ = resolve({mkey_0b: [(1,)]})


def test_key_condition_qasm() -> None:
    with pytest.raises(ValueError, match='QASM is defined only for SympyConditions'):
        _ = cirq.KeyCondition(cirq.MeasurementKey('a')).qasm


def test_key_condition_qasm_protocol() -> None:
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'}, meas_key_bitcount={'m_a': 1})
    qasm = cirq.qasm(cond, args=args)
    assert qasm == 'm_a==1'


def test_key_condition_qasm_protocol_v3() -> None:
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    args = cirq.QasmArgs(meas_key_id_map={'a': 'm_a'}, version='3.0')
    qasm = cirq.qasm(cond, args=args)
    assert qasm == 'm_a!=0'


@pytest.mark.parametrize('version', ['2.0', '3.0'])
def test_key_condition_qasm_protocol_rejects_non_default_index(version: str) -> None:
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'), index=0)
    args = cirq.QasmArgs(
        meas_key_id_map={'a': 'm_a'}, meas_key_bitcount={'m_a': 1}, version=version
    )
    with pytest.raises(
        ValueError, match='Only most recent measurement at key can be used for QASM'
    ):
        _ = cirq.qasm(cond, args=args)


def test_key_condition_qasm_protocol_invalid_args() -> None:
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


def test_sympy_condition_validation() -> None:
    with pytest.raises(TypeError, match="Expected SymPy expression, got.*class 'str'"):
        _ = cirq.SympyCondition("0:a >= 1")
    with pytest.raises(TypeError, match="Expected SymPy expression, got.*class 'list'"):
        _ = cirq.SympyCondition([sympy.Symbol('0:a') >= 1])


def test_sympy_condition_with_keys() -> None:
    c = init_sympy_condition.replace_key(key_a, key_b)
    assert c.keys == (key_b,)
    c = init_sympy_condition.replace_key(key_b, key_c)
    assert c.keys == (key_a,)


def test_sympy_condition_str() -> None:
    assert str(init_sympy_condition) == '0:a >= 1'


def test_sympy_condition_repr() -> None:
    cirq.testing.assert_equivalent_repr(init_sympy_condition)


def test_sympy_condition_resolve() -> None:
    def resolve(records: dict[cirq.MeasurementKey, list[tuple[int, ...]]]) -> bool:
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return init_sympy_condition.resolve(classical_data)

    mkey_0a = cirq.MeasurementKey.parse_serialized('0:a')
    mkey_0b = cirq.MeasurementKey.parse_serialized('0:b')
    assert resolve({mkey_0a: [(1,)]})
    assert resolve({mkey_0a: [(2,)]})
    assert resolve({mkey_0a: [(0, 1)]})
    assert resolve({mkey_0a: [(1, 0)]})
    assert not resolve({mkey_0a: [(0,)]})
    assert not resolve({mkey_0a: [(0, 0)]})
    assert not resolve({mkey_0a: [()]})
    assert not resolve({mkey_0a: [(0,)], mkey_0b: [(1,)]})
    with pytest.raises(
        ValueError,
        match=re.escape("Measurement keys ['0:a'] missing when testing classical control"),
    ):
        _ = resolve({})
    with pytest.raises(
        ValueError,
        match=re.escape("Measurement keys ['0:a'] missing when testing classical control"),
    ):
        _ = resolve({mkey_0b: [(1,)]})


def test_sympy_indexed_condition() -> None:
    a = sympy.IndexedBase('a')
    cond = cirq.SympyCondition(sympy.Xor(a[0], a[1]))
    mkey_a = cirq.MeasurementKey('a')
    assert cond.keys == (mkey_a,)
    assert str(cond) == 'a[0] ^ a[1]'

    def resolve(records: dict[cirq.MeasurementKey, list[tuple[int, ...]]]) -> bool:
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return cond.resolve(classical_data)

    assert not resolve({mkey_a: [(0, 0)]})
    assert resolve({mkey_a: [(1, 0)]})
    assert resolve({mkey_a: [(0, 1)]})
    assert not resolve({mkey_a: [(1, 1)]})
    assert resolve({mkey_a: [(0, 1, 0)]})
    assert resolve({mkey_a: [(0, 1, 1)]})
    assert not resolve({mkey_a: [(1, 1, 0)]})
    assert not resolve({mkey_a: [(1, 1, 1)]})
    with pytest.raises(IndexError):
        assert resolve({mkey_a: [()]})
    with pytest.raises(IndexError):
        assert resolve({mkey_a: [(0,)]})
    with pytest.raises(IndexError):
        assert resolve({mkey_a: [(1,)]})


def test_sympy_indexed_condition_qudits() -> None:
    a = sympy.IndexedBase('a')
    cond = cirq.SympyCondition(sympy.And(a[1] >= 2, a[2] <= 3))
    mkey_a = cirq.MeasurementKey('a')
    assert cond.keys == (mkey_a,)
    assert str(cond) == '(a[1] >= 2) & (a[2] <= 3)'

    def resolve(records: dict[cirq.MeasurementKey, list[tuple[int, ...]]]) -> bool:
        classical_data = cirq.ClassicalDataDictionaryStore(_records=records)
        return cond.resolve(classical_data)

    assert not resolve({mkey_a: [(0, 0, 0)]})
    assert not resolve({mkey_a: [(0, 1, 0)]})
    assert resolve({mkey_a: [(0, 2, 0)]})
    assert resolve({mkey_a: [(0, 3, 0)]})
    assert not resolve({mkey_a: [(0, 0, 4)]})
    assert not resolve({mkey_a: [(0, 1, 4)]})
    assert not resolve({mkey_a: [(0, 2, 4)]})
    assert not resolve({mkey_a: [(0, 3, 4)]})


def test_sympy_condition_qasm() -> None:
    # Measurements get prepended with "m_", so the condition needs to be too.
    assert cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), 2)).qasm == 'm_a==2'
    with pytest.raises(
        ValueError, match='QASM is defined only for SympyConditions of type key == constant'
    ):
        _ = cirq.SympyCondition(sympy.Unequality(sympy.Symbol('a'), 2)).qasm


@pytest.mark.parametrize(
    ['cond', 'cond_str'],
    [
        (
            cirq.BitMaskKeyCondition(
                bitmask=None, equal_target=False, index=59, key='a', target_value=0
            ),
            'a[59]',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=None, equal_target=False, index=-1, key='a', target_value=0
            ),
            'a',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=None, equal_target=False, index=58, key='b', target_value=3
            ),
            'b[58] != 3',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=None, equal_target=False, index=-1, key='b', target_value=3
            ),
            'b != 3',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=False, index=57, key='c', target_value=0
            ),
            'c[57] & 13',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=False, index=-1, key='c', target_value=0
            ),
            'c & 13',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=False, index=56, key='d', target_value=12
            ),
            '(d[56] & 13) != 12',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=False, index=-1, key='d', target_value=12
            ),
            '(d & 13) != 12',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=True, index=55, key='d', target_value=12
            ),
            '(d[55] & 13) == 12',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=13, equal_target=True, index=-1, key='d', target_value=12
            ),
            '(d & 13) == 12',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=11, equal_target=True, index=54, key='e', target_value=11
            ),
            '(e[54] & 11) == 11',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=11, equal_target=True, index=-1, key='e', target_value=11
            ),
            '(e & 11) == 11',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=9, equal_target=False, index=53, key='e', target_value=9
            ),
            '(e[53] & 9) != 9',
        ),
        (
            cirq.BitMaskKeyCondition(
                bitmask=9, equal_target=False, index=-1, key='e', target_value=9
            ),
            '(e & 9) != 9',
        ),
    ],
)
def test_bitmask_condition_str(cond: cirq.BitMaskKeyCondition, cond_str: str) -> None:
    assert str(cond) == cond_str


@pytest.mark.parametrize(
    ['cond', 'value'],
    [
        (cirq.BitMaskKeyCondition('c', bitmask=13, target_value=9), False),
        (cirq.BitMaskKeyCondition('d', bitmask=13, target_value=12, equal_target=True), True),
    ],
)
def test_bitmask_condition_resolve(cond: cirq.BitMaskKeyCondition, value: bool) -> None:
    resolver = cirq.ClassicalDataDictionaryStore(
        _records={
            cirq.MeasurementKey('c'): [(1, 0, 0, 1)],
            cirq.MeasurementKey('d'): [(1, 0, 1, 1, 0, 0)],
        }
    )
    assert cond.resolve(resolver) == value


def test_bitmask_condition_resolve_invalid_input_raises() -> None:
    cond = cirq.BitMaskKeyCondition('a')
    resolver = cirq.ClassicalDataDictionaryStore(
        _records={
            cirq.MeasurementKey('c'): [(1, 0, 0, 1)],
            cirq.MeasurementKey('d'): [(1, 0, 1, 1, 0, 0)],
        }
    )
    with pytest.raises(ValueError):
        _ = cond.resolve(resolver)


@pytest.mark.parametrize(
    'cond',
    [
        cirq.BitMaskKeyCondition(
            bitmask=None, equal_target=False, index=59, key='a', target_value=0
        ),
        cirq.BitMaskKeyCondition(
            bitmask=None, equal_target=False, index=-1, key='a', target_value=0
        ),
        cirq.BitMaskKeyCondition(
            bitmask=None, equal_target=False, index=58, key='b', target_value=3
        ),
        cirq.BitMaskKeyCondition(
            bitmask=None, equal_target=False, index=-1, key='b', target_value=3
        ),
        cirq.BitMaskKeyCondition(bitmask=13, equal_target=False, index=57, key='c', target_value=0),
        cirq.BitMaskKeyCondition(bitmask=13, equal_target=False, index=-1, key='c', target_value=0),
        cirq.BitMaskKeyCondition(
            bitmask=13, equal_target=False, index=56, key='d', target_value=12
        ),
        cirq.BitMaskKeyCondition(
            bitmask=13, equal_target=False, index=-1, key='d', target_value=12
        ),
        cirq.BitMaskKeyCondition(bitmask=13, equal_target=True, index=55, key='d', target_value=12),
        cirq.BitMaskKeyCondition(bitmask=13, equal_target=True, index=-1, key='d', target_value=12),
        cirq.BitMaskKeyCondition(bitmask=11, equal_target=True, index=54, key='e', target_value=11),
        cirq.BitMaskKeyCondition(bitmask=11, equal_target=True, index=-1, key='e', target_value=11),
        cirq.BitMaskKeyCondition(bitmask=9, equal_target=False, index=53, key='e', target_value=9),
        cirq.BitMaskKeyCondition(bitmask=9, equal_target=False, index=-1, key='e', target_value=9),
    ],
)
def test_bitmask_condition_repr(cond: cirq.BitMaskKeyCondition) -> None:
    cirq.testing.assert_equivalent_repr(cond)


def test_bitmask_condition_keys() -> None:
    assert cirq.BitMaskKeyCondition('test').keys == ('test',)


def test_bitmask_create_equal_mask() -> None:
    assert cirq.BitMaskKeyCondition.create_equal_mask('a', 9) == cirq.BitMaskKeyCondition(
        'a', equal_target=True, bitmask=9, target_value=9
    )


def test_bitmask_create_not_equal_mask() -> None:
    assert cirq.BitMaskKeyCondition.create_not_equal_mask('b', 14) == cirq.BitMaskKeyCondition(
        'b', equal_target=False, bitmask=14, target_value=14
    )


def test_bitmask_replace_key() -> None:
    cond = cirq.BitMaskKeyCondition('a')
    assert cond.replace_key('a', 'b') == cirq.BitMaskKeyCondition('b')
    assert cond.replace_key('c', 'd') is cond

    full = cirq.BitMaskKeyCondition('a', index=2, target_value=9, equal_target=True, bitmask=13)
    assert full.replace_key('a', 'b') == cirq.BitMaskKeyCondition(
        'b', index=2, target_value=9, equal_target=True, bitmask=13
    )
    assert full.replace_key(
        cirq.MeasurementKey('a'), cirq.MeasurementKey('b')
    ) == cirq.BitMaskKeyCondition('b', index=2, target_value=9, equal_target=True, bitmask=13)
    assert full.replace_key('c', 'd') is full
    assert cirq.with_measurement_key_mapping(full, {'a': 'b'}) == cirq.BitMaskKeyCondition(
        'b', index=2, target_value=9, equal_target=True, bitmask=13
    )
    assert cirq.with_key_path_prefix(full, ('p',)) == cirq.BitMaskKeyCondition(
        cirq.MeasurementKey('a', path=('p',)),
        index=2,
        target_value=9,
        equal_target=True,
        bitmask=13,
    )
    assert cirq.with_rescoped_keys(full, ('p',), frozenset([cirq.MeasurementKey('a')])) == full


def test_indexed_key_condition_in_circuit_operation() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    ops = [
        cirq.measure(q0, key='m'),
        cirq.X(q0),
        cirq.measure(q0, key='m'),
        cirq.X(q1).with_classical_controls(cirq.KeyCondition(cirq.MeasurementKey('m'), index=0)),
        cirq.measure(q1, key='out'),
    ]
    flat = cirq.Circuit(cirq.X(q0), *ops)
    nested = cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit(*ops)))
    sim = cirq.Simulator()
    assert sim.run(flat).records['out'].tolist() == [[[1]]]
    assert sim.run(nested).records['out'].tolist() == [[[1]]]


def test_bitmask_key_condition_in_circuit_operation() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    # First measurement is 0b11, second is 0b00; dropping any of index, bitmask,
    # target_value, or equal_target flips this condition from True to False.
    cond = cirq.BitMaskKeyCondition(
        'm', index=0, bitmask=0b01, target_value=0b01, equal_target=True
    )
    ops = [
        cirq.measure(q0, q1, key='m'),
        cirq.X(q0),
        cirq.X(q1),
        cirq.measure(q0, q1, key='m'),
        cirq.X(q2).with_classical_controls(cond),
        cirq.measure(q2, key='out'),
    ]
    flat = cirq.Circuit(cirq.X(q0), cirq.X(q1), *ops)
    nested = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.CircuitOperation(cirq.FrozenCircuit(*ops)))
    sim = cirq.Simulator()
    assert sim.run(flat).records['out'].tolist() == [[[1]]]
    assert sim.run(nested).records['out'].tolist() == [[[1]]]
