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

import pytest

import cirq


def test_empty_init():
    mkey = cirq.MeasurementKey()
    assert mkey.name == ''
    mkey2 = cirq.MeasurementKey('')
    assert mkey2.name == ''


@pytest.mark.parametrize(
    'qubits',
    [
        (cirq.LineQubit(0), cirq.LineQubit(1)),
        (cirq.NamedQubit('a'), cirq.NamedQubit('b')),
        (cirq.GridQubit(1, 2), cirq.GridQubit(2, 3)),
    ],
)
def test_qubit_key(qubits):
    qubit_key = cirq.MeasurementKey(qubits=qubits)

    assert qubit_key.name == ''
    assert qubit_key.path == tuple()
    assert qubit_key.qubits == qubits


def test_nested_key():
    with pytest.raises(ValueError, match=': is not allowed.*use `MeasurementKey.parse_serialized'):
        _ = cirq.MeasurementKey('nested:key')
    nested_key = cirq.MeasurementKey.parse_serialized('nested:key')

    assert nested_key.name == 'key'
    assert nested_key.path == ('nested',)

    qubits = (cirq.LineQubit(0), cirq.LineQubit(1))
    nested_qubit_key = cirq.MeasurementKey(qubits=qubits, path=('nested',))

    assert nested_qubit_key.name == ''
    assert nested_qubit_key.path == ('nested',)
    assert nested_qubit_key.qubits == qubits


def test_eq_and_hash():
    class SomeRandomClass:
        def __init__(self, some_str):
            self.some_str = some_str

        def __str__(self):
            return self.some_str  # coverage: ignore

    mkey = cirq.MeasurementKey('key')
    assert mkey == 'key'
    assert hash(mkey) == hash('key')
    nested_key = cirq.MeasurementKey.parse_serialized('nested:key')
    assert nested_key == 'nested:key'
    assert hash(nested_key) == hash('nested:key')
    qubits = (cirq.LineQubit(0), cirq.LineQubit(1))
    qubit_key = cirq.MeasurementKey(qubits=qubits)
    assert qubit_key == cirq.default_measurement_key_str(qubits)
    assert hash(qubit_key) == hash(cirq.default_measurement_key_str(qubits))
    assert qubit_key != cirq.MeasurementKey(name=cirq.default_measurement_key_str(qubits))
    assert hash(qubit_key) == hash(
        cirq.MeasurementKey(name=cirq.default_measurement_key_str(qubits))
    )
    non_str_or_measurement_key = SomeRandomClass('key')
    assert mkey != non_str_or_measurement_key


@pytest.mark.parametrize('key_string', ['key', 'nested:key', '0,1,2', 'nested:0,1,2'])
def test_str(key_string):
    mkey = cirq.MeasurementKey.parse_serialized(key_string)
    assert str(mkey) == key_string
    assert str(mkey) == mkey


def test_repr():
    mkey = cirq.MeasurementKey('key_string')
    assert repr(mkey) == f'cirq.MeasurementKey(name=\'key_string\')'
    qubits = (cirq.LineQubit(0), cirq.LineQubit(1))
    qubit_key = cirq.MeasurementKey(qubits=qubits)
    assert (
        repr(qubit_key)
        == f'cirq.MeasurementKey(qubits=(\'cirq.LineQubit(0)\', \'cirq.LineQubit(1)\'))'
    )
    mkey = cirq.MeasurementKey.parse_serialized('nested:key')
    assert repr(mkey) == f'cirq.MeasurementKey(path=(\'nested\',), name=\'key\')'


def test_json_dict():
    mkey = cirq.MeasurementKey('key')
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'key',
        'path': tuple(),
        'qubits': tuple(),
    }
    mkey = cirq.MeasurementKey.parse_serialized('nested:key')
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'key',
        'path': ('nested',),
        'qubits': tuple(),
    }
    qubits = (cirq.LineQubit(0), cirq.LineQubit(1))
    mkey = cirq.MeasurementKey(qubits=qubits)
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': '',
        'path': tuple(),
        'qubits': qubits,
    }


@pytest.mark.parametrize(
    'mkey',
    [
        cirq.MeasurementKey('key'),
        cirq.MeasurementKey(qubits=(cirq.LineQubit(0), cirq.LineQubit(1))),
    ],
)
def test_with_key_path(mkey):
    mkey2 = cirq.with_key_path(mkey, ('a',))
    assert mkey2.name == mkey.name
    assert mkey2.qubits == mkey.qubits
    assert mkey2.path == ('a',)
    assert mkey2 == mkey.with_key_path_prefix('a')

    mkey3 = mkey2.with_key_path_prefix('b')
    assert mkey3.name == mkey.name
    assert mkey3.qubits == mkey.qubits
    assert mkey3.path == ('b', 'a')


@pytest.mark.parametrize(
    ['mkey', 'base_key_str'],
    [
        (cirq.MeasurementKey('key'), 'key'),
        (cirq.MeasurementKey(qubits=(cirq.LineQubit(0), cirq.LineQubit(1))), '0,1'),
        (cirq.MeasurementKey(qubits=(cirq.NamedQubit('a'), cirq.NamedQubit('b'))), 'a,b'),
        (cirq.MeasurementKey(qubits=(cirq.GridQubit(1, 2), cirq.GridQubit(2, 3))), '(1, 2),(2, 3)'),
    ],
)
def test_with_measurement_key_mapping(mkey, base_key_str):
    print(mkey)
    with pytest.raises(ValueError, match='Remapped key should be non-empty'):
        _ = cirq.with_measurement_key_mapping(mkey, {base_key_str: ''})
    mkey2 = cirq.with_measurement_key_mapping(mkey, {base_key_str: 'new_key'})
    assert mkey2.name == 'new_key'
    assert mkey2.qubits == mkey.qubits

    mkey3 = mkey2.with_key_path_prefix('a')
    mkey3 = cirq.with_measurement_key_mapping(mkey3, {'new_key': 'newer_key'})
    assert mkey3.name == 'newer_key'
    assert mkey3.path == ('a',)
    assert mkey3.qubits == mkey.qubits


def test_with_qubit_mapping():
    qubits = (cirq.LineQubit(0), cirq.LineQubit(1))
    mkey = cirq.MeasurementKey(qubits=qubits)
    assert mkey == '0,1'

    qubits2 = (cirq.LineQubit(2), cirq.LineQubit(3))
    assert mkey.with_qubits(qubits2) == '2,3'
    assert mkey.with_qubits(qubits2) != mkey
    assert mkey.with_qubits(qubits2) != mkey.replace(name='2,3')

    assert mkey.with_key_path_prefix('a').with_qubits(qubits2) == 'a:2,3'

    assert mkey.replace(name='key') == 'key'
    assert mkey.replace(name='key').with_qubits(qubits2) == 'key'
    assert mkey.replace(name='key').with_qubits(qubits2) != mkey.replace(name='key')
