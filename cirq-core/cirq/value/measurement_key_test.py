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
    with pytest.raises(TypeError, match='required positional argument'):
        _ = cirq.MeasurementKey()
    with pytest.raises(ValueError, match='valid string'):
        _ = cirq.MeasurementKey(None)
    with pytest.raises(ValueError, match='valid string'):
        _ = cirq.MeasurementKey(4.2)
    # Initialization of empty string should be allowed
    _ = cirq.MeasurementKey('')


def test_nested_key():
    with pytest.raises(ValueError, match=': is not allowed.*use `MeasurementKey.parse_serialized'):
        _ = cirq.MeasurementKey('nested:key')
    nested_key = cirq.MeasurementKey.parse_serialized('nested:key')

    assert nested_key.name == 'key'
    assert nested_key.path == ('nested',)


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
    non_str_or_measurement_key = SomeRandomClass('key')
    assert mkey != non_str_or_measurement_key


@pytest.mark.parametrize('key_string', ['key', 'nested:key'])
def test_str(key_string):
    mkey = cirq.MeasurementKey.parse_serialized(key_string)
    assert str(mkey) == key_string
    assert str(mkey) == mkey


def test_repr():
    mkey = cirq.MeasurementKey('key_string')
    assert repr(mkey) == f"cirq.MeasurementKey(name='key_string')"
    assert eval(repr(mkey)) == mkey
    mkey = cirq.MeasurementKey.parse_serialized('nested:key')
    assert repr(mkey) == f"cirq.MeasurementKey(path=('nested',), name='key')"
    assert eval(repr(mkey)) == mkey


def test_json_dict():
    mkey = cirq.MeasurementKey('key')
    assert mkey._json_dict_() == {'cirq_type': 'MeasurementKey', 'name': 'key', 'path': tuple()}
    mkey = cirq.MeasurementKey.parse_serialized('nested:key')
    assert mkey._json_dict_() == {'cirq_type': 'MeasurementKey', 'name': 'key', 'path': ('nested',)}


def test_with_key_path():
    mkey = cirq.MeasurementKey('key')
    mkey2 = cirq.with_key_path(mkey, ('a',))
    assert mkey2.name == mkey.name
    assert mkey2.path == ('a',)
    assert mkey2 == mkey.with_key_path_prefix('a')

    mkey3 = mkey2.with_key_path_prefix('b')
    assert mkey3.name == mkey.name
    assert mkey3.path == ('b', 'a')


def test_with_measurement_key_mapping():
    mkey = cirq.MeasurementKey('key')
    mkey2 = cirq.with_measurement_key_mapping(mkey, {'key': 'new_key'})
    assert mkey2.name == 'new_key'

    mkey3 = mkey2.with_key_path_prefix('a')
    mkey3 = cirq.with_measurement_key_mapping(mkey3, {'new_key': 'newer_key'})
    assert mkey3.name == 'newer_key'
    assert mkey3.path == ('a',)
