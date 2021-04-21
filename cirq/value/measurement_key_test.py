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
    mkey = cirq.MeasurementKey('')
    assert mkey.name == ''


def test_nested_key():
    with pytest.raises(ValueError, match='not allowed .* set the allow_nested_key'):
        _ = cirq.MeasurementKey('nested:key')
    with pytest.raises(ValueError, match='not allowed .* set the allow_nested_key'):
        _ = cirq.MeasurementKey('nested:key', allow_nested_key=False)
    nested_key = cirq.MeasurementKey('nested:key', allow_nested_key=True)
    assert nested_key.name == 'nested:key'
    assert nested_key.allow_nested_key


def test_eq_and_hash():
    mkey = cirq.MeasurementKey('key')
    assert mkey == 'key'
    assert hash(mkey) == hash('key')
    mkey2 = cirq.MeasurementKey('key', allow_nested_key=False)
    assert mkey2 == 'key'
    assert mkey2 == mkey
    assert hash(mkey2) == hash(mkey)
    mkey3 = cirq.MeasurementKey('key', allow_nested_key=True)
    assert mkey3 == 'key'
    assert mkey3 == mkey
    assert hash(mkey3) == hash(mkey)
    nested_key = cirq.MeasurementKey('nested:key', allow_nested_key=True)
    assert nested_key == 'nested:key'


@pytest.mark.parametrize(
    'key_string, allow_nested_key',
    [
        ('key', None),
        ('key', False),
        ('key', True),
        ('nested:key', True),
    ],
)
def test_str(key_string, allow_nested_key):
    if allow_nested_key is not None:
        mkey = cirq.MeasurementKey(key_string, allow_nested_key)
    else:
        mkey = cirq.MeasurementKey(key_string)
    assert str(mkey) == key_string
    assert str(mkey) == mkey


@pytest.mark.parametrize(
    'key_string, allow_nested_key',
    [
        ('key', None),
        ('key', False),
        ('key', True),
        ('nested:key', True),
    ],
)
def test_repr(key_string, allow_nested_key):
    if allow_nested_key is not None:
        mkey = cirq.MeasurementKey(key_string, allow_nested_key)
    else:
        mkey = cirq.MeasurementKey(key_string)
    assert repr(mkey) == f'cirq.MeasurementKey(name={key_string})'


def test_json_dict():
    mkey = cirq.MeasurementKey('key')
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'key',
        'allow_nested_key': False,
    }
    mkey = cirq.MeasurementKey('nested:key', True)
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'nested:key',
        'allow_nested_key': True,
    }
