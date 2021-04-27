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
    nested_key = cirq.MeasurementKey('nested:key')
    assert nested_key.name == 'nested:key'


def test_eq_and_hash():
    class SomeRandomClass:
        def __init__(self, some_str):
            self.some_str = some_str

        def __str__(self):
            return self.some_str  # coverage: ignore

    mkey = cirq.MeasurementKey('key')
    assert mkey == 'key'
    assert hash(mkey) == hash('key')
    nested_key = cirq.MeasurementKey('nested:key')
    assert nested_key == 'nested:key'
    non_str_or_measurement_key = SomeRandomClass('key')
    assert mkey != non_str_or_measurement_key


@pytest.mark.parametrize('key_string', ['key', 'nested:key'])
def test_str(key_string):
    mkey = cirq.MeasurementKey(key_string)
    assert str(mkey) == key_string
    assert str(mkey) == mkey


@pytest.mark.parametrize('key_string', ['key', 'nested:key'])
def test_repr(key_string):
    mkey = cirq.MeasurementKey(key_string)
    assert repr(mkey) == f'cirq.MeasurementKey(name={key_string})'


def test_json_dict():
    mkey = cirq.MeasurementKey('key')
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'key',
    }
    mkey = cirq.MeasurementKey('nested:key')
    assert mkey._json_dict_() == {
        'cirq_type': 'MeasurementKey',
        'name': 'nested:key',
    }
