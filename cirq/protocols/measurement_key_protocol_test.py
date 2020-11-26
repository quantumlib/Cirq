# Copyright 2019 The Cirq Developers
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


def test_measurement_key():

    class ReturnsStr():

        def _measurement_key_(self):
            return 'door locker'

    assert cirq.measurement_key(ReturnsStr()) == 'door locker'

    assert cirq.measurement_key(ReturnsStr(), None) == 'door locker'
    assert cirq.measurement_key(ReturnsStr(), NotImplemented) == 'door locker'
    assert cirq.measurement_key(ReturnsStr(), 'a') == 'door locker'


def test_measurement_key_no_method():

    class NoMethod():
        pass

    with pytest.raises(TypeError, match='no measurement keys'):
        cirq.measurement_key(NoMethod())

    with pytest.raises(ValueError, match='multiple measurement keys'):
        cirq.measurement_key(
            cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='a'),
                         cirq.measure(cirq.LineQubit(0), key='b')))

    assert cirq.measurement_key(NoMethod(), None) is None
    assert cirq.measurement_key(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.measurement_key(NoMethod(), 'a') == 'a'

    assert cirq.measurement_key(cirq.X, None) is None
    assert cirq.measurement_key(cirq.X(cirq.LineQubit(0)), None) is None


def test_measurement_key_not_implemented():

    class ReturnsNotImplemented():

        def _measurement_key_(self):
            return NotImplemented

    with pytest.raises(TypeError, match='NotImplemented'):
        cirq.measurement_key(ReturnsNotImplemented())

    assert cirq.measurement_key(ReturnsNotImplemented(), None) is None
    assert cirq.measurement_key(ReturnsNotImplemented(),
                                NotImplemented) is NotImplemented
    assert cirq.measurement_key(ReturnsNotImplemented(), 'a') == 'a'


def test_is_measurement():
    q = cirq.NamedQubit('q')
    assert cirq.is_measurement(cirq.measure(q))
    assert cirq.is_measurement(cirq.MeasurementGate(num_qubits=1, key='b'))

    assert not cirq.is_measurement(cirq.X(q))
    assert not cirq.is_measurement(cirq.X)
    assert not cirq.is_measurement(cirq.bit_flip(1))

    class NotImplementedOperation(cirq.Operation):

        def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
            raise NotImplementedError()

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

    assert not cirq.is_measurement(NotImplementedOperation())


def test_measurement_keys():

    class Composite(cirq.Gate):

        def _decompose_(self, qubits):
            yield cirq.measure(qubits[0], key='inner1')
            yield cirq.measure(qubits[1], key='inner2')
            yield cirq.reset(qubits[0])

        def num_qubits(self) -> int:
            return 2

    class MeasurementKeysGate(cirq.Gate):

        def _measurement_keys_(self):
            return ['a', 'b']

        def num_qubits(self) -> int:
            return 1

    a, b = cirq.LineQubit.range(2)
    assert cirq.measurement_keys(Composite()) == {'inner1', 'inner2'}
    assert cirq.measurement_keys(Composite().on(a, b)) == {'inner1', 'inner2'}
    assert cirq.measurement_keys(Composite(), allow_decompose=False) == set()
    assert cirq.measurement_keys(Composite().on(a, b),
                                 allow_decompose=False) == set()

    assert cirq.measurement_keys(None) == set()
    assert cirq.measurement_keys([]) == set()
    assert cirq.measurement_keys(cirq.X) == set()
    assert cirq.measurement_keys(cirq.X(a)) == set()
    assert cirq.measurement_keys(None, allow_decompose=False) == set()
    assert cirq.measurement_keys([], allow_decompose=False) == set()
    assert cirq.measurement_keys(cirq.X, allow_decompose=False) == set()
    assert cirq.measurement_keys(cirq.measure(a, key='out')) == {'out'}
    assert cirq.measurement_keys(cirq.measure(a, key='out'),
                                 allow_decompose=False) == {'out'}

    assert cirq.measurement_keys(
        cirq.Circuit(cirq.measure(a, key='a'),
                     cirq.measure(b, key='2'))) == {'a', '2'}
    assert cirq.measurement_keys(MeasurementKeysGate()) == {'a', 'b'}
    assert cirq.measurement_keys(MeasurementKeysGate().on(a)) == {'a', 'b'}


def test_measurement_key_mapping():

    class MultiKeyGate:

        def __init__(self, keys):
            self._keys = set(keys)

        def _measurement_keys_(self):
            return self._keys

        def _with_measurement_key_mapping_(self, key_map):
            if not all(key in key_map for key in self._keys):
                raise ValueError('missing keys')
            return MultiKeyGate([key_map[key] for key in self._keys])

    assert cirq.measurement_keys(MultiKeyGate([])) == set()
    assert cirq.measurement_keys(MultiKeyGate(['a'])) == {'a'}

    mkg_ab = MultiKeyGate(['a', 'b'])
    assert cirq.measurement_keys(mkg_ab) == {'a', 'b'}

    mkg_cd = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c', 'b': 'd'})
    assert cirq.measurement_keys(mkg_cd) == {'c', 'd'}

    mkg_ac = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'a', 'b': 'c'})
    assert cirq.measurement_keys(mkg_ac) == {'a', 'c'}

    mkg_ba = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'b', 'b': 'a'})
    assert cirq.measurement_keys(mkg_ba) == {'a', 'b'}

    with pytest.raises(ValueError):
        cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c'})

    assert cirq.with_measurement_key_mapping(cirq.X,
                                             {'a': 'c'}) is NotImplemented

    mkg_cdx = cirq.with_measurement_key_mapping(mkg_ab, {
        'a': 'c',
        'b': 'd',
        'x': 'y'
    })
    assert cirq.measurement_keys(mkg_cdx) == {'c', 'd'}
