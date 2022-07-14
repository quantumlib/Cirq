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


class ReturnsStr:
    def _measurement_key_name_(self):
        return 'door locker'


class ReturnsObj:
    def _measurement_key_obj_(self):
        return cirq.MeasurementKey(name='door locker')


@pytest.mark.parametrize('gate', [ReturnsStr(), ReturnsObj()])
def test_measurement_key_name(gate):
    assert isinstance(cirq.measurement_key_name(gate), str)
    assert cirq.measurement_key_name(gate) == 'door locker'
    assert cirq.measurement_key_obj(gate) == cirq.MeasurementKey(name='door locker')

    assert cirq.measurement_key_name(gate, None) == 'door locker'
    assert cirq.measurement_key_name(gate, NotImplemented) == 'door locker'
    assert cirq.measurement_key_name(gate, 'a') == 'door locker'


@pytest.mark.parametrize('gate', [ReturnsStr(), ReturnsObj()])
def test_measurement_key_obj(gate):
    assert isinstance(cirq.measurement_key_obj(gate), cirq.MeasurementKey)
    assert cirq.measurement_key_obj(gate) == cirq.MeasurementKey(name='door locker')
    assert cirq.measurement_key_obj(gate) == 'door locker'

    assert cirq.measurement_key_obj(gate, None) == 'door locker'
    assert cirq.measurement_key_obj(gate, NotImplemented) == 'door locker'
    assert cirq.measurement_key_obj(gate, 'a') == 'door locker'


@pytest.mark.parametrize('key_method', [cirq.measurement_key_name, cirq.measurement_key_obj])
def test_measurement_key_no_method(key_method):
    class NoMethod:
        pass

    with pytest.raises(TypeError, match='no measurement keys'):
        key_method(NoMethod())

    with pytest.raises(ValueError, match='multiple measurement keys'):
        key_method(
            cirq.Circuit(
                cirq.measure(cirq.LineQubit(0), key='a'), cirq.measure(cirq.LineQubit(0), key='b')
            )
        )

    assert key_method(NoMethod(), None) is None
    assert key_method(NoMethod(), NotImplemented) is NotImplemented
    assert key_method(NoMethod(), 'a') == 'a'

    assert key_method(cirq.X, None) is None
    assert key_method(cirq.X(cirq.LineQubit(0)), None) is None


@pytest.mark.parametrize('key_method', [cirq.measurement_key_name, cirq.measurement_key_obj])
def test_measurement_key_not_implemented_default_behavior(key_method):
    class ReturnsNotImplemented:
        def _measurement_key_name_(self):
            return NotImplemented

        def _measurement_key_obj_(self):
            return NotImplemented

    with pytest.raises(TypeError, match='NotImplemented'):
        key_method(ReturnsNotImplemented())

    assert key_method(ReturnsNotImplemented(), None) is None
    assert key_method(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert key_method(ReturnsNotImplemented(), 'a') == 'a'


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


def test_measurement_without_key():
    class MeasurementWithoutKey:
        def _is_measurement_(self):
            return True

    with pytest.raises(TypeError, match='no measurement keys'):
        _ = cirq.measurement_key_name(MeasurementWithoutKey())

    assert cirq.is_measurement(MeasurementWithoutKey())


def test_non_measurement_with_key():
    class NonMeasurementGate(cirq.Gate):
        def _is_measurement_(self):
            return False

        def _decompose_(self, qubits):
            # Decompose should not be called by `is_measurement`
            assert False

        def _measurement_key_name_(self):
            # `measurement_key_name`` should not be called by `is_measurement`
            assert False

        def _measurement_key_names_(self):
            # `measurement_key_names`` should not be called by `is_measurement`
            assert False

        def _measurement_key_obj_(self):
            # `measurement_key_obj`` should not be called by `is_measurement`
            assert False

        def _measurement_key_objs_(self):
            # `measurement_key_objs`` should not be called by `is_measurement`
            assert False

        def num_qubits(self) -> int:
            return 2  # coverage: ignore

    assert not cirq.is_measurement(NonMeasurementGate())


@pytest.mark.parametrize(
    ('key_method', 'keys'),
    [(cirq.measurement_key_names, {'a', 'b'}), (cirq.measurement_key_objs, {'c', 'd'})],
)
def test_measurement_keys(key_method, keys):
    class MeasurementKeysGate(cirq.Gate):
        def _measurement_key_names_(self):
            return frozenset(['a', 'b'])

        def _measurement_key_objs_(self):
            return frozenset([cirq.MeasurementKey('c'), cirq.MeasurementKey('d')])

        def num_qubits(self) -> int:
            return 1

    a, b = cirq.LineQubit.range(2)
    assert key_method(None) == set()
    assert key_method([]) == set()
    assert key_method(cirq.X) == set()
    assert key_method(cirq.X(a)) == set()
    assert key_method(cirq.measure(a, key='out')) == {'out'}
    assert key_method(cirq.Circuit(cirq.measure(a, key='a'), cirq.measure(b, key='2'))) == {
        'a',
        '2',
    }
    assert key_method(MeasurementKeysGate()) == keys
    assert key_method(MeasurementKeysGate().on(a)) == keys


def test_measurement_key_mapping():
    class MultiKeyGate:
        def __init__(self, keys):
            self._keys = frozenset(keys)

        def _measurement_key_names_(self):
            return self._keys

        def _with_measurement_key_mapping_(self, key_map):
            if not all(key in key_map for key in self._keys):
                raise ValueError('missing keys')
            return MultiKeyGate([key_map[key] for key in self._keys])

    assert cirq.measurement_key_names(MultiKeyGate([])) == set()
    assert cirq.measurement_key_names(MultiKeyGate(['a'])) == {'a'}

    mkg_ab = MultiKeyGate(['a', 'b'])
    assert cirq.measurement_key_names(mkg_ab) == {'a', 'b'}

    mkg_cd = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c', 'b': 'd'})
    assert cirq.measurement_key_names(mkg_cd) == {'c', 'd'}

    mkg_ac = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'a', 'b': 'c'})
    assert cirq.measurement_key_names(mkg_ac) == {'a', 'c'}

    mkg_ba = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'b', 'b': 'a'})
    assert cirq.measurement_key_names(mkg_ba) == {'a', 'b'}

    with pytest.raises(ValueError):
        cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c'})

    assert cirq.with_measurement_key_mapping(cirq.X, {'a': 'c'}) is NotImplemented

    mkg_cdx = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c', 'b': 'd', 'x': 'y'})
    assert cirq.measurement_key_names(mkg_cdx) == {'c', 'd'}


def test_measurement_key_path():
    class MultiKeyGate:
        def __init__(self, keys):
            self._keys = frozenset(cirq.MeasurementKey.parse_serialized(key) for key in keys)

        def _measurement_key_names_(self):
            return frozenset(str(key) for key in self._keys)

        def _with_key_path_(self, path):
            return MultiKeyGate([str(key._with_key_path_(path)) for key in self._keys])

    assert cirq.measurement_key_names(MultiKeyGate([])) == set()
    assert cirq.measurement_key_names(MultiKeyGate(['a'])) == {'a'}

    mkg_ab = MultiKeyGate(['a', 'b'])
    assert cirq.measurement_key_names(mkg_ab) == {'a', 'b'}

    mkg_cd = cirq.with_key_path(mkg_ab, ('c', 'd'))
    assert cirq.measurement_key_names(mkg_cd) == {'c:d:a', 'c:d:b'}

    assert cirq.with_key_path(cirq.X, ('c', 'd')) is NotImplemented
