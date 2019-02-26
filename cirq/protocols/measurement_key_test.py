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

    with pytest.raises(TypeError, match='no _measurement_key_'):
        cirq.measurement_key(NoMethod())

    assert cirq.measurement_key(NoMethod(), None) is None
    assert cirq.measurement_key(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.measurement_key(NoMethod(), 'a') == 'a'


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
