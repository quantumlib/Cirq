# Copyright 2018 The Cirq Developers
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


class NoMethod:
    pass


class ReturnsNotImplemented:
    def _qasm_(self):
        return NotImplemented


class ReturnsText:
    def _qasm_(self):
        return 'text'


class ExpectsArgs:
    def _qasm_(self, args):
        return 'text'


class ExpectsArgsQubits:
    def _qasm_(self, args, qubits):
        return 'text'


def test_qasm():
    assert cirq.qasm(NoMethod(), default=None) is None
    assert cirq.qasm(NoMethod(), default=5) == 5
    assert cirq.qasm(ReturnsText()) == 'text'

    with pytest.raises(TypeError, match='no _qasm_ method'):
        _ = cirq.qasm(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.qasm(ReturnsNotImplemented())

    assert cirq.qasm(ExpectsArgs(), args=cirq.QasmArgs()) == 'text'
    assert cirq.qasm(ExpectsArgsQubits(), args=cirq.QasmArgs(), qubits=()) == 'text'
