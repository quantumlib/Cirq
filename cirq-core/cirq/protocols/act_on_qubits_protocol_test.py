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
from cirq.protocols.act_on_protocol_test import DummyActOnArgs

qubits = cirq.LineQubit.range(1)


def test_act_on_fallback_succeeds():
    args = DummyActOnArgs(fallback_result=True)
    cirq.act_on_qubits(cirq.X, args, qubits)


def test_act_on_fallback_fails():
    args = DummyActOnArgs(fallback_result=NotImplemented)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on_qubits(cirq.X, args, qubits)


def test_act_on_fallback_errors():
    args = DummyActOnArgs(fallback_result=False)
    with pytest.raises(ValueError, match='_act_on_fallback_ must return True or NotImplemented'):
        cirq.act_on_qubits(cirq.X, args, qubits)


def test_act_on_errors():
    class Op:
        def _act_on_(self, args, qubits):
            return False

    args = DummyActOnArgs(fallback_result=True)
    with pytest.raises(ValueError, match='_act_on_ must return True or NotImplemented'):
        cirq.act_on_qubits(Op(), args, qubits)
