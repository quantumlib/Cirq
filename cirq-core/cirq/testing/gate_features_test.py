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


def test_two_qubit_gate_is_abstract_can_implement():
    class Included(cirq.testing.TwoQubitGate):
        def matrix(self):
            pass

    assert isinstance(Included(), cirq.testing.TwoQubitGate)


def test_two_qubit_gate_validate_pass():
    class Example(cirq.testing.TwoQubitGate):
        def matrix(self):
            pass

    g = Example()
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')

    assert g.num_qubits() == 2
    g.validate_args([q1, q2])
    g.validate_args([q2, q3])
    g.validate_args([q3, q2])


def test_two_qubit_gate_validate_wrong_number():
    class Example(cirq.testing.TwoQubitGate):
        def matrix(self):
            pass

    g = Example()
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')

    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2, q3])


def test_three_qubit_gate_validate():
    class Example(cirq.testing.ThreeQubitGate):
        def matrix(self):
            pass

    g = Example()
    a, b, c, d = cirq.LineQubit.range(4)

    assert g.num_qubits() == 3

    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])
