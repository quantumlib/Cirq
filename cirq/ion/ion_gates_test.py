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

import numpy as np
import pytest

import cirq


def test_MSGate_init():
    assert cirq.MSGate(exponent=2).exponent == 2


def test_MSGate_arguments():
    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.MSGate(exponent=2),
                                 cirq.MSGate() ** 2)
    eq_tester.add_equality_group(cirq.MSGate(exponent=4),
                                 cirq.MSGate(rads=np.pi))


def test_MSGate_str():
    assert str(cirq.MSGate()) == 'MS'
    assert str(cirq.MSGate(exponent=3)) == 'MS**3.0'


def test_MSGate_matrix():
    s = np.sqrt(0.5)
    assert np.allclose(cirq.unitary(cirq.MSGate()),
                       np.array([[s, 0, 0, -1j*s],
                                 [0, s, -1j*s, 0],
                                 [0, -1j*s, s, 0],
                                 [-1j*s, 0, 0, s]]))
    assert np.allclose(cirq.unitary(cirq.MSGate(exponent=4)),
                       np.diag([-1, -1, -1, -1]))


def test_MSGate_repr():
    assert repr(cirq.MSGate()) == 'cirq.MS'
    assert repr(cirq.MSGate(exponent=0.5)) == '(cirq.MS**0.5)'
    cirq.testing.assert_equivalent_repr(cirq.MSGate())
    cirq.testing.assert_equivalent_repr(cirq.MSGate() ** 0.1)
    cirq.testing.assert_equivalent_repr(cirq.MS)
    cirq.testing.assert_equivalent_repr(cirq.MS ** 0.1)


def test_MSGate_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.SWAP(a, b),
        cirq.X(a),
        cirq.Y(a),
        cirq.MS(a, b))


    cirq.testing.assert_has_diagram(circuit, """
a: ───×───X───Y───MS───
      │           │
b: ───×───────────MS───
""")

    cirq.testing.assert_has_diagram(circuit, """
a: ---swap---X---Y---MS---
      |              |
b: ---swap-----------MS---
""", use_unicode_characters=False)
