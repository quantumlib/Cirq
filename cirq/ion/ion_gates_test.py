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

import cirq


def test_MS_arguments():
    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.MS(np.pi/2),
                                 cirq.XXPowGate(global_shift=-0.5))


def test_MS_str():
    assert str(cirq.MS(np.pi/2)) == 'MS(π/2)'
    assert str(cirq.MS(np.pi)) == 'MS(2.0π/2)'


def test_MS_matrix():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.MS(np.pi/4)),
                       np.array([[s, 0, 0, -1j*s],
                                 [0, s, -1j*s, 0],
                                 [0, -1j*s, s, 0],
                                 [-1j*s, 0, 0, s]]),
                                 atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(cirq.MS(np.pi)),
                               np.diag([-1, -1, -1, -1]),
                               atol=1e-8)


def test_MS_repr():
    assert repr(cirq.MS(np.pi/2)) == 'cirq.MS(np.pi/2)'
    assert repr(cirq.MS(np.pi/4)) == 'cirq.MS(0.5*np.pi/2)'
    cirq.testing.assert_equivalent_repr(cirq.MS(np.pi/4))


def test_MS_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.SWAP(a, b),
        cirq.X(a),
        cirq.Y(a),
        cirq.MS(np.pi).on(a, b))
    cirq.testing.assert_has_diagram(circuit, """
a: ───×───X───Y───MS(π)───
      │           │
b: ───×───────────MS(π)───
""")
