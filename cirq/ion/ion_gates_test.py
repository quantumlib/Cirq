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


def test_MSGate_arguments():
    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.MSGate(np.pi/2),
                                 cirq.XXPowGate(global_shift=-0.5))


def test_MSGate_str():
    assert str(cirq.MSGate(np.pi/2)) == 'MS(np.pi/2)'
    assert str(cirq.MSGate(np.pi)) == 'MS(np.pi/2*2)'


def test_MSGate_matrix():
    s = np.sqrt(0.5)
    assert np.allclose(cirq.unitary(cirq.MSGate(np.pi/4)),
                       np.array([[s, 0, 0, -1j*s],
                                 [0, s, -1j*s, 0],
                                 [0, -1j*s, s, 0],
                                 [-1j*s, 0, 0, s]]))
    assert np.allclose(cirq.unitary(cirq.MSGate(np.pi)),
                       np.diag([-1, -1, -1, -1]))


def test_MSGate_repr():
    assert repr(cirq.MSGate(np.pi/4)) == '(cirq.MSGate(np.pi/2*0.5))'
    cirq.testing.assert_equivalent_repr(cirq.MSGate(np.pi/4))
