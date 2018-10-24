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
import ion

def test_CxNOTGate_init():
    assert ion.CxNOTGate(exponent=8) == ion.CxNOTGate(exponent=0)
    assert ion.CxNOTGate(exponent=9) == ion.CxNOTGate()

def test_CxNOTGate_str():
    assert str(ion.CxNOTGate()) == 'CxNOT'
    assert str(ion.CxNOTGate(exponent=5)) == 'CxNOT**5'

def test_CxNOTGate_matrix():
    s = np.sqrt(0.5)
    assert np.allclose(cirq.unitary(ion.CxNOTGate()),
                       np.array([[s, 0, 0, -1j*s],
                                 [0, s, -1j*s, 0],
                                 [0, -1j*s, s, 0],
                                 [-1j*s, 0, 0, s]]))
    assert np.allclose(cirq.unitary(ion.CxNOTGate(exponent=4)),
                       np.diag([-1, -1, -1, -1]))
    assert np.allclose(cirq.unitary(ion.CxNOTGate()**8),
                       np.diag([1, 1, 1, 1]))

def test_CxNOTGate_repr():
    assert repr(ion.CxNOTGate()) == 'ion.CxNOT'
    assert repr(ion.CxNOTGate(exponent=0.5)) == '(ion.CxNOT**0.5)'