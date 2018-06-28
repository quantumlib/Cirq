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

from cirq.contrib.paulistring import Pauli, PAULI_X, PAULI_Y, PAULI_Z


def test_equals():
    assert Pauli(0) == PAULI_X
    assert Pauli(1) == PAULI_Y
    assert Pauli(2) == PAULI_Z

def test_index_out_of_range():
    with pytest.raises(IndexError):
        Pauli(-1)
    with pytest.raises(IndexError):
        Pauli(-4)
    with pytest.raises(IndexError):
        Pauli(3)
    with pytest.raises(IndexError):
        Pauli(4)
    with pytest.raises(IndexError):
        Pauli(1000000000)

def test_difference():
    assert PAULI_X - PAULI_X == 0
    assert PAULI_X - PAULI_Y == -1
    assert PAULI_X - PAULI_Z == 1
    assert PAULI_Y - PAULI_X == 1
    assert PAULI_Y - PAULI_Y == 0
    assert PAULI_Y - PAULI_Z == -1
    assert PAULI_Z - PAULI_X == -1
    assert PAULI_Z - PAULI_Y == 1
    assert PAULI_Z - PAULI_Z == 0

def test_addition():
    assert PAULI_X + -3 == PAULI_X
    assert PAULI_X + -2 == PAULI_Y
    assert PAULI_X + -1 == PAULI_Z
    assert PAULI_X + 0 == PAULI_X
    assert PAULI_X + 1 == PAULI_Y
    assert PAULI_X + 2 == PAULI_Z
    assert PAULI_X + 3 == PAULI_X
    assert PAULI_X + 4 == PAULI_Y
    assert PAULI_X + 5 == PAULI_Z
    assert PAULI_X + 6 == PAULI_X
    assert PAULI_Y + 0 == PAULI_Y
    assert PAULI_Y + 1 == PAULI_Z
    assert PAULI_Y + 2 == PAULI_X
    assert PAULI_Z + 0 == PAULI_Z
    assert PAULI_Z + 1 == PAULI_X
    assert PAULI_Z + 2 == PAULI_Y

def test_subtraction():
    assert PAULI_X - -3 == PAULI_X
    assert PAULI_X - -2 == PAULI_Z
    assert PAULI_X - -1 == PAULI_Y
    assert PAULI_X - 0 == PAULI_X
    assert PAULI_X - 1 == PAULI_Z
    assert PAULI_X - 2 == PAULI_Y
    assert PAULI_X - 3 == PAULI_X
    assert PAULI_X - 4 == PAULI_Z
    assert PAULI_X - 5 == PAULI_Y
    assert PAULI_X - 6 == PAULI_X
    assert PAULI_Y - 0 == PAULI_Y
    assert PAULI_Y - 1 == PAULI_X
    assert PAULI_Y - 2 == PAULI_Z
    assert PAULI_Z - 0 == PAULI_Z
    assert PAULI_Z - 1 == PAULI_Y
    assert PAULI_Z - 2 == PAULI_X

def test_str():
    assert str(PAULI_X) == 'X'
    assert str(PAULI_Y) == 'Y'
    assert str(PAULI_Z) == 'Z'

def test_repr():
    assert repr(PAULI_X) == 'PAULI_X'
    assert repr(PAULI_Y) == 'PAULI_Y'
    assert repr(PAULI_Z) == 'PAULI_Z'

def test_third():
    assert PAULI_X.third(PAULI_Y) == PAULI_Z
    assert PAULI_Y.third(PAULI_X) == PAULI_Z
    assert PAULI_Y.third(PAULI_Z) == PAULI_X
    assert PAULI_Z.third(PAULI_Y) == PAULI_X
    assert PAULI_Z.third(PAULI_X) == PAULI_Y
    assert PAULI_X.third(PAULI_Z) == PAULI_Y

    assert PAULI_X.third(PAULI_X) == PAULI_X
    assert PAULI_Y.third(PAULI_Y) == PAULI_Y
    assert PAULI_Z.third(PAULI_Z) == PAULI_Z

def test_commutes_with():
    assert PAULI_X.commutes_with(PAULI_X)
    assert not PAULI_X.commutes_with(PAULI_Y)
    assert not PAULI_X.commutes_with(PAULI_Z)
    assert not PAULI_Y.commutes_with(PAULI_X)
    assert PAULI_Y.commutes_with(PAULI_Y)
    assert not PAULI_Y.commutes_with(PAULI_Z)
    assert not PAULI_Z.commutes_with(PAULI_X)
    assert not PAULI_Z.commutes_with(PAULI_Y)
    assert PAULI_Z.commutes_with(PAULI_Z)
