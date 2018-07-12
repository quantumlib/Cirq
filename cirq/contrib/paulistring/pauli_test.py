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

from cirq.contrib.paulistring import Pauli


def test_equals():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(Pauli.X)
    eq.add_equality_group(Pauli.Y)
    eq.add_equality_group(Pauli.Z)


def test_singletons():
    assert Pauli.XYZ[0] == Pauli.X
    assert Pauli.XYZ[1] == Pauli.Y
    assert Pauli.XYZ[2] == Pauli.Z
    assert len(Pauli.XYZ) == 3


def test_difference():
    assert Pauli.X - Pauli.X == 0
    assert Pauli.X - Pauli.Y == -1
    assert Pauli.X - Pauli.Z == 1
    assert Pauli.Y - Pauli.X == 1
    assert Pauli.Y - Pauli.Y == 0
    assert Pauli.Y - Pauli.Z == -1
    assert Pauli.Z - Pauli.X == -1
    assert Pauli.Z - Pauli.Y == 1
    assert Pauli.Z - Pauli.Z == 0


def test_gt():
    assert not Pauli.X > Pauli.X
    assert not Pauli.X > Pauli.Y
    assert Pauli.X > Pauli.Z
    assert Pauli.Y > Pauli.X
    assert not Pauli.Y > Pauli.Y
    assert not Pauli.Y > Pauli.Z
    assert not Pauli.Z > Pauli.X
    assert Pauli.Z > Pauli.Y
    assert not Pauli.Z > Pauli.Z


@cirq.testing.only_test_in_python3
def test_gt_other_type():
    with pytest.raises(TypeError):
        _ = Pauli.X > object()


def test_lt():
    assert not Pauli.X < Pauli.X
    assert Pauli.X < Pauli.Y
    assert not Pauli.X < Pauli.Z
    assert not Pauli.Y < Pauli.X
    assert not Pauli.Y < Pauli.Y
    assert Pauli.Y < Pauli.Z
    assert Pauli.Z < Pauli.X
    assert not Pauli.Z < Pauli.Y
    assert not Pauli.Z < Pauli.Z


@cirq.testing.only_test_in_python3
def test_lt_other_type():
    with pytest.raises(TypeError):
        _ = Pauli.X < object()


def test_addition():
    assert Pauli.X + -3 == Pauli.X
    assert Pauli.X + -2 == Pauli.Y
    assert Pauli.X + -1 == Pauli.Z
    assert Pauli.X + 0 == Pauli.X
    assert Pauli.X + 1 == Pauli.Y
    assert Pauli.X + 2 == Pauli.Z
    assert Pauli.X + 3 == Pauli.X
    assert Pauli.X + 4 == Pauli.Y
    assert Pauli.X + 5 == Pauli.Z
    assert Pauli.X + 6 == Pauli.X
    assert Pauli.Y + 0 == Pauli.Y
    assert Pauli.Y + 1 == Pauli.Z
    assert Pauli.Y + 2 == Pauli.X
    assert Pauli.Z + 0 == Pauli.Z
    assert Pauli.Z + 1 == Pauli.X
    assert Pauli.Z + 2 == Pauli.Y


def test_subtraction():
    assert Pauli.X - -3 == Pauli.X
    assert Pauli.X - -2 == Pauli.Z
    assert Pauli.X - -1 == Pauli.Y
    assert Pauli.X - 0 == Pauli.X
    assert Pauli.X - 1 == Pauli.Z
    assert Pauli.X - 2 == Pauli.Y
    assert Pauli.X - 3 == Pauli.X
    assert Pauli.X - 4 == Pauli.Z
    assert Pauli.X - 5 == Pauli.Y
    assert Pauli.X - 6 == Pauli.X
    assert Pauli.Y - 0 == Pauli.Y
    assert Pauli.Y - 1 == Pauli.X
    assert Pauli.Y - 2 == Pauli.Z
    assert Pauli.Z - 0 == Pauli.Z
    assert Pauli.Z - 1 == Pauli.Y
    assert Pauli.Z - 2 == Pauli.X


def test_str():
    assert str(Pauli.X) == 'X'
    assert str(Pauli.Y) == 'Y'
    assert str(Pauli.Z) == 'Z'


def test_repr():
    assert repr(Pauli.X) == 'Pauli.X'
    assert repr(Pauli.Y) == 'Pauli.Y'
    assert repr(Pauli.Z) == 'Pauli.Z'


def test_third():
    assert Pauli.X.third(Pauli.Y) == Pauli.Z
    assert Pauli.Y.third(Pauli.X) == Pauli.Z
    assert Pauli.Y.third(Pauli.Z) == Pauli.X
    assert Pauli.Z.third(Pauli.Y) == Pauli.X
    assert Pauli.Z.third(Pauli.X) == Pauli.Y
    assert Pauli.X.third(Pauli.Z) == Pauli.Y

    assert Pauli.X.third(Pauli.X) == Pauli.X
    assert Pauli.Y.third(Pauli.Y) == Pauli.Y
    assert Pauli.Z.third(Pauli.Z) == Pauli.Z


def test_commutes_with():
    assert Pauli.X.commutes_with(Pauli.X)
    assert not Pauli.X.commutes_with(Pauli.Y)
    assert not Pauli.X.commutes_with(Pauli.Z)
    assert not Pauli.Y.commutes_with(Pauli.X)
    assert Pauli.Y.commutes_with(Pauli.Y)
    assert not Pauli.Y.commutes_with(Pauli.Z)
    assert not Pauli.Z.commutes_with(Pauli.X)
    assert not Pauli.Z.commutes_with(Pauli.Y)
    assert Pauli.Z.commutes_with(Pauli.Z)
