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


def test_equals():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.Pauli.X, cirq.Pauli(_index=0, _name='X'))
    eq.add_equality_group(cirq.Pauli.Y, cirq.Pauli(_index=1, _name='Y'))
    eq.add_equality_group(cirq.Pauli.Z, cirq.Pauli(_index=2, _name='Z'))


def test_singletons():
    assert cirq.Pauli.XYZ[0] == cirq.Pauli.X
    assert cirq.Pauli.XYZ[1] == cirq.Pauli.Y
    assert cirq.Pauli.XYZ[2] == cirq.Pauli.Z
    assert len(cirq.Pauli.XYZ) == 3


def test_difference():
    assert cirq.Pauli.X - cirq.Pauli.X == 0
    assert cirq.Pauli.X - cirq.Pauli.Y == -1
    assert cirq.Pauli.X - cirq.Pauli.Z == 1
    assert cirq.Pauli.Y - cirq.Pauli.X == 1
    assert cirq.Pauli.Y - cirq.Pauli.Y == 0
    assert cirq.Pauli.Y - cirq.Pauli.Z == -1
    assert cirq.Pauli.Z - cirq.Pauli.X == -1
    assert cirq.Pauli.Z - cirq.Pauli.Y == 1
    assert cirq.Pauli.Z - cirq.Pauli.Z == 0


def test_gt():
    assert not cirq.Pauli.X > cirq.Pauli.X
    assert not cirq.Pauli.X > cirq.Pauli.Y
    assert cirq.Pauli.X > cirq.Pauli.Z
    assert cirq.Pauli.Y > cirq.Pauli.X
    assert not cirq.Pauli.Y > cirq.Pauli.Y
    assert not cirq.Pauli.Y > cirq.Pauli.Z
    assert not cirq.Pauli.Z > cirq.Pauli.X
    assert cirq.Pauli.Z > cirq.Pauli.Y
    assert not cirq.Pauli.Z > cirq.Pauli.Z


@cirq.testing.only_test_in_python3
def test_gt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.Pauli.X > object()


def test_lt():
    assert not cirq.Pauli.X < cirq.Pauli.X
    assert cirq.Pauli.X < cirq.Pauli.Y
    assert not cirq.Pauli.X < cirq.Pauli.Z
    assert not cirq.Pauli.Y < cirq.Pauli.X
    assert not cirq.Pauli.Y < cirq.Pauli.Y
    assert cirq.Pauli.Y < cirq.Pauli.Z
    assert cirq.Pauli.Z < cirq.Pauli.X
    assert not cirq.Pauli.Z < cirq.Pauli.Y
    assert not cirq.Pauli.Z < cirq.Pauli.Z


@cirq.testing.only_test_in_python3
def test_lt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.Pauli.X < object()


def test_addition():
    assert cirq.Pauli.X + -3 == cirq.Pauli.X
    assert cirq.Pauli.X + -2 == cirq.Pauli.Y
    assert cirq.Pauli.X + -1 == cirq.Pauli.Z
    assert cirq.Pauli.X + 0 == cirq.Pauli.X
    assert cirq.Pauli.X + 1 == cirq.Pauli.Y
    assert cirq.Pauli.X + 2 == cirq.Pauli.Z
    assert cirq.Pauli.X + 3 == cirq.Pauli.X
    assert cirq.Pauli.X + 4 == cirq.Pauli.Y
    assert cirq.Pauli.X + 5 == cirq.Pauli.Z
    assert cirq.Pauli.X + 6 == cirq.Pauli.X
    assert cirq.Pauli.Y + 0 == cirq.Pauli.Y
    assert cirq.Pauli.Y + 1 == cirq.Pauli.Z
    assert cirq.Pauli.Y + 2 == cirq.Pauli.X
    assert cirq.Pauli.Z + 0 == cirq.Pauli.Z
    assert cirq.Pauli.Z + 1 == cirq.Pauli.X
    assert cirq.Pauli.Z + 2 == cirq.Pauli.Y


def test_subtraction():
    assert cirq.Pauli.X - -3 == cirq.Pauli.X
    assert cirq.Pauli.X - -2 == cirq.Pauli.Z
    assert cirq.Pauli.X - -1 == cirq.Pauli.Y
    assert cirq.Pauli.X - 0 == cirq.Pauli.X
    assert cirq.Pauli.X - 1 == cirq.Pauli.Z
    assert cirq.Pauli.X - 2 == cirq.Pauli.Y
    assert cirq.Pauli.X - 3 == cirq.Pauli.X
    assert cirq.Pauli.X - 4 == cirq.Pauli.Z
    assert cirq.Pauli.X - 5 == cirq.Pauli.Y
    assert cirq.Pauli.X - 6 == cirq.Pauli.X
    assert cirq.Pauli.Y - 0 == cirq.Pauli.Y
    assert cirq.Pauli.Y - 1 == cirq.Pauli.X
    assert cirq.Pauli.Y - 2 == cirq.Pauli.Z
    assert cirq.Pauli.Z - 0 == cirq.Pauli.Z
    assert cirq.Pauli.Z - 1 == cirq.Pauli.Y
    assert cirq.Pauli.Z - 2 == cirq.Pauli.X


def test_str():
    assert str(cirq.Pauli.X) == 'X'
    assert str(cirq.Pauli.Y) == 'Y'
    assert str(cirq.Pauli.Z) == 'Z'


def test_repr():
    assert repr(cirq.Pauli.X) == 'cirq.Pauli.X'
    assert repr(cirq.Pauli.Y) == 'cirq.Pauli.Y'
    assert repr(cirq.Pauli.Z) == 'cirq.Pauli.Z'


def test_third():
    assert cirq.Pauli.X.third(cirq.Pauli.Y) == cirq.Pauli.Z
    assert cirq.Pauli.Y.third(cirq.Pauli.X) == cirq.Pauli.Z
    assert cirq.Pauli.Y.third(cirq.Pauli.Z) == cirq.Pauli.X
    assert cirq.Pauli.Z.third(cirq.Pauli.Y) == cirq.Pauli.X
    assert cirq.Pauli.Z.third(cirq.Pauli.X) == cirq.Pauli.Y
    assert cirq.Pauli.X.third(cirq.Pauli.Z) == cirq.Pauli.Y

    assert cirq.Pauli.X.third(cirq.Pauli.X) == cirq.Pauli.X
    assert cirq.Pauli.Y.third(cirq.Pauli.Y) == cirq.Pauli.Y
    assert cirq.Pauli.Z.third(cirq.Pauli.Z) == cirq.Pauli.Z


def test_commutes_with():
    assert cirq.Pauli.X.commutes_with(cirq.Pauli.X)
    assert not cirq.Pauli.X.commutes_with(cirq.Pauli.Y)
    assert not cirq.Pauli.X.commutes_with(cirq.Pauli.Z)
    assert not cirq.Pauli.Y.commutes_with(cirq.Pauli.X)
    assert cirq.Pauli.Y.commutes_with(cirq.Pauli.Y)
    assert not cirq.Pauli.Y.commutes_with(cirq.Pauli.Z)
    assert not cirq.Pauli.Z.commutes_with(cirq.Pauli.X)
    assert not cirq.Pauli.Z.commutes_with(cirq.Pauli.Y)
    assert cirq.Pauli.Z.commutes_with(cirq.Pauli.Z)


def test_unitary():
    np.testing.assert_equal(cirq.unitary(cirq.Pauli.X), cirq.unitary(cirq.X))
    np.testing.assert_equal(cirq.unitary(cirq.Pauli.Y), cirq.unitary(cirq.Y))
    np.testing.assert_equal(cirq.unitary(cirq.Pauli.Z), cirq.unitary(cirq.Z))


def test_apply_unitary():
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Pauli.X)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Pauli.Y)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Pauli.Z)
