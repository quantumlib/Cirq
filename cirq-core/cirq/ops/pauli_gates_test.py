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

import itertools

import numpy as np
import pytest
import cirq


def test_equals():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.X, cirq.ops.pauli_gates.X, cirq.XPowGate())
    eq.add_equality_group(cirq.Y, cirq.ops.pauli_gates.Y, cirq.YPowGate())
    eq.add_equality_group(cirq.Z, cirq.ops.pauli_gates.Z, cirq.ZPowGate())


def test_phased_pauli_product():
    assert cirq.X.phased_pauli_product(cirq.I) == (1, cirq.X)
    assert cirq.X.phased_pauli_product(cirq.X) == (1, cirq.I)
    assert cirq.X.phased_pauli_product(cirq.Y) == (1j, cirq.Z)
    assert cirq.X.phased_pauli_product(cirq.Z) == (-1j, cirq.Y)

    assert cirq.Y.phased_pauli_product(cirq.I) == (1, cirq.Y)
    assert cirq.Y.phased_pauli_product(cirq.X) == (-1j, cirq.Z)
    assert cirq.Y.phased_pauli_product(cirq.Y) == (1, cirq.I)
    assert cirq.Y.phased_pauli_product(cirq.Z) == (1j, cirq.X)

    assert cirq.Z.phased_pauli_product(cirq.I) == (1, cirq.Z)
    assert cirq.Z.phased_pauli_product(cirq.X) == (1j, cirq.Y)
    assert cirq.Z.phased_pauli_product(cirq.Y) == (-1j, cirq.X)
    assert cirq.Z.phased_pauli_product(cirq.Z) == (1, cirq.I)


def test_isinstance():
    assert isinstance(cirq.X, cirq.XPowGate)
    assert isinstance(cirq.Y, cirq.YPowGate)
    assert isinstance(cirq.Z, cirq.ZPowGate)

    assert not isinstance(cirq.X, cirq.YPowGate)
    assert not isinstance(cirq.X, cirq.ZPowGate)

    assert not isinstance(cirq.Y, cirq.XPowGate)
    assert not isinstance(cirq.Y, cirq.ZPowGate)

    assert not isinstance(cirq.Z, cirq.XPowGate)
    assert not isinstance(cirq.Z, cirq.YPowGate)


def test_by_index():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.X, *[cirq.Pauli.by_index(i) for i in (-3, 0, 3, 6)])
    eq.add_equality_group(cirq.Y, *[cirq.Pauli.by_index(i) for i in (-2, 1, 4, 7)])
    eq.add_equality_group(cirq.Z, *[cirq.Pauli.by_index(i) for i in (-1, 2, 5, 8)])


def test_relative_index():
    assert cirq.X.relative_index(cirq.X) == 0
    assert cirq.X.relative_index(cirq.Y) == -1
    assert cirq.X.relative_index(cirq.Z) == 1
    assert cirq.Y.relative_index(cirq.X) == 1
    assert cirq.Y.relative_index(cirq.Y) == 0
    assert cirq.Y.relative_index(cirq.Z) == -1
    assert cirq.Z.relative_index(cirq.X) == -1
    assert cirq.Z.relative_index(cirq.Y) == 1
    assert cirq.Z.relative_index(cirq.Z) == 0


def test_by_relative_index():
    assert cirq.Pauli.by_relative_index(cirq.X, -1) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.X, 0) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.X, 1) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.X, 2) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.X, 3) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, -1) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, 0) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Y, 1) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.Y, 2) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, 3) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, -1) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, 0) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.Z, 1) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Z, 2) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, 3) == cirq.Z


def test_too_many_qubits():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='single qubit'):
        _ = cirq.X.on(a, b)

    x = cirq.X(a)
    with pytest.raises(ValueError, match=r'len\(new_qubits\)'):
        _ = x.with_qubits(a, b)


def test_relative_index_consistency():
    for pauli_1 in (cirq.X, cirq.Y, cirq.Z):
        for pauli_2 in (cirq.X, cirq.Y, cirq.Z):
            shift = pauli_2.relative_index(pauli_1)
            assert cirq.Pauli.by_relative_index(pauli_1, shift) == pauli_2


def test_gt():
    assert not cirq.X > cirq.X
    assert not cirq.X > cirq.Y
    assert cirq.X > cirq.Z
    assert cirq.Y > cirq.X
    assert not cirq.Y > cirq.Y
    assert not cirq.Y > cirq.Z
    assert not cirq.Z > cirq.X
    assert cirq.Z > cirq.Y
    assert not cirq.Z > cirq.Z


def test_gt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.X > object()


def test_lt():
    assert not cirq.X < cirq.X
    assert cirq.X < cirq.Y
    assert not cirq.X < cirq.Z
    assert not cirq.Y < cirq.X
    assert not cirq.Y < cirq.Y
    assert cirq.Y < cirq.Z
    assert cirq.Z < cirq.X
    assert not cirq.Z < cirq.Y
    assert not cirq.Z < cirq.Z


def test_lt_other_type():
    with pytest.raises(TypeError):
        _ = cirq.X < object()


def test_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.Y) == 'Y'
    assert str(cirq.Z) == 'Z'


def test_repr():
    assert repr(cirq.X) == 'cirq.X'
    assert repr(cirq.Y) == 'cirq.Y'
    assert repr(cirq.Z) == 'cirq.Z'


def test_third():
    assert cirq.X.third(cirq.Y) == cirq.Z
    assert cirq.Y.third(cirq.X) == cirq.Z
    assert cirq.Y.third(cirq.Z) == cirq.X
    assert cirq.Z.third(cirq.Y) == cirq.X
    assert cirq.Z.third(cirq.X) == cirq.Y
    assert cirq.X.third(cirq.Z) == cirq.Y

    assert cirq.X.third(cirq.X) == cirq.X
    assert cirq.Y.third(cirq.Y) == cirq.Y
    assert cirq.Z.third(cirq.Z) == cirq.Z


def test_commutes():
    for A, B in itertools.product([cirq.X, cirq.Y, cirq.Z], repeat=2):
        assert cirq.commutes(A, B) == (A == B)
    with pytest.raises(TypeError):
        assert cirq.commutes(cirq.X, 'X')
    assert cirq.commutes(cirq.X, 'X', default='default') == 'default'
    assert cirq.commutes(cirq.Z, cirq.read_json(json_text=cirq.to_json(cirq.Z)))


def test_unitary():
    np.testing.assert_equal(cirq.unitary(cirq.X), cirq.unitary(cirq.X))
    np.testing.assert_equal(cirq.unitary(cirq.Y), cirq.unitary(cirq.Y))
    np.testing.assert_equal(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))


def test_apply_unitary():
    cirq.testing.assert_has_consistent_apply_unitary(cirq.X)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Y)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Z)


def test_identity_multiplication():
    a, b, c = cirq.LineQubit.range(3)
    assert cirq.X(a) * cirq.I(a) == cirq.X(a)
    assert cirq.X(a) * cirq.I(b) == cirq.X(a)
    assert cirq.X(a) * cirq.Y(b) * cirq.I(c) == cirq.X(a) * cirq.Y(b)
    assert cirq.I(c) * cirq.X(a) * cirq.Y(b) == cirq.X(a) * cirq.Y(b)
    with pytest.raises(TypeError):
        _ = cirq.H(c) * cirq.X(a) * cirq.Y(b)
    with pytest.raises(TypeError):
        _ = cirq.X(a) * cirq.Y(b) * cirq.H(c)
    with pytest.raises(TypeError):
        _ = cirq.I(a) * str(cirq.Y(b))


def test_powers():
    assert isinstance(cirq.X, cirq.Pauli)
    assert isinstance(cirq.Y, cirq.Pauli)
    assert isinstance(cirq.Z, cirq.Pauli)
    assert not isinstance(cirq.X ** -0.5, cirq.Pauli)
    assert not isinstance(cirq.Y ** 0.2, cirq.Pauli)
    assert not isinstance(cirq.Z ** 0.5, cirq.Pauli)
    assert isinstance(cirq.X ** -0.5, cirq.XPowGate)
    assert isinstance(cirq.Y ** 0.2, cirq.YPowGate)
    assert isinstance(cirq.Z ** 0.5, cirq.ZPowGate)

    assert isinstance(cirq.X ** 1, cirq.Pauli)
    assert isinstance(cirq.Y ** 1, cirq.Pauli)
    assert isinstance(cirq.Z ** 1, cirq.Pauli)
