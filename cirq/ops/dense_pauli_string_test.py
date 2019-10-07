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

from typing import Union, List, Optional, Sequence, TYPE_CHECKING, Any, Tuple, TypeVar, Type
import abc

import numpy as np
import pytest

import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase


def test_init():
    mask = np.array([0, 3, 1, 2], dtype=np.uint8)
    p = cirq.DensePauliString(coefficient=2, pauli_mask=mask)
    m = cirq.MutableDensePauliString(coefficient=3, pauli_mask=mask)
    assert p.coefficient == 2
    assert m.coefficient == 3
    np.testing.assert_allclose(p.pauli_mask, [0, 3, 1, 2])
    np.testing.assert_allclose(m.pauli_mask, [0, 3, 1, 2])

    # The non-mutable initializer makes a copy.
    assert m.pauli_mask is mask
    assert p.pauli_mask is not mask
    mask[:] = 0
    assert m.pauli_mask[2] == 0
    assert p.pauli_mask[2] == 1

    # Copies and converts non-uint8 arrays.
    p2 = cirq.DensePauliString(coefficient=2, pauli_mask=[1, 2, 3])
    m2 = cirq.DensePauliString(coefficient=2, pauli_mask=[1, 2, 3])
    assert p2.pauli_mask.dtype == m2.pauli_mask.dtype == np.uint8
    assert list(p2.pauli_mask) == list(m2.pauli_mask) == [1, 2, 3]


def test_immutable_eq():
    eq = cirq.testing.EqualsTester()

    # Immutables
    eq.make_equality_group(lambda:
        cirq.DensePauliString(coefficient=2,
                              pauli_mask=[1]))
    eq.add_equality_group(lambda:
        cirq.DensePauliString(coefficient=3,
                              pauli_mask=[1]))
    eq.make_equality_group(lambda:
        cirq.DensePauliString(coefficient=2,
                              pauli_mask=[]))
    eq.add_equality_group(lambda:
        cirq.DensePauliString(coefficient=2,
                              pauli_mask=[0]))
    eq.make_equality_group(lambda:
        cirq.DensePauliString(coefficient=2,
                              pauli_mask=[2]))

    # Mutables
    eq.make_equality_group(lambda:
        cirq.MutableDensePauliString(coefficient=2,
                              pauli_mask=[1]))
    eq.add_equality_group(lambda:
        cirq.MutableDensePauliString(coefficient=3,
                              pauli_mask=[1]))
    eq.make_equality_group(lambda:
        cirq.MutableDensePauliString(coefficient=2,
                              pauli_mask=[]))
    eq.make_equality_group(lambda:
        cirq.MutableDensePauliString(coefficient=2,
                              pauli_mask=[2]))


def test_eye():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text
    assert cirq.BaseDensePauliString.eye(4) == f('IIII')
    assert cirq.DensePauliString.eye(4) == f('IIII')
    assert cirq.MutableDensePauliString.eye(4) == m('IIII')


def test_from_text():
    assert cirq.DensePauliString.from_text('') == cirq.DensePauliString(
        pauli_mask=[],
        coefficient=1)

    assert cirq.DensePauliString.from_text('YYXYY') == cirq.DensePauliString(
        pauli_mask=[3, 3, 1, 3, 3],
        coefficient=1)
    assert cirq.DensePauliString.from_text('+XYZI') == cirq.DensePauliString(
        pauli_mask=[1, 3, 2, 0],
        coefficient=1)
    assert cirq.DensePauliString.from_text('-III') == cirq.DensePauliString(
        pauli_mask=[0, 0, 0],
        coefficient=-1)
    assert cirq.DensePauliString.from_text('1j*XXY') == cirq.DensePauliString(
        pauli_mask=[1, 1, 3],
        coefficient=1j)

    assert isinstance(cirq.BaseDensePauliString.from_text(''),
                      cirq.DensePauliString)
    assert isinstance(cirq.MutableDensePauliString.from_text(''),
                      cirq.MutableDensePauliString)
    assert isinstance(cirq.DensePauliString.from_text(''),
                      cirq.DensePauliString)


def test_sparse():
    a, b, c = cirq.LineQubit.range(3)
    assert cirq.DensePauliString.from_text('-XXY').sparse() == cirq.PauliString(
        -1,
        cirq.X(a),
        cirq.X(b),
        cirq.Y(c))


def test_mul_vectorized_pauli_mul_phase():
    f = _vectorized_pauli_mul_phase
    paulis = [cirq.I, cirq.X, cirq.Z, cirq.Y]
    q = cirq.LineQubit(0)

    # Check single qubit cases.
    for i in range(4):
        for j in range(4):
            sparse1 = cirq.PauliString(paulis[i].on(q))
            sparse2 = cirq.PauliString(paulis[j].on(q))
            assert f(i, j) == (sparse1 * sparse2).coefficient

    # Check a vector case.
    assert _vectorized_pauli_mul_phase(
        np.array([0, 1, 2, 3], dtype=np.uint8),
        np.array([0, 1, 3, 0], dtype=np.uint8)) == -1j
    assert _vectorized_pauli_mul_phase(
        np.array([], dtype=np.uint8),
        np.array([], dtype=np.uint8)) == 1


def test_mul():
    f = cirq.DensePauliString.from_text

    # Scalar.
    assert -1 * f('XXX') == f('-XXX')
    assert -1 * f('XXX') == -f('XXX')
    assert f('-XXX') == -1 * f('XXX')
    assert f('-XXX') == -1.0 * f('XXX')
    assert 2 * f('XXX') == f('XXX') * 2 == f('2*XXX')

    # Pair.
    assert f('') * f('') == f('')
    assert f('-X') * f('1j*X') == f('-1j*I')
    assert f('IXYZ') * f('XXXX') == f('XIZY')
    assert f('IXYX') * f('XXXX') == -1j*f('XIZI')
    assert f('XXXX') * f('IXYX') == 1j*f('XIZI')

    # Pauli operations.
    assert f('IXYZ') * cirq.X(cirq.LineQubit(0)) == f('XXYZ')
    assert f('-IXYZ') * cirq.X(cirq.LineQubit(1)) == f('-IIYZ')
    assert f('IXYZ') * cirq.X(cirq.LineQubit(2)) == -1j * f('IXZZ')
    assert cirq.X(cirq.LineQubit(0)) * f('IXYZ') == f('XXYZ')
    assert cirq.X(cirq.LineQubit(1)) * f('-IXYZ') == f('-IIYZ')
    assert cirq.X(cirq.LineQubit(2)) * f('IXYZ') == 1j * f('IXZZ')

    # Mixed types.
    m = cirq.MutableDensePauliString.from_text
    assert m('X') * m('Z') == -1j * f('Y')
    assert m('X') * f('Z') == -1j * f('Y')
    assert isinstance(f('') * f(''), cirq.DensePauliString)
    assert isinstance(m('') * m(''), cirq.DensePauliString)
    assert isinstance(m('') * f(''), cirq.DensePauliString)

    # Different lengths.
    assert f('I') * f('III') == f('III')
    assert f('X') * f('XXX') == f('IXX')
    assert f('XXX') * f('X') == f('IXX')


def test_imul():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text

    # Immutable not modified by imul.
    p = f('III')
    p2 = p
    p2 *= 2
    assert p.coefficient == 1
    assert p is not p2

    # Mutable is modified by imul.
    p = m('III')
    p2 = p
    p2 *= 2
    assert p.coefficient == 2
    assert p is p2

    p *= f('X')
    assert p == m('2*XII')

    p *= m('XY')
    assert p == m('2*IYI')

    p *= 1j
    assert p == m('2j*IYI')

    p *= 0.5
    assert p == m('1j*IYI')

    p *= cirq.X(cirq.LineQubit(1))
    assert p == m('IZI')

    with pytest.raises(ValueError, match='smaller than'):
        p *= f('XXXXXXXXXXXX')


def test_pos_neg():
    p = cirq.DensePauliString.from_text('1j*XYZ')
    assert +p == p
    assert -p == -1 * p


def test_abs():
    f = cirq.DensePauliString.from_text
    m = cirq.DensePauliString.from_text
    assert abs(f('-XX')) == f('XX')
    assert abs(f('2j*XX')) == f('2*XX')
    assert abs(m('2j*XX')) == f('2*XX')


def test_approx_eq():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text

    # Tolerance matters.
    assert cirq.approx_eq(f('1.00001*X'), f('X'), atol=1e-4)
    assert cirq.approx_eq(m('1.00001*X'), m('X'), atol=1e-4)
    assert not cirq.approx_eq(f('1.00001*X'), f('X'), atol=1e-8)
    assert not cirq.approx_eq(m('1.00001*X'), m('X'), atol=1e-8)

    # Must be same type.
    assert not cirq.approx_eq(f('X'), m('X'), atol=1e-4)

    # Differing paulis ignores tolerance.
    assert not cirq.approx_eq(f('X'), f('YYY'), atol=1e-8)
    assert not cirq.approx_eq(f('X'), f('Y'), atol=1e-8)
    assert not cirq.approx_eq(f('X'), f('Y'), atol=500)


def test_pow():
    f = cirq.DensePauliString.from_text
    m = cirq.DensePauliString.from_text
    p = f('1j*IXYZ')
    assert p**0 == p**4 == p**8 == cirq.DensePauliString.eye(4)
    assert p**1 == p**5 == p**-3 == p == p**101
    assert p**2 == p**-2 == p**6 == f('-IIII')
    assert p**3 == p**-1 == p**7 == f('-1j*IXYZ')

    p = f('-IXYZ')
    assert p == p**1 == p**-1 == p**-3 == p**-303
    assert p**0 == p**2 == p**-2 == p**-4 == p**102

    p = f('2*XX')
    assert p**-1 == f('(0.5+0j)*XX')
    assert p**0 == f('II')
    assert p**1 == f('2*XX')
    assert p**2 == f('4*II')
    assert p**3 == f('8*XX')
    assert p**4 == f('16*II')

    p = f('-1j*XY')
    assert p**101 == p == p**-103

    p = f('2j*XY')
    assert (p**-1)**-1 == p
    assert p**-2 == f('II') / -4

    p = f('XY')
    assert p**-100 == p**0 == p**100 == f('II')
    assert p**-101 == p**1 == p**101 == f('XY')

    # Becomes an immutable copy.
    assert m('X')**3 == f('X')


def test_div():
    f = cirq.DensePauliString.from_text
    assert f('X') / 2 == 0.5 * f('X')


def test_str():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text

    assert str(f('')) == '+'
    assert str(f('XXX')) == '+XXX'
    assert str(m('XXX')) == '+XXX (mutable)'
    assert str(2*f('')) == '(2+0j)*'
    assert str(f('(1+1j)*XXX')) == '(1+1j)*XXX'
    assert str(f('1j*XXX')) == '1j*XXX'
    assert str(f('-IXYZ')) == '-IXYZ'


def test_repr():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text
    cirq.testing.assert_equivalent_repr(f(''))
    cirq.testing.assert_equivalent_repr(f('-X'))
    cirq.testing.assert_equivalent_repr(f('1j*XYZII'))
    cirq.testing.assert_equivalent_repr(m(''))
    cirq.testing.assert_equivalent_repr(m('-X'))
    cirq.testing.assert_equivalent_repr(m('1j*XYZII'))


def test_one_hot():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text

    assert cirq.DensePauliString.one_hot(index=3, length=5, pauli=cirq.X) == f(
        'IIIXI')
    assert cirq.MutableDensePauliString.one_hot(
        index=3, length=5, pauli=cirq.X) == m('IIIXI')

    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli='X') == f('XIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli='Y') == f('YIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli='Z') == f('ZIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli='I') == f('IIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli=cirq.X) == f('XIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli=cirq.Y) == f('YIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli=cirq.Z) == f('ZIIII')
    assert cirq.BaseDensePauliString.one_hot(
        index=0, length=5, pauli=cirq.I) == f('IIIII')

    with pytest.raises(IndexError):
        _ = cirq.BaseDensePauliString.one_hot(
            index=50, length=5, pauli=cirq.X)

    with pytest.raises(IndexError):
        _ = cirq.BaseDensePauliString.one_hot(
            index=0, length=0, pauli=cirq.X)


def test_protocols():
    cirq.testing.assert_implements_consistent_protocols(
        cirq.DensePauliString.from_text('Y'))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.DensePauliString.from_text('-Z'))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.DensePauliString.from_text('1j*X'))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.DensePauliString.from_text('2*X'))

    cirq.testing.assert_implements_consistent_protocols(
        cirq.DensePauliString.from_text('-XYIZ'))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.MutableDensePauliString.from_text('-XYIZ'))

    assert cirq.has_unitary(cirq.DensePauliString.from_text('1j*X'))
    assert not cirq.has_unitary(cirq.DensePauliString.from_text('2j*X'))
    p = cirq.DensePauliString.from_text('-XYIZ')
    assert cirq.num_qubits(p) == len(p) == 4


def test_item_immutable():
    p = cirq.DensePauliString.from_text('-XYIZ')
    assert p[-1] == cirq.Z
    assert p[0] == cirq.X
    assert p[1] == cirq.Y
    assert p[2] == cirq.I
    assert p[3] == cirq.Z

    with pytest.raises(IndexError):
        _ = p[4]
    with pytest.raises(TypeError):
        p[2] = cirq.X
    with pytest.raises(TypeError):
        p[:] = p

    assert p[:] == abs(p)
    assert p[1:] == cirq.DensePauliString.from_text('YIZ')
    assert p[::2] == cirq.DensePauliString.from_text('XI')


def test_item_mutable():
    m = cirq.MutableDensePauliString.from_text
    p = m('-XYIZ')
    assert p[-1] == cirq.Z
    assert p[0] == cirq.X
    assert p[1] == cirq.Y
    assert p[2] == cirq.I
    assert p[3] == cirq.Z
    with pytest.raises(IndexError):
        _ = p[4]

    # Mutable.
    p[2] = cirq.X
    assert p == m('-XYXZ')
    p[3] = 'X'
    p[0] = 'I'
    assert p == m('-IYXX')
    p[2:] = p[:2]
    assert p == m('-IYIY')
    p[2:] = 'ZZ'
    assert p == m('-IYZZ')
    p[2:] = 'IY'
    assert p == m('-IYIY')

    # Aliased views.
    q = p[:2]
    assert q == m('IY')
    q[0] = cirq.Z
    assert q == m('ZY')
    assert p == m('-ZYIY')

    with pytest.raises(ValueError, match='coefficient is not 1'):
        p[:] = p

    assert p[:] == m('ZYIY')
    assert p[1:] == m('YIY')
    assert p[::2] == m('ZI')


def test_tensor_product():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text
    assert f('2*XX').tensor_product(f('-XI')) == f('-2*XXXI')
    assert m('2*XX').tensor_product(f('-XI')) == f('-2*XXXI')
    assert m('2*XX').tensor_product(m('-XI')) == f('-2*XXXI')


def test_commutes():
    f = cirq.DensePauliString.from_text
    m = cirq.MutableDensePauliString.from_text
    # TODO(craiggidney,bryano): use commutes protocol instead
    commutes = cirq.BaseDensePauliString._commutes_
    assert cirq.commutes is cirq.linalg.commutes

    assert commutes(f('XX'), m('ZZ'))
    assert commutes(f('2*XX'), m('3*ZZ'))
    assert commutes(f('2*IX'), f('3*IX'))
    assert not commutes(f('IX'), f('IZ'))
    assert commutes(f('IIIXII'), cirq.X(cirq.LineQubit(3)))
    assert commutes(f('IIIXII'), cirq.X(cirq.LineQubit(2)))
    assert not commutes(f('IIIXII'), cirq.Z(cirq.LineQubit(3)))
    assert commutes(f('IIIXII'), cirq.Z(cirq.LineQubit(2)))


def test_copy():
    p = cirq.DensePauliString.from_text('-XYZ')
    m = cirq.MutableDensePauliString.from_text('-XYZ')

    assert p.copy() is p
    assert p.frozen() is p
    assert p.mutable_copy() is not p
    assert p.mutable_copy() == m

    assert m.copy() is not m
    assert m.copy() == m
    assert m.frozen() == p
    assert m.mutable_copy() is not m
    assert m.mutable_copy() == m

    assert p.copy(coefficient=-1) is p
    assert p.copy(coefficient=-2) is not p
    assert p.copy(coefficient=-2) == cirq.DensePauliString.from_text('-2*XYZ')
    assert p.copy(coefficient=-2,
                  pauli_mask=[2]) == cirq.DensePauliString.from_text('-2*Z')

    assert m.copy(coefficient=-1) is not m
    assert m.copy(coefficient=-2) is not m
    assert m.copy(coefficient=-2) == cirq.MutableDensePauliString.from_text('-2*XYZ')
    assert m.copy(coefficient=-2,
                  pauli_mask=[2]) == cirq.MutableDensePauliString.from_text('-2*Z')
