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

from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq


def test_approx_eq_primitives():
    assert not cirq.approx_eq(1, 2, atol=1e-01)
    assert cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-09)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-11)
    assert cirq.approx_eq(0.0, 1e-10, atol=1e-09)
    assert not cirq.approx_eq(0.0, 1e-10, atol=1e-11)
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.3)
    assert not cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.1)
    assert cirq.approx_eq('ab', 'ab', atol=1e-3)
    assert not cirq.approx_eq('ab', 'ac', atol=1e-3)
    assert not cirq.approx_eq('1', '2', atol=999)
    assert not cirq.approx_eq('test', 1, atol=1e-3)
    assert not cirq.approx_eq('1', 1, atol=1e-3)


def test_approx_eq_mixed_primitives():
    assert cirq.approx_eq(complex(1, 1e-10), 1, atol=1e-09)
    assert not cirq.approx_eq(complex(1, 1e-4), 1, atol=1e-09)
    assert cirq.approx_eq(complex(1, 1e-10), 1.0, atol=1e-09)
    assert not cirq.approx_eq(complex(1, 1e-8), 1.0, atol=1e-09)
    assert cirq.approx_eq(1, 1.0 + 1e-10, atol=1e-9)
    assert not cirq.approx_eq(1, 1.0 + 1e-10, atol=1e-11)


def test_numpy_dtype_compatibility():
    i_a, i_b, i_c = 0, 1, 2
    i_types = [np.intc, np.intp, np.int0, np.int8, np.int16, np.int32, np.int64]
    for i_type in i_types:
        assert cirq.approx_eq(i_type(i_a), i_type(i_b), atol=1)
        assert not cirq.approx_eq(i_type(i_a), i_type(i_c), atol=1)
    u_types = [np.uint, np.uint0, np.uint8, np.uint16, np.uint32, np.uint64]
    for u_type in u_types:
        assert cirq.approx_eq(u_type(i_a), u_type(i_b), atol=1)
        assert not cirq.approx_eq(u_type(i_a), u_type(i_c), atol=1)

    f_a, f_b, f_c = 0, 1e-8, 1
    f_types = [np.float16, np.float32, np.float64]
    if hasattr(np, 'float128'):
        f_types.append(np.float128)
    for f_type in f_types:
        assert cirq.approx_eq(f_type(f_a), f_type(f_b), atol=1e-8)
        assert not cirq.approx_eq(f_type(f_a), f_type(f_c), atol=1e-8)

    c_a, c_b, c_c = 0, 1e-8j, 1j
    c_types = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):
        c_types.append(np.complex256)
    for c_type in c_types:
        assert cirq.approx_eq(c_type(c_a), c_type(c_b), atol=1e-8)
        assert not cirq.approx_eq(c_type(c_a), c_type(c_c), atol=1e-8)


def test_fractions_compatibility():
    assert cirq.approx_eq(Fraction(0), Fraction(1, int(1e10)), atol=1e-9)
    assert not cirq.approx_eq(Fraction(0), Fraction(1, int(1e7)), atol=1e-9)


def test_decimal_compatibility():
    assert cirq.approx_eq(Decimal('0'), Decimal('0.0000000001'), atol=1e-9)
    assert not cirq.approx_eq(Decimal('0'), Decimal('0.00000001'), atol=1e-9)
    assert not cirq.approx_eq(Decimal('NaN'), Decimal('-Infinity'), atol=1e-9)


def test_approx_eq_mixed_types():
    assert cirq.approx_eq(np.float32(1), 1.0 + 1e-10, atol=1e-9)
    assert cirq.approx_eq(np.float64(1), np.complex64(1 + 1e-8j), atol=1e-4)
    assert cirq.approx_eq(np.uint8(1), np.complex64(1 + 1e-8j), atol=1e-4)
    if hasattr(np, 'complex256'):
        assert cirq.approx_eq(np.complex256(1), complex(1, 1e-8), atol=1e-4)
    assert cirq.approx_eq(np.int32(1), 1, atol=1e-9)
    assert cirq.approx_eq(complex(0.5, 0), Fraction(1, 2), atol=0.0)
    assert cirq.approx_eq(0.5 + 1e-4j, Fraction(1, 2), atol=1e-4)
    assert cirq.approx_eq(0, Fraction(1, 100000000), atol=1e-8)
    assert cirq.approx_eq(np.uint16(1), Decimal('1'), atol=0.0)
    assert cirq.approx_eq(np.float64(1.0), Decimal('1.00000001'), atol=1e-8)
    assert not cirq.approx_eq(np.complex64(1e-5j), Decimal('0.001'), atol=1e-4)


def test_approx_eq_special_numerics():
    assert not cirq.approx_eq(float('nan'), 0, atol=0.0)
    assert not cirq.approx_eq(float('nan'), float('nan'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), float('-inf'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), 5, atol=0.0)
    assert not cirq.approx_eq(float('inf'), 0, atol=0.0)
    assert cirq.approx_eq(float('inf'), float('inf'), atol=0.0)


class X(Number):
    """Subtype of Number that can fallback to __eq__"""

    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self.val == other.val


class Y(Number):
    """Subtype of Number that cannot fallback to __eq__"""

    def __init__(self):
        pass


def test_approx_eq_number_uses__eq__():
    assert cirq.approx_eq(C(0), C(0), atol=0.0)
    assert not cirq.approx_eq(X(0), X(1), atol=0.0)
    assert not cirq.approx_eq(X(0), 0, atol=0.0)
    assert not cirq.approx_eq(Y(), 1, atol=0.0)


def test_approx_eq_tuple():
    assert cirq.approx_eq((1, 1), (1, 1), atol=0.0)
    assert not cirq.approx_eq((1, 1), (1, 1, 1), atol=0.0)
    assert not cirq.approx_eq((1, 1), (1,), atol=0.0)
    assert cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), atol=0.4)
    assert not cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), atol=0.2)


def test_approx_eq_list():
    assert cirq.approx_eq([], [], atol=0.0)
    assert not cirq.approx_eq([], [[]], atol=0.0)
    assert cirq.approx_eq([1, 1], [1, 1], atol=0.0)
    assert not cirq.approx_eq([1, 1], [1, 1, 1], atol=0.0)
    assert not cirq.approx_eq([1, 1], [1], atol=0.0)
    assert cirq.approx_eq([1.1, 1.2, 1.3], [1, 1, 1], atol=0.4)
    assert not cirq.approx_eq([1.1, 1.2, 1.3], [1, 1, 1], atol=0.2)


def test_approx_eq_symbol():
    q = cirq.GridQubit(0, 0)
    s = sympy.Symbol("s")
    t = sympy.Symbol("t")

    assert not cirq.approx_eq(t + 1.51 + s, t + 1.50 + s, atol=0.005)
    assert cirq.approx_eq(t + 1.51 + s, t + 1.50 + s, atol=0.020)

    with pytest.raises(
        AttributeError,
        match="Insufficient information to decide whether expressions are "
        "approximately equal .* vs .*",
    ):
        cirq.approx_eq(t, 0.0, atol=0.005)

    symbol_1 = cirq.Circuit(cirq.rz(1.515 + s)(q))
    symbol_2 = cirq.Circuit(cirq.rz(1.510 + s)(q))
    assert cirq.approx_eq(symbol_1, symbol_2, atol=0.2)

    symbol_3 = cirq.Circuit(cirq.rz(1.510 + t)(q))
    with pytest.raises(
        AttributeError,
        match="Insufficient information to decide whether expressions are "
        "approximately equal .* vs .*",
    ):
        cirq.approx_eq(symbol_1, symbol_3, atol=0.2)


def test_approx_eq_default():
    assert cirq.approx_eq(1.0, 1.0 + 1e-9)
    assert cirq.approx_eq(1.0, 1.0 - 1e-9)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-7)
    assert not cirq.approx_eq(1.0, 1.0 - 1e-7)


def test_approx_eq_iterables():
    def gen_1_1():
        yield 1
        yield 1

    assert cirq.approx_eq((1, 1), [1, 1], atol=0.0)
    assert cirq.approx_eq((1, 1), gen_1_1(), atol=0.0)
    assert cirq.approx_eq(gen_1_1(), [1, 1], atol=0.0)


class A:
    def __init__(self, val):
        self.val = val

    def _approx_eq_(self, other, atol):
        if not isinstance(self, type(other)):
            return NotImplemented
        return cirq.approx_eq(self.val, other.val, atol=atol)


class B:
    def __init__(self, val):
        self.val = val

    def _approx_eq_(self, other, atol):
        if not isinstance(self.val, type(other)):
            return NotImplemented
        return cirq.approx_eq(self.val, other, atol=atol)


def test_approx_eq_supported():
    assert cirq.approx_eq(A(0.0), A(0.1), atol=0.1)
    assert not cirq.approx_eq(A(0.0), A(0.1), atol=0.0)
    assert cirq.approx_eq(B(0.0), 0.1, atol=0.1)
    assert cirq.approx_eq(0.1, B(0.0), atol=0.1)


class C:
    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self.val == other.val


def test_approx_eq_uses__eq__():
    assert cirq.approx_eq(C(0), C(0), atol=0.0)
    assert not cirq.approx_eq(C(1), C(2), atol=0.0)
    assert cirq.approx_eq([C(0)], [C(0)], atol=0.0)
    assert not cirq.approx_eq([C(1)], [C(2)], atol=0.0)
    assert cirq.approx_eq(complex(0, 0), 0, atol=0.0)
    assert cirq.approx_eq(0, complex(0, 0), atol=0.0)


def test_approx_eq_types_mismatch():
    assert not cirq.approx_eq(0, A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), 0, atol=0.0)
    assert not cirq.approx_eq(B(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), B(0), atol=0.0)
    assert not cirq.approx_eq(C(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), C(0), atol=0.0)
    assert not cirq.approx_eq(0, [0], atol=1.0)
    assert not cirq.approx_eq([0], 0, atol=0.0)
