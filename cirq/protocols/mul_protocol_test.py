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
import sympy

import cirq


class Neither:
    pass


class MulReturnsNotImplemented:

    def __mul__(self, other):
        return NotImplemented


class RMulReturnsNotImplemented:

    def __rmul__(self, other):
        return NotImplemented


class MulReturnsFive:

    def __mul__(self, other):
        return 5


class RMulReturnsSix:

    def __rmul__(self, other):
        return 6


class MulSevenRMulEight:

    def __mul__(self, other):
        return 7

    def __rmul__(self, other):
        return 8


def test_equivalent_to_builtin_mul():
    test_vals = [
        0,
        1,
        1j,
        -2.5,
        Neither(),
        MulReturnsNotImplemented(),
        RMulReturnsNotImplemented(),
        MulReturnsFive(),
        RMulReturnsSix(),
        MulSevenRMulEight(),
    ]

    for a in test_vals:
        for b in test_vals:
            if type(a) == type(b) == RMulReturnsSix:
                # Python doesn't do __rmul__ if __mul__ failed and
                # type(a) == type(b). But we do.
                continue

            c = cirq.mul(a, b, default=None)
            if c is None:
                with pytest.raises(TypeError):
                    _ = a * b
                with pytest.raises(TypeError):
                    _ = cirq.mul(a, b)
            else:
                assert c == a * b


def test_symbol_special_case():
    x = sympy.Symbol('x')
    assert cirq.mul(x, 1.0) is x
    assert cirq.mul(1.0, x) is x
    assert str(cirq.mul(-1.0, x)) == '-x'
    assert str(cirq.mul(x, -1.0)) == '-x'
