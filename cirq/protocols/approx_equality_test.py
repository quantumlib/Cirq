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

import math

import cirq


def test_approx_eq_primitives():
    assert cirq.approx_eq(1.0, 1.0 + 1e-10) is True
    assert cirq.approx_eq(1.0, 1.0 + 1e-09) is False
    assert cirq.approx_eq(1, 1.0 + 1e-10) is True
    assert cirq.approx_eq(0.0, 1e-10) is False
    assert cirq.approx_eq(0.0, 1e-10, abs_tol=1e-09) is True
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), rel_tol=0.2) \
           is True
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), rel_tol=0.1) \
           is False


def test_approx_eq_tuple():
    assert cirq.approx_eq((1, 1), (1, 1)) is True
    assert cirq.approx_eq((1, 1), (1, 1, 1)) is False
    assert cirq.approx_eq((1, 1), (1,)) is False
    assert cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), rel_tol=0.3) is True
    assert cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), rel_tol=0.2) is False


def test_approx_eq_iterables():
    assert cirq.approx_eq((1, 1), [1, 1]) is True


class A:

    def __init__(self, val):
        self.val =  val

    def _approx_eq_(self, other, rel_tol, abs_tol):
        if not isinstance(self, type(other)):
            return NotImplemented
        return math.isclose(
            self.val,
            other.val,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )


def test_approx_eq_supported():
    assert cirq.approx_eq(A(0.0), A(0.1), abs_tol=0.1) is True
    assert cirq.approx_eq(A(0.0), A(0.1)) is False


class B:
    pass


def test_approx_eq_not_supported():
    assert cirq.approx_eq(B(), B()) is NotImplemented
    assert cirq.approx_eq([B()], [B()]) is NotImplemented
    assert cirq.approx_eq(B(), A(0)) is NotImplemented
    assert cirq.approx_eq(A(0), B()) is NotImplemented
