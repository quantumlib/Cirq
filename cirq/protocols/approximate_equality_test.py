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

import cirq


def test_approx_eq_primitives():
    assert cirq.approx_eq(1.0, 1.0 + 1e-10)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-09)
    assert cirq.approx_eq(1, 1.0 + 1e-10)
    assert not cirq.approx_eq(0.0, 1e-10)
    assert cirq.approx_eq(0.0, 1e-10, abs_tol=1e-09)
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), rel_tol=0.2)
    assert not cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), rel_tol=0.1)


def test_approx_eq_tuple():
    assert cirq.approx_eq((1, 1), (1, 1))
    assert not cirq.approx_eq((1, 1), (1, 1, 1))
    assert not cirq.approx_eq((1, 1), (1,))
    assert cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), rel_tol=0.3)
    assert not cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), rel_tol=0.2)


def test_approx_eq_iterables():
    assert cirq.approx_eq((1, 1), [1, 1])


class A:

    def __init__(self, val):
        self.val = val

    def _approx_eq_(self, other, rel_tol, abs_tol):
        if not isinstance(self, type(other)):
            return NotImplemented
        return cirq.approx_eq(
            self.val,
            other.val,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )


class B:

    def __init__(self, val):
        self.val = val

    def _approx_eq_(self, other, rel_tol, abs_tol):
        if not isinstance(self.val, type(other)):
            return NotImplemented
        return cirq.approx_eq(self.val, other, rel_tol=rel_tol, abs_tol=abs_tol)


def test_approx_eq_supported():
    assert cirq.approx_eq(A(0.0), A(0.1), abs_tol=0.1)
    assert not cirq.approx_eq(A(0.0), A(0.1))
    assert cirq.approx_eq(B(0.0), 0.1, abs_tol=0.1)
    assert cirq.approx_eq(0.1, B(0.0), abs_tol=0.1)


class C:

    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self.val == other.val


def test_approx_eq_uses__eq__():
    assert cirq.approx_eq(C(0), C(0))
    assert not cirq.approx_eq(C(1), C(2))
    assert cirq.approx_eq([C(0)], [C(0)])
    assert not cirq.approx_eq([C(1)], [C(2)])


def test_approx_eq_types_mismatch():
    assert not cirq.approx_eq(0, A(0))
    assert not cirq.approx_eq(A(0), 0)
    assert not cirq.approx_eq(B(0), A(0))
    assert not cirq.approx_eq(A(0), B(0))
    assert not cirq.approx_eq(C(0), A(0))
    assert not cirq.approx_eq(A(0), C(0))
    assert not cirq.approx_eq(complex(0, 0), 0)
    assert not cirq.approx_eq(0, complex(0, 0))
