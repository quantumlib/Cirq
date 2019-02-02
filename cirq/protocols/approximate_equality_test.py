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
    assert cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-09)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-11)
    assert cirq.approx_eq(0.0, 1e-10, atol=1e-09)
    assert not cirq.approx_eq(0.0, 1e-10, atol=1e-11)
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.3)
    assert not cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.1)


def test_approx_eq_special_numerics():
    assert not cirq.approx_eq(float('nan'), 0, atol=0.0)
    assert not cirq.approx_eq(float('nan'), float('nan'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), float('-inf'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), 5, atol=0.0)
    assert not cirq.approx_eq(float('inf'), 0, atol=0.0)
    assert cirq.approx_eq(float('inf'), float('inf'), atol=0.0)


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
    assert not cirq.approx_eq([1, 1], [1,], atol=0.0)
    assert cirq.approx_eq([1.1, 1.2, 1.3], [1, 1, 1], atol=0.4)
    assert not cirq.approx_eq([1.1, 1.2, 1.3], [1, 1, 1], atol=0.2)


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


def test_approx_eq_types_mismatch():
    assert not cirq.approx_eq(0, A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), 0, atol=0.0)
    assert not cirq.approx_eq(B(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), B(0), atol=0.0)
    assert not cirq.approx_eq(C(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), C(0), atol=0.0)
    assert not cirq.approx_eq(complex(0, 0), 0, atol=0.0)
    assert not cirq.approx_eq(0, complex(0, 0), atol=0.0)
    assert not cirq.approx_eq(0, [0], atol=1.0)
    assert not cirq.approx_eq([0], 0, atol=0.0)
