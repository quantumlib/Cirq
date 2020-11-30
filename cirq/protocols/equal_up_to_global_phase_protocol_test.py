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
import cirq


def test_equal_up_to_global_phase_primitives():
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 + 1j, atol=1e-9)
    assert not cirq.equal_up_to_global_phase(2.0, 1.0 + 1j, atol=1e-9)
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 - 1j, atol=1e-9)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1.0 + 0.0j, atol=1e-9)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1.0j, atol=1e-9)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1, atol=1e-9)
    assert not cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 0, atol=1e-9)
    assert cirq.equal_up_to_global_phase(1j, 1 + 1e-10, atol=1e-9)
    assert not cirq.equal_up_to_global_phase(1j, 1 + 1e-10, atol=1e-11)
    # atol is applied to magnitude of complex vector, not components.
    assert cirq.equal_up_to_global_phase(1.0 + 0.1j, 1.0, atol=0.01)
    assert not cirq.equal_up_to_global_phase(1.0 + 0.1j, 1.0, atol=0.001)
    assert cirq.equal_up_to_global_phase(1.0 + 1j, np.sqrt(2) + 1e-8, atol=1e-7)
    assert not cirq.equal_up_to_global_phase(1.0 + 1j, np.sqrt(2) + 1e-7, atol=1e-8)
    assert cirq.equal_up_to_global_phase(1.0 + 1e-10j, 1.0, atol=1e-15)


def test_equal_up_to_global_numeric_iterables():
    assert cirq.equal_up_to_global_phase([], [], atol=1e-9)
    assert cirq.equal_up_to_global_phase([[]], [[]], atol=1e-9)
    assert cirq.equal_up_to_global_phase([1j, 1], [1j, 1], atol=1e-9)
    assert cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 + 0.1j], atol=0.01)
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 - 0.1j], atol=0.01)
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1 + 0.1j, 1 + 0.1j], atol=1e-3)
    assert not cirq.equal_up_to_global_phase([1j, -1j], [1, 1], atol=0.0)
    assert not cirq.equal_up_to_global_phase([1j, 1], [1, 1j], atol=0.0)
    assert not cirq.equal_up_to_global_phase([1j, 1], [1j, 1, 0], atol=0.0)
    assert cirq.equal_up_to_global_phase((1j, 1j), (1, 1 + 1e-4), atol=1e-3)
    assert not cirq.equal_up_to_global_phase((1j, 1j), (1, 1 + 1e-4), atol=1e-5)
    assert not cirq.equal_up_to_global_phase((1j, 1), (1, 1j), atol=1e-09)


def test_equal_up_to_global_numpy_array():
    assert cirq.equal_up_to_global_phase(
        np.asarray([1j, 1j]), np.asarray([1, 1], dtype=np.complex64)
    )
    assert not cirq.equal_up_to_global_phase(
        np.asarray([1j, -1j]), np.asarray([1, 1], dtype=np.complex64)
    )
    assert cirq.equal_up_to_global_phase(np.asarray([]), np.asarray([]))
    assert cirq.equal_up_to_global_phase(np.asarray([[]]), np.asarray([[]]))


def test_equal_up_to_global_mixed_array_types():
    a = [1j, 1, -1j, -1]
    b = [-1, 1j, 1, -1j]
    c = [-1, 1, -1, 1]
    assert cirq.equal_up_to_global_phase(a, tuple(b))
    assert not cirq.equal_up_to_global_phase(a, tuple(c))

    c_types = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):
        c_types.append(np.complex256)
    for c_type in c_types:
        assert cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), tuple(b))
        assert not cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), tuple(c))
        assert cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), b)
        assert not cirq.equal_up_to_global_phase(np.asarray(a, dtype=c_type), c)

    # Object arrays and mixed array/scalar comparisons.
    assert not cirq.equal_up_to_global_phase([1j], 1j)
    assert not cirq.equal_up_to_global_phase(np.asarray([1], dtype=np.complex128), np.exp(1j))
    assert not cirq.equal_up_to_global_phase([1j, 1j], [1j, "1j"])
    assert not cirq.equal_up_to_global_phase([1j], "Non-numeric iterable")
    assert not cirq.equal_up_to_global_phase([], [[]], atol=0.0)


# Dummy container class implementing _equal_up_to_global_phase_
# for homogeneous comparison, with nontrivial getter.
class A:
    def __init__(self, val):
        self.val = [val]

    def _equal_up_to_global_phase_(self, other, atol):
        if not isinstance(other, A):
            return NotImplemented
        return cirq.equal_up_to_global_phase(self.val[0], other.val[0], atol=atol)


# Dummy container class implementing _equal_up_to_global_phase_
# for heterogeneous comparison.
class B:
    def __init__(self, val):
        self.val = [val]

    def _equal_up_to_global_phase_(self, other, atol):
        if not isinstance(self.val[0], type(other)):
            return NotImplemented
        return cirq.equal_up_to_global_phase(self.val[0], other, atol=atol)


def test_equal_up_to_global_phase_eq_supported():
    assert cirq.equal_up_to_global_phase(A(0.1 + 0j), A(0.1j), atol=1e-2)
    assert not cirq.equal_up_to_global_phase(A(0.0 + 0j), A(0.1j), atol=0.0)
    assert not cirq.equal_up_to_global_phase(A(0.0 + 0j), 0.1j, atol=0.0)
    assert cirq.equal_up_to_global_phase(B(0.0j), 1e-8j, atol=1e-8)
    assert cirq.equal_up_to_global_phase(1e-8j, B(0.0j), atol=1e-8)
    assert not cirq.equal_up_to_global_phase(1e-8j, B(0.0 + 0j), atol=1e-10)
    # cast types
    assert cirq.equal_up_to_global_phase(A(0.1), A(0.1j), atol=1e-2)
    assert not cirq.equal_up_to_global_phase(1e-8j, B(0.0), atol=1e-10)
