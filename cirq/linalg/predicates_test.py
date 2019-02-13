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

import cmath
import numpy as np

import cirq
from cirq import Tolerance


def test_is_diagonal():
    assert cirq.is_diagonal(np.empty((0, 0)))
    assert cirq.is_diagonal(np.empty((1, 0)))
    assert cirq.is_diagonal(np.empty((0, 1)))

    assert cirq.is_diagonal(np.array([[1]]))
    assert cirq.is_diagonal(np.array([[-1]]))
    assert cirq.is_diagonal(np.array([[5]]))
    assert cirq.is_diagonal(np.array([[3j]]))

    assert cirq.is_diagonal(np.array([[1, 0]]))
    assert cirq.is_diagonal(np.array([[1], [0]]))
    assert not cirq.is_diagonal(np.array([[1, 1]]))
    assert not cirq.is_diagonal(np.array([[1], [1]]))

    assert cirq.is_diagonal(np.array([[5j, 0], [0, 2]]))
    assert cirq.is_diagonal(np.array([[1, 0], [0, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_diagonal(np.array([[1, 0.1], [0.1, 1]]))

    assert cirq.is_diagonal(np.array([[1, 1e-11], [1e-10, 1]]))


def test_is_diagonal_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_diagonal(np.array([[1, 0], [-0.5, 1]]), tol)
    assert not cirq.is_diagonal(np.array([[1, 0], [-0.6, 1]]), tol)

    # Error isn't accumulated across entries.
    assert cirq.is_diagonal(np.array([[1, 0.5], [-0.5, 1]]), tol)
    assert not cirq.is_diagonal(np.array([[1, 0.5], [-0.6, 1]]), tol)


def test_is_hermitian():
    assert cirq.is_hermitian(np.empty((0, 0)))
    assert not cirq.is_hermitian(np.empty((1, 0)))
    assert not cirq.is_hermitian(np.empty((0, 1)))

    assert cirq.is_hermitian(np.array([[1]]))
    assert cirq.is_hermitian(np.array([[-1]]))
    assert cirq.is_hermitian(np.array([[5]]))
    assert not cirq.is_hermitian(np.array([[3j]]))

    assert not cirq.is_hermitian(np.array([[0, 0]]))
    assert not cirq.is_hermitian(np.array([[0], [0]]))

    assert not cirq.is_hermitian(np.array([[5j, 0], [0, 2]]))
    assert cirq.is_hermitian(np.array([[5, 0], [0, 2]]))
    assert cirq.is_hermitian(np.array([[1, 0], [0, 1]]))
    assert not cirq.is_hermitian(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_hermitian(np.array([[1, 1], [0, 1]]))
    assert cirq.is_hermitian(np.array([[1, 1], [1, 1]]))
    assert cirq.is_hermitian(np.array([[1, 1j], [-1j, 1]]))
    assert cirq.is_hermitian(np.array([[1, 1j], [-1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_hermitian(np.array([[1, 1j], [1j, 1]]))
    assert not cirq.is_hermitian(np.array([[1, 0.1], [-0.1, 1]]))

    assert cirq.is_hermitian(
        np.array([[1, 1j + 1e-11], [-1j, 1 + 1j * 1e-9]]))


def test_is_hermitian_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_hermitian(np.array([[1, 0], [-0.5, 1]]), tol)
    assert cirq.is_hermitian(np.array([[1, 0.25], [-0.25, 1]]), tol)
    assert not cirq.is_hermitian(np.array([[1, 0], [-0.6, 1]]), tol)
    assert not cirq.is_hermitian(np.array([[1, 0.25], [-0.35, 1]]), tol)

    # Error isn't accumulated across entries.
    assert cirq.is_hermitian(
        np.array([[1, 0.5, 0.5], [0, 1, 0], [0, 0, 1]]), tol)
    assert not cirq.is_hermitian(
        np.array([[1, 0.5, 0.6], [0, 1, 0], [0, 0, 1]]), tol)
    assert not cirq.is_hermitian(
        np.array([[1, 0, 0.6], [0, 1, 0], [0, 0, 1]]), tol)


def test_is_unitary():
    assert cirq.is_unitary(np.empty((0, 0)))
    assert not cirq.is_unitary(np.empty((1, 0)))
    assert not cirq.is_unitary(np.empty((0, 1)))

    assert cirq.is_unitary(np.array([[1]]))
    assert cirq.is_unitary(np.array([[-1]]))
    assert cirq.is_unitary(np.array([[1j]]))
    assert not cirq.is_unitary(np.array([[5]]))
    assert not cirq.is_unitary(np.array([[3j]]))

    assert not cirq.is_unitary(np.array([[1, 0]]))
    assert not cirq.is_unitary(np.array([[1], [0]]))

    assert not cirq.is_unitary(np.array([[1, 0], [0, -2]]))
    assert cirq.is_unitary(np.array([[1, 0], [0, -1]]))
    assert cirq.is_unitary(np.array([[1j, 0], [0, 1]]))
    assert not cirq.is_unitary(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_unitary(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_unitary(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_unitary(np.array([[1, -1], [1, 1]]))
    assert cirq.is_unitary(np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert cirq.is_unitary(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_unitary(
        np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_unitary(
        np.array([[1, 1j + 1e-11], [1j, 1 + 1j * 1e-9]]) * np.sqrt(0.5))


def test_is_unitary_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_unitary(np.array([[1, 0], [-0.5, 1]]), tol)
    assert not cirq.is_unitary(np.array([[1, 0], [-0.6, 1]]), tol)

    # Error isn't accumulated across entries.
    assert cirq.is_unitary(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), tol)
    assert not cirq.is_unitary(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1.2]]), tol)


def test_is_orthogonal():
    assert cirq.is_orthogonal(np.empty((0, 0)))
    assert not cirq.is_orthogonal(np.empty((1, 0)))
    assert not cirq.is_orthogonal(np.empty((0, 1)))

    assert cirq.is_orthogonal(np.array([[1]]))
    assert cirq.is_orthogonal(np.array([[-1]]))
    assert not cirq.is_orthogonal(np.array([[1j]]))
    assert not cirq.is_orthogonal(np.array([[5]]))
    assert not cirq.is_orthogonal(np.array([[3j]]))

    assert not cirq.is_orthogonal(np.array([[1, 0]]))
    assert not cirq.is_orthogonal(np.array([[1], [0]]))

    assert not cirq.is_orthogonal(np.array([[1, 0], [0, -2]]))
    assert cirq.is_orthogonal(np.array([[1, 0], [0, -1]]))
    assert not cirq.is_orthogonal(np.array([[1j, 0], [0, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_orthogonal(np.array([[1, -1], [1, 1]]))
    assert cirq.is_orthogonal(np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert not cirq.is_orthogonal(
        np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_orthogonal(
        np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_orthogonal(np.array([[1, 1e-11], [0, 1 + 1e-11]]))


def test_is_orthogonal_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_orthogonal(np.array([[1, 0], [-0.5, 1]]), tol)
    assert not cirq.is_orthogonal(np.array([[1, 0], [-0.6, 1]]), tol)

    # Error isn't accumulated across entries.
    assert cirq.is_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), tol)
    assert not cirq.is_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1.2]]), tol)


def test_is_special_orthogonal():
    assert cirq.is_special_orthogonal(np.empty((0, 0)))
    assert not cirq.is_special_orthogonal(np.empty((1, 0)))
    assert not cirq.is_special_orthogonal(np.empty((0, 1)))

    assert cirq.is_special_orthogonal(np.array([[1]]))
    assert not cirq.is_special_orthogonal(np.array([[-1]]))
    assert not cirq.is_special_orthogonal(np.array([[1j]]))
    assert not cirq.is_special_orthogonal(np.array([[5]]))
    assert not cirq.is_special_orthogonal(np.array([[3j]]))

    assert not cirq.is_special_orthogonal(np.array([[1, 0]]))
    assert not cirq.is_special_orthogonal(np.array([[1], [0]]))

    assert not cirq.is_special_orthogonal(np.array([[1, 0], [0, -2]]))
    assert not cirq.is_special_orthogonal(np.array([[1, 0], [0, -1]]))
    assert cirq.is_special_orthogonal(np.array([[-1, 0], [0, -1]]))
    assert not cirq.is_special_orthogonal(np.array([[1j, 0], [0, 1]]))
    assert not cirq.is_special_orthogonal(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_special_orthogonal(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_special_orthogonal(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_special_orthogonal(np.array([[1, -1], [1, 1]]))
    assert cirq.is_special_orthogonal(
        np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(
        np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(
        np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_special_orthogonal(
        np.array([[1, 1e-11], [0, 1 + 1e-11]]))


def test_is_special_orthogonal_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_special_orthogonal(
        np.array([[1, 0], [-0.5, 1]]), tol)
    assert not cirq.is_special_orthogonal(
        np.array([[1, 0], [-0.6, 1]]), tol)

    # Error isn't accumulated across entries, except for determinant factors.
    assert cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1 / 1.2]]), tol)
    assert not cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), tol)
    assert not cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1 / 1.2]]), tol)


def test_is_special_unitary():
    assert cirq.is_special_unitary(np.empty((0, 0)))
    assert not cirq.is_special_unitary(np.empty((1, 0)))
    assert not cirq.is_special_unitary(np.empty((0, 1)))

    assert cirq.is_special_unitary(np.array([[1]]))
    assert not cirq.is_special_unitary(np.array([[-1]]))
    assert not cirq.is_special_unitary(np.array([[5]]))
    assert not cirq.is_special_unitary(np.array([[3j]]))

    assert not cirq.is_special_unitary(np.array([[1, 0], [0, -2]]))
    assert not cirq.is_special_unitary(np.array([[1, 0], [0, -1]]))
    assert cirq.is_special_unitary(np.array([[-1, 0], [0, -1]]))
    assert not cirq.is_special_unitary(np.array([[1j, 0], [0, 1]]))
    assert cirq.is_special_unitary(np.array([[1j, 0], [0, -1j]]))
    assert not cirq.is_special_unitary(np.array([[1, 0], [1, 1]]))
    assert not cirq.is_special_unitary(np.array([[1, 1], [0, 1]]))
    assert not cirq.is_special_unitary(np.array([[1, 1], [1, 1]]))
    assert not cirq.is_special_unitary(np.array([[1, -1], [1, 1]]))
    assert cirq.is_special_unitary(
        np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert cirq.is_special_unitary(
        np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_unitary(
        np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_special_unitary(
        np.array([[1, 1j + 1e-11], [1j, 1 + 1j * 1e-9]]) * np.sqrt(0.5))


def test_is_special_unitary_tolerance():
    tol = Tolerance(atol=0.5)

    # Pays attention to specified tolerance.
    assert cirq.is_special_unitary(np.array([[1, 0], [-0.5, 1]]), tol)
    assert not cirq.is_special_unitary(np.array([[1, 0], [-0.6, 1]]), tol)
    assert cirq.is_special_unitary(
        np.array([[1, 0], [0, 1]]) * cmath.exp(1j * 0.1), tol)
    assert not cirq.is_special_unitary(
        np.array([[1, 0], [0, 1]]) * cmath.exp(1j * 0.3), tol)

    # Error isn't accumulated across entries, except for determinant factors.
    assert cirq.is_special_unitary(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1 / 1.2]]), tol)
    assert not cirq.is_special_unitary(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), tol)
    assert not cirq.is_special_unitary(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1 / 1.2]]), tol)


def test_commutes():
    assert cirq.commutes(
        np.empty((0, 0)),
        np.empty((0, 0)))
    assert not cirq.commutes(
        np.empty((1, 0)),
        np.empty((0, 1)))
    assert not cirq.commutes(
        np.empty((0, 1)),
        np.empty((1, 0)))
    assert not cirq.commutes(
        np.empty((1, 0)),
        np.empty((1, 0)))
    assert not cirq.commutes(
        np.empty((0, 1)),
        np.empty((0, 1)))

    assert cirq.commutes(np.array([[1]]), np.array([[2]]))
    assert cirq.commutes(np.array([[1]]), np.array([[0]]))

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    xx = np.kron(x, x)
    zz = np.kron(z, z)

    assert cirq.commutes(x, x)
    assert cirq.commutes(y, y)
    assert cirq.commutes(z, z)
    assert not cirq.commutes(x, y)
    assert not cirq.commutes(x, z)
    assert not cirq.commutes(y, z)

    assert cirq.commutes(xx, zz)
    assert cirq.commutes(xx, np.diag([1, -1, -1, 1 + 1e-9]))


def test_commutes_tolerance():
    tol = Tolerance(atol=0.5)

    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1, 0], [0, -1]])

    # Pays attention to specified tolerance.
    assert cirq.commutes(x, x + z * 0.1, tol)
    assert not cirq.commutes(x, x + z * 0.5, tol)


def test_allclose_up_to_global_phase():
    assert cirq.allclose_up_to_global_phase(
        np.array([1]),
        np.array([1j]))

    assert cirq.allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[1]]))
    assert cirq.allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[-1]]))

    assert cirq.allclose_up_to_global_phase(
        np.array([[0]]),
        np.array([[0]]))

    assert cirq.allclose_up_to_global_phase(
        np.array([[1, 2]]),
        np.array([[1j, 2j]]))

    assert cirq.allclose_up_to_global_phase(
        np.array([[1, 2.0000000001]]),
        np.array([[1j, 2j]]))

    assert not cirq.allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[1, 0]]))
    assert not cirq.allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[2]]))
    assert not cirq.allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[2]]))


def test_binary_sub_tensor_slice():
    a = slice(None)
    e = Ellipsis

    assert cirq.slice_for_qubits_equal_to([], 0) == (e,)
    assert cirq.slice_for_qubits_equal_to([0], 0b0) == (0, e)
    assert cirq.slice_for_qubits_equal_to([0], 0b1) == (1, e)
    assert cirq.slice_for_qubits_equal_to([1], 0b0) == (a, 0, e)
    assert cirq.slice_for_qubits_equal_to([1], 0b1) == (a, 1, e)
    assert cirq.slice_for_qubits_equal_to([2], 0b0) == (a, a, 0, e)
    assert cirq.slice_for_qubits_equal_to([2], 0b1) == (a, a, 1, e)

    assert cirq.slice_for_qubits_equal_to([0, 1], 0b00) == (0, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 2], 0b00) == (a, 0, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 3], 0b00) == (a, 0, a, 0, e)
    assert cirq.slice_for_qubits_equal_to([1, 3], 0b10) == (a, 0, a, 1, e)
    assert cirq.slice_for_qubits_equal_to([3, 1], 0b10) == (a, 1, a, 0, e)

    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 0b001) == (0, 0, 1, e)
    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 0b010) == (0, 1, 0, e)
    assert cirq.slice_for_qubits_equal_to([2, 1, 0], 0b100) == (1, 0, 0, e)
    assert cirq.slice_for_qubits_equal_to([0, 1, 2], 0b101) == (1, 0, 1, e)
    assert cirq.slice_for_qubits_equal_to([0, 2, 1], 0b101) == (1, 1, 0, e)

    m = np.array([0] * 16).reshape((2, 2, 2, 2))
    for k in range(16):
        m[cirq.slice_for_qubits_equal_to([3, 2, 1, 0], k)] = k
    assert list(m.reshape(16)) == list(range(16))
