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
import pytest

import cirq
from cirq.linalg import matrix_commutes


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
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_diagonal(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_diagonal(np.array([[1, 0], [-0.6, 1]]), atol=atol)

    # Error isn't accumulated across entries.
    assert cirq.is_diagonal(np.array([[1, 0.5], [-0.5, 1]]), atol=atol)
    assert not cirq.is_diagonal(np.array([[1, 0.5], [-0.6, 1]]), atol=atol)


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

    assert cirq.is_hermitian(np.array([[1, 1j + 1e-11], [-1j, 1 + 1j * 1e-9]]))


def test_is_hermitian_tolerance():
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_hermitian(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert cirq.is_hermitian(np.array([[1, 0.25], [-0.25, 1]]), atol=atol)
    assert not cirq.is_hermitian(np.array([[1, 0], [-0.6, 1]]), atol=atol)
    assert not cirq.is_hermitian(np.array([[1, 0.25], [-0.35, 1]]), atol=atol)

    # Error isn't accumulated across entries.
    assert cirq.is_hermitian(np.array([[1, 0.5, 0.5], [0, 1, 0], [0, 0, 1]]), atol=atol)
    assert not cirq.is_hermitian(np.array([[1, 0.5, 0.6], [0, 1, 0], [0, 0, 1]]), atol=atol)
    assert not cirq.is_hermitian(np.array([[1, 0, 0.6], [0, 1, 0], [0, 0, 1]]), atol=atol)


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
    assert not cirq.is_unitary(np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_unitary(np.array([[1, 1j + 1e-11], [1j, 1 + 1j * 1e-9]]) * np.sqrt(0.5))


def test_is_unitary_tolerance():
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_unitary(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_unitary(np.array([[1, 0], [-0.6, 1]]), atol=atol)

    # Error isn't accumulated across entries.
    assert cirq.is_unitary(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), atol=atol)
    assert not cirq.is_unitary(np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1.2]]), atol=atol)


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
    assert not cirq.is_orthogonal(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_orthogonal(np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_orthogonal(np.array([[1, 1e-11], [0, 1 + 1e-11]]))


def test_is_orthogonal_tolerance():
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_orthogonal(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_orthogonal(np.array([[1, 0], [-0.6, 1]]), atol=atol)

    # Error isn't accumulated across entries.
    assert cirq.is_orthogonal(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), atol=atol)
    assert not cirq.is_orthogonal(np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1.2]]), atol=atol)


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
    assert cirq.is_special_orthogonal(np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(np.array([[1, 1], [1, -1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_orthogonal(np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_special_orthogonal(np.array([[1, 1e-11], [0, 1 + 1e-11]]))


def test_is_special_orthogonal_tolerance():
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_special_orthogonal(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_special_orthogonal(np.array([[1, 0], [-0.6, 1]]), atol=atol)

    # Error isn't accumulated across entries, except for determinant factors.
    assert cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1 / 1.2]]), atol=atol
    )
    assert not cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), atol=atol
    )
    assert not cirq.is_special_orthogonal(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1 / 1.2]]), atol=atol
    )


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
    assert cirq.is_special_unitary(np.array([[1, -1], [1, 1]]) * np.sqrt(0.5))
    assert cirq.is_special_unitary(np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert not cirq.is_special_unitary(np.array([[1, -1j], [1j, 1]]) * np.sqrt(0.5))

    assert cirq.is_special_unitary(np.array([[1, 1j + 1e-11], [1j, 1 + 1j * 1e-9]]) * np.sqrt(0.5))


def test_is_special_unitary_tolerance():
    atol = 0.5

    # Pays attention to specified tolerance.
    assert cirq.is_special_unitary(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_special_unitary(np.array([[1, 0], [-0.6, 1]]), atol=atol)
    assert cirq.is_special_unitary(np.array([[1, 0], [0, 1]]) * cmath.exp(1j * 0.1), atol=atol)
    assert not cirq.is_special_unitary(np.array([[1, 0], [0, 1]]) * cmath.exp(1j * 0.3), atol=atol)

    # Error isn't accumulated across entries, except for determinant factors.
    assert cirq.is_special_unitary(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1 / 1.2]]), atol=atol)
    assert not cirq.is_special_unitary(np.array([[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), atol=atol)
    assert not cirq.is_special_unitary(
        np.array([[1.2, 0, 0], [0, 1.3, 0], [0, 0, 1 / 1.2]]), atol=atol
    )


def test_is_normal():
    assert cirq.is_normal(np.array([[1]]))
    assert cirq.is_normal(np.array([[3j]]))
    assert cirq.is_normal(cirq.testing.random_density_matrix(4))
    assert cirq.is_normal(cirq.testing.random_unitary(5))
    assert not cirq.is_normal(np.array([[0, 1], [0, 0]]))
    assert not cirq.is_normal(np.zeros((1, 0)))


def test_is_normal_tolerance():
    atol = 0.25

    # Pays attention to specified tolerance.
    assert cirq.is_normal(np.array([[0, 0.5], [0, 0]]), atol=atol)
    assert not cirq.is_normal(np.array([[0, 0.6], [0, 0]]), atol=atol)

    # Error isn't accumulated across entries.
    assert cirq.is_normal(np.array([[0, 0.5, 0], [0, 0, 0.5], [0, 0, 0]]), atol=atol)
    assert not cirq.is_normal(np.array([[0, 0.5, 0], [0, 0, 0.6], [0, 0, 0]]), atol=atol)


def test_is_cptp():
    rt2 = np.sqrt(0.5)
    # Amplitude damping with gamma=0.5.
    assert cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, rt2]]), np.array([[0, rt2], [0, 0]])])
    # Depolarizing channel with p=0.75.
    assert cirq.is_cptp(
        kraus_ops=[
            np.array([[1, 0], [0, 1]]) * 0.5,
            np.array([[0, 1], [1, 0]]) * 0.5,
            np.array([[0, -1j], [1j, 0]]) * 0.5,
            np.array([[1, 0], [0, -1]]) * 0.5,
        ]
    )

    assert not cirq.is_cptp(kraus_ops=[np.array([[1, 0], [0, 1]]), np.array([[0, 1], [0, 0]])])
    assert not cirq.is_cptp(
        kraus_ops=[
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
    )

    # Makes 4 2x2 kraus ops.
    one_qubit_u = cirq.testing.random_unitary(8)
    one_qubit_kraus = np.reshape(one_qubit_u[:, :2], (-1, 2, 2))
    assert cirq.is_cptp(kraus_ops=one_qubit_kraus)

    # Makes 16 4x4 kraus ops.
    two_qubit_u = cirq.testing.random_unitary(64)
    two_qubit_kraus = np.reshape(two_qubit_u[:, :4], (-1, 4, 4))
    assert cirq.is_cptp(kraus_ops=two_qubit_kraus)


def test_is_cptp_tolerance():
    rt2_ish = np.sqrt(0.5) - 0.01
    atol = 0.25
    # Moderately-incorrect amplitude damping with gamma=0.5.
    assert cirq.is_cptp(
        kraus_ops=[np.array([[1, 0], [0, rt2_ish]]), np.array([[0, rt2_ish], [0, 0]])], atol=atol
    )
    assert not cirq.is_cptp(
        kraus_ops=[np.array([[1, 0], [0, rt2_ish]]), np.array([[0, rt2_ish], [0, 0]])], atol=1e-8
    )


def test_commutes():
    assert matrix_commutes(np.empty((0, 0)), np.empty((0, 0)))
    assert not matrix_commutes(np.empty((1, 0)), np.empty((0, 1)))
    assert not matrix_commutes(np.empty((0, 1)), np.empty((1, 0)))
    assert not matrix_commutes(np.empty((1, 0)), np.empty((1, 0)))
    assert not matrix_commutes(np.empty((0, 1)), np.empty((0, 1)))

    assert matrix_commutes(np.array([[1]]), np.array([[2]]))
    assert matrix_commutes(np.array([[1]]), np.array([[0]]))

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    xx = np.kron(x, x)
    zz = np.kron(z, z)

    assert matrix_commutes(x, x)
    assert matrix_commutes(y, y)
    assert matrix_commutes(z, z)
    assert not matrix_commutes(x, y)
    assert not matrix_commutes(x, z)
    assert not matrix_commutes(y, z)

    assert matrix_commutes(xx, zz)
    assert matrix_commutes(xx, np.diag([1, -1, -1, 1 + 1e-9]))


def test_commutes_tolerance():
    atol = 0.5

    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1, 0], [0, -1]])

    # Pays attention to specified tolerance.
    assert matrix_commutes(x, x + z * 0.1, atol=atol)
    assert not matrix_commutes(x, x + z * 0.5, atol=atol)


def test_allclose_up_to_global_phase():
    assert cirq.allclose_up_to_global_phase(np.array([1]), np.array([1j]))

    assert not cirq.allclose_up_to_global_phase(np.array([[[1]]]), np.array([1]))

    assert cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[1]]))
    assert cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[-1]]))

    assert cirq.allclose_up_to_global_phase(np.array([[0]]), np.array([[0]]))

    assert cirq.allclose_up_to_global_phase(np.array([[1, 2]]), np.array([[1j, 2j]]))

    assert cirq.allclose_up_to_global_phase(np.array([[1, 2.0000000001]]), np.array([[1j, 2j]]))

    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[1, 0]]))
    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]))
    assert not cirq.allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]))


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

    assert cirq.slice_for_qubits_equal_to([0], 0b1, num_qubits=1) == (1,)
    assert cirq.slice_for_qubits_equal_to([1], 0b0, num_qubits=2) == (a, 0)
    assert cirq.slice_for_qubits_equal_to([1], 0b0, num_qubits=3) == (a, 0, a)
    assert cirq.slice_for_qubits_equal_to([2], 0b0, num_qubits=3) == (a, a, 0)


def test_binary_sub_tensor_slice_big_endian():
    a = slice(None)
    e = Ellipsis
    sfqet = cirq.slice_for_qubits_equal_to

    assert sfqet([], big_endian_qureg_value=0) == (e,)
    assert sfqet([0], big_endian_qureg_value=0b0) == (0, e)
    assert sfqet([0], big_endian_qureg_value=0b1) == (1, e)
    assert sfqet([1], big_endian_qureg_value=0b0) == (a, 0, e)
    assert sfqet([1], big_endian_qureg_value=0b1) == (a, 1, e)
    assert sfqet([2], big_endian_qureg_value=0b0) == (a, a, 0, e)
    assert sfqet([2], big_endian_qureg_value=0b1) == (a, a, 1, e)

    assert sfqet([0, 1], big_endian_qureg_value=0b00) == (0, 0, e)
    assert sfqet([1, 2], big_endian_qureg_value=0b00) == (a, 0, 0, e)
    assert sfqet([1, 3], big_endian_qureg_value=0b00) == (a, 0, a, 0, e)
    assert sfqet([1, 3], big_endian_qureg_value=0b01) == (a, 0, a, 1, e)
    assert sfqet([3, 1], big_endian_qureg_value=0b01) == (a, 1, a, 0, e)

    assert sfqet([2, 1, 0], big_endian_qureg_value=0b100) == (0, 0, 1, e)
    assert sfqet([2, 1, 0], big_endian_qureg_value=0b010) == (0, 1, 0, e)
    assert sfqet([2, 1, 0], big_endian_qureg_value=0b001) == (1, 0, 0, e)
    assert sfqet([0, 1, 2], big_endian_qureg_value=0b101) == (1, 0, 1, e)
    assert sfqet([0, 2, 1], big_endian_qureg_value=0b101) == (1, 1, 0, e)

    m = np.array([0] * 16).reshape((2, 2, 2, 2))
    for k in range(16):
        m[sfqet([0, 1, 2, 3], big_endian_qureg_value=k)] = k
    assert list(m.reshape(16)) == list(range(16))

    assert sfqet([0], big_endian_qureg_value=0b1, num_qubits=1) == (1,)
    assert sfqet([1], big_endian_qureg_value=0b0, num_qubits=2) == (a, 0)
    assert sfqet([1], big_endian_qureg_value=0b0, num_qubits=3) == (a, 0, a)
    assert sfqet([2], big_endian_qureg_value=0b0, num_qubits=3) == (a, a, 0)


def test_qudit_sub_tensor_slice():
    a = slice(None)
    sfqet = cirq.slice_for_qubits_equal_to

    assert sfqet([], 0, qid_shape=()) == ()
    assert sfqet([0], 0, qid_shape=(3,)) == (0,)
    assert sfqet([0], 1, qid_shape=(3,)) == (1,)
    assert sfqet([0], 2, qid_shape=(3,)) == (2,)
    assert sfqet([2], 0, qid_shape=(1, 2, 3)) == (a, a, 0)
    assert sfqet([2], 2, qid_shape=(1, 2, 3)) == (a, a, 2)
    assert sfqet([2], big_endian_qureg_value=2, qid_shape=(1, 2, 3)) == (a, a, 2)

    assert sfqet([1, 3], 3 * 2 + 1, qid_shape=(2, 3, 4, 5)) == (a, 1, a, 2)
    assert sfqet([3, 1], 5 * 2 + 1, qid_shape=(2, 3, 4, 5)) == (a, 2, a, 1)
    assert sfqet([2, 1, 0], 9 * 2 + 3 * 1, qid_shape=(3,) * 3) == (2, 1, 0)
    assert sfqet([1, 3], big_endian_qureg_value=5 * 1 + 2, qid_shape=(2, 3, 4, 5)) == (a, 1, a, 2)
    assert sfqet([3, 1], big_endian_qureg_value=3 * 1 + 2, qid_shape=(2, 3, 4, 5)) == (a, 2, a, 1)

    m = np.array([0] * 24).reshape((1, 2, 3, 4))
    for k in range(24):
        m[sfqet([3, 2, 1, 0], k, qid_shape=(1, 2, 3, 4))] = k
    assert list(m.reshape(24)) == list(range(24))

    assert sfqet([0], 1, num_qubits=1, qid_shape=(3,)) == (1,)
    assert sfqet([1], 0, num_qubits=3, qid_shape=(3, 3, 3)) == (a, 0, a)

    with pytest.raises(ValueError, match='len.* !='):
        sfqet([], num_qubits=2, qid_shape=(1, 2, 3))

    with pytest.raises(ValueError, match='exactly one'):
        sfqet([0, 1, 2], 0b101, big_endian_qureg_value=0b101)
