# coding=utf-8

# Copyright 2017 Google LLC
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


"""Utility methods for breaking matrices into useful pieces."""

import numpy as np
from typing import Tuple, Callable, List, TypeVar

from cirq.linalg import combinators
from cirq.linalg import diagonalize
from cirq.linalg import predicates
from cirq.linalg.tolerance import Tolerance

T = TypeVar('T')


def _group_similar(items: List[T],
                   comparer: Callable[[T, T], bool]) -> List[List[T]]:
    """Combines similar items into groups.

  Args:
    items: The list of items to group.
    comparer: Determines if two items are similar.

  Returns:
    A list of groups of items.
  """
    groups = []
    used = set()
    for i in range(len(items)):
        if i not in used:
            group = [i]
            for j in range(i + 1, len(items)):
                if j not in used and comparer(items[i], items[j]):
                    used.add(j)
                    group.append(j)
            groups.append(group)
    return groups


def _perp_eigendecompose(matrix: np.matrix, tolerance: Tolerance
                         ) -> Tuple[np.array, List[np.matrix]]:
    """An eigendecomposition that ensures eigenvectors are perpendicular.

    numpy.linalg.eig doesn't guarantee that eigenvectors from the same
    eigenspace will be perpendicular. This method uses Gram-Schmidt to recover
    a perpendicular set. It further checks that all eigenvectors are
    perpendicular and raises an ArithmeticError otherwise.

    Args:
        matrix: The matrix to decompose.
        tolerance: Thresholds for determining whether eigenvalues are from the
            same eigenspace and whether eigenvectors are perpendicular.

    Returns:
        The eigenvalues and column eigenvectors. The i'th eigenvalue is
        associated with the i'th column eigenvector.

    Raises:
        ArithmeticError: Failed to find perpendicular eigenvectors.
    """
    vals, cols = np.linalg.eig(matrix)
    vecs = [cols[:, i] for i in range(len(cols))]

    # Group by similar eigenvalue.
    n = len(vecs)
    groups = _group_similar(
        list(range(n)),
        lambda k1, k2: tolerance.all_close(vals[k1], vals[k2]))

    # Remove overlap between eigenvectors with the same eigenvalue.
    for g in groups:
        q, _ = np.linalg.qr(np.concatenate([vecs[i] for i in g], axis=1))
        for i in range(len(g)):
            vecs[g[i]] = q[:, i]

    # Ensure no eigenvectors overlap.
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if not tolerance.all_near_zero(vecs[i].H.dot(vecs[j])):
                raise ArithmeticError('Eigenvectors overlap.')

    return vals, vecs


def map_eigenvalues(
        matrix: np.matrix,
        func: Callable[[complex], complex],
        tolerance: Tolerance = Tolerance.DEFAULT
) -> np.matrix:
    """Applies a function to the eigenvalues of a matrix.

    Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

    Args:
        matrix: The matrix to modify with the function.
        func: The function to apply to the eigenvalues of the matrix.
        tolerance: Thresholds used when separating eigenspaces.

    Returns:
        The transformed matrix.
    """
    vals, vecs = _perp_eigendecompose(matrix, tolerance)
    pieces = [np.mat(np.outer(vec, vec.H)) for vec in vecs]
    out_vals = np.vectorize(func)(vals)

    total = np.zeros(shape=matrix.shape)
    for piece, val in zip(pieces, out_vals):
        total = np.add(total, piece * val)

    return np.mat(total)


def kron_factor_4x4_to_2x2s(
        matrix: np.matrix,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> Tuple[complex, np.matrix, np.matrix]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.

    Args:
        matrix: The 4x4 unitary matrix to factor.
        tolerance: Acceptable numeric error thresholds.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(
        ((i, j) for i in range(4) for j in range(4)),
        key=lambda t: abs(matrix[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.mat(np.zeros((2, 2), dtype=np.complex128))
    f2 = np.mat(np.zeros((2, 2), dtype=np.complex128))
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    f1 /= (np.sqrt(np.linalg.det(f1)) or 1)
    f2 /= (np.sqrt(np.linalg.det(f2)) or 1)

    # Determine global phase.
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    restored = g * combinators.kron(f1, f2)
    if np.any(np.isnan(restored)) or not tolerance.all_close(restored, matrix):
        raise ValueError("Can't factor into kronecker product.")

    return g, f1, f2


def so4_to_magic_su2s(
        mat: np.matrix,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> Tuple[np.matrix, np.matrix]:
    """Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

    Mag is the magic basis matrix:

        1  0  0  i
        0  i  1  0
        0  i -1  0     (times sqrt(0.5) to normalize)
        1  0  0 -i

    Args:
        mat: A real 4x4 orthogonal matrix.
        tolerance: Per-matrix-entry tolerance on equality.

    Returns:
        A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
        is approximately equal to the given matrix.

    Raises:
        ValueError: Bad matrix.
        ArithmeticError: Failed to perform the decomposition to desired
            tolerance.
        """
    if mat.shape != (4, 4) or not predicates.is_special_orthogonal(mat,
                                                                   tolerance):
        raise ValueError('mat must be 4x4 special orthogonal.')

    magic = np.mat([[1, 0, 0, 1j],
                    [0, 1j, 1, 0],
                    [0, 1j, -1, 0],
                    [1, 0, 0, -1j]]) * np.sqrt(0.5)
    ab = combinators.dot(magic, mat, magic.H)
    _, a, b = kron_factor_4x4_to_2x2s(ab, tolerance)

    # Check decomposition against desired tolerance.
    reconstructed = combinators.dot(magic.H, combinators.kron(a, b), magic)
    if not tolerance.all_close(reconstructed, mat):
        raise ArithmeticError('Failed to decompose to desired tolerance.')

    return a, b


def kak_canonicalize_vector(
        x: float, y: float, z: float
) -> Tuple[complex,
           Tuple[np.matrix, np.matrix],
           Tuple[float, float, float],
           Tuple[np.matrix, np.matrix]]:
    """Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

    Args:
        x: The strength of the XX interaction.
        y: The strength of the YY interaction.
        z: The strength of the ZZ interaction.

    Returns:
        A nested tuple (g, (a1, a0), (x2, y2, z2), (b1, b0)) containing:

            0. A global phase factor.
            1. Post-non-local-operation matrices for the second/first qubit.
            2. The canonicalized XX/YY/ZZ weights.
            3. Pre-non-local-operation matrices for the second/first qubit.

        Guarantees that the canonicalized x2, y2, z2 satisfy:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            z2 ≠ -π/4

        Guarantees that the implied output matrix:

            g · (a1 ⊗ a0) · exp(i·(x2·XX + y2·YY + z2·ZZ)) · (b1 ⊗ b0)

        is approximately equal to the implied input matrix:

            exp(i·(x·XX + y·YY + z·ZZ))
    """

    phase = [complex(1)]  # Accumulated global phase.
    left = [np.mat(np.eye(2))] * 2  # Per-qubit left factors.
    right = [np.mat(np.eye(2))] * 2  # Per-qubit right factors.
    v = [x, y, z]  # Remaining XX/YY/ZZ interaction vector.

    # These special-unitary matrices flip the X, Y, and Z axes respectively.
    flippers = [
        np.mat([[0, 1], [1, 0]]) * 1j,
        np.mat([[0, -1j], [1j, 0]]) * 1j,
        np.mat([[1, 0], [0, -1]]) * 1j
    ]

    # Each of these special-unitary matrices swaps two the roles of two axes.
    # The matrix at index k swaps the *other two* axes (e.g. swappers[1] is a
    # Hadamard operation that swaps X and Z).
    swappers = [
        np.mat([[1, -1j], [1j, -1]]) * 1j * np.sqrt(0.5),
        np.mat([[1, 1], [1, -1]]) * 1j * np.sqrt(0.5),
        np.mat([[0, 1 - 1j], [1 + 1j, 0]]) * 1j * np.sqrt(0.5)
    ]

    # Shifting strength by ½π is equivalent to local ops (e.g. exp(i½π XX)∝XX).
    def shift(k, step):
        v[k] += step * np.pi / 2
        phase[0] *= 1j**step
        right[0] = combinators.dot(flippers[k]**(step % 4), right[0])
        right[1] = combinators.dot(flippers[k]**(step % 4), right[1])

    # Two negations is equivalent to temporarily flipping along the other axis.
    def negate(k1, k2):
        v[k1] *= -1
        v[k2] *= -1
        phase[0] *= -1
        s = flippers[3 - k1 - k2]  # The other axis' flipper.
        left[1] = combinators.dot(left[1], s)
        right[1] = combinators.dot(s, right[1])

    # Swapping components is equivalent to temporarily swapping the two axes.
    def swap(k1, k2):
        v[k1], v[k2] = v[k2], v[k1]
        s = swappers[3 - k1 - k2]  # The other axis' swapper.
        left[0] = combinators.dot(left[0], s)
        left[1] = combinators.dot(left[1], s)
        right[0] = combinators.dot(s, right[0])
        right[1] = combinators.dot(s, right[1])

    # Shifts an axis strength into the range (-π/4, π/4].
    def canonical_shift(k):
        while v[k] <= -np.pi / 4:
            shift(k, +1)
        while v[k] > np.pi / 4:
            shift(k, -1)

    # Sorts axis strengths into descending order by absolute magnitude.
    def sort():
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)
        if abs(v[1]) < abs(v[2]):
            swap(1, 2)
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)

    # Get all strengths to (-¼π, ¼π] in descending order by absolute magnitude.
    canonical_shift(0)
    canonical_shift(1)
    canonical_shift(2)
    sort()

    # Move all negativity into z.
    if v[0] < 0:
        negate(0, 2)
    if v[1] < 0:
        negate(1, 2)
    canonical_shift(2)

    return (
        phase[0],
        (left[1], left[0]),
        (v[0], v[1], v[2]),
        (right[1], right[0]),
    )


def kak_decomposition(
        mat: np.matrix,
        tolerance: Tolerance = Tolerance.DEFAULT
) -> Tuple[complex,
           Tuple[np.matrix, np.matrix],
           Tuple[float, float, float],
           Tuple[np.matrix, np.matrix]]:
    """Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

    Args:
        mat: The 4x4 unitary matrix to decompose.
        tolerance: Per-matrix-entry tolerance on equality.

    Returns:
        A nested tuple (g, (a1, a0), (x, y, z), (b1, b0)) containing:

            0. A global phase factor.
            1. The pre-operation matrices to apply to the second/firs qubit.
            2. The XX/YY/ZZ weights of the non-local operation.
            3. The post-operation matrices to apply to the second/firs qubit.

        Guarantees that the x2, y2, z2 are canonicalized to satisfy:

            0 ≤ abs(z) ≤ y ≤ x ≤ π/4
            z ≠ -π/4

        Guarantees that the input matrix should approximately equal:

            g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

    Raises:
        ValueError: Bad matrix.
        ArithmeticError: Failed to perform the decomposition.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    magic = np.mat([[1, 0, 0, 1j],
                    [0, 1j, 1, 0],
                    [0, 1j, -1, 0],
                    [1, 0, 0, -1j]]) * np.sqrt(0.5)
    gamma = np.mat([[1, 1, 1, 1],
                    [1, 1, -1, -1],
                    [-1, 1, -1, 1],
                    [1, -1, -1, 1]]) * 0.25

    # Diagonalize in magic basis.
    left, d, right = (
        diagonalize.bidiagonalize_unitary_with_special_orthogonals(
            combinators.dot(magic.H, mat, magic),
            tolerance))

    # Recover pieces.
    a1, a0 = so4_to_magic_su2s(left.T, tolerance)
    b1, b0 = so4_to_magic_su2s(right.T, tolerance)
    w, x, y, z = gamma.dot(np.vstack(np.angle(d))).getA().flatten()
    g = np.exp(1j * w)

    # Canonicalize.
    g2, (c1, c0), (x2, y2, z2), (d1, d0) = kak_canonicalize_vector(x, y, z)
    return (
        g * g2,
        (a1.dot(c1), a0.dot(c0)),
        (x2, y2, z2),
        (d1.dot(b1), d0.dot(b0))
    )
