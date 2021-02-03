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

"""Utility methods for diagonalizing matrices."""

from typing import Tuple, Callable, List

import numpy as np

from cirq.linalg import combinators, predicates, tolerance


def diagonalize_real_symmetric_matrix(
    matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes the given matrix.

    Args:
        matrix: A real symmetric matrix to diagonalize.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        check_preconditions: If set, verifies that the input matrix is real and
            symmetric.

    Returns:
        An orthogonal matrix P such that P.T @ matrix @ P is diagonal.

    Raises:
        ValueError: Matrix isn't real symmetric.
    """

    if check_preconditions and (
        np.any(np.imag(matrix) != 0) or not predicates.is_hermitian(matrix, rtol=rtol, atol=atol)
    ):
        raise ValueError('Input must be real and symmetric.')

    _, result = np.linalg.eigh(matrix)

    return result


def _contiguous_groups(
    length: int, comparator: Callable[[int, int], bool]
) -> List[Tuple[int, int]]:
    """Splits range(length) into approximate equivalence classes.

    Args:
        length: The length of the range to split.
        comparator: Determines if two indices have approximately equal items.

    Returns:
        A list of (inclusive_start, exclusive_end) range endpoints. Each
        corresponds to a run of approximately-equivalent items.
    """
    result = []
    start = 0
    while start < length:
        past = start + 1
        while past < length and comparator(start, past):
            past += 1
        result.append((start, past))
        start = past
    return result


def diagonalize_real_symmetric_and_sorted_diagonal_matrices(
    symmetric_matrix: np.ndarray,
    diagonal_matrix: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes both given matrices.

    The given matrices must commute.
    Guarantees that the sorted diagonal matrix is not permuted by the
    diagonalization (except for nearly-equal values).

    Args:
        symmetric_matrix: A real symmetric matrix.
        diagonal_matrix: A real diagonal matrix with entries along the diagonal
            sorted into descending order.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input matrices commute
            and are respectively symmetric and diagonal descending.

    Returns:
        An orthogonal matrix P such that P.T @ symmetric_matrix @ P is diagonal
        and P.T @ diagonal_matrix @ P = diagonal_matrix (up to tolerance).

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not symmetric).
    """

    # Verify preconditions.
    if check_preconditions:
        if np.any(np.imag(symmetric_matrix)) or not predicates.is_hermitian(
            symmetric_matrix, rtol=rtol, atol=atol
        ):
            raise ValueError('symmetric_matrix must be real symmetric.')
        if (
            not predicates.is_diagonal(diagonal_matrix, atol=atol)
            or np.any(np.imag(diagonal_matrix))
            or np.any(diagonal_matrix[:-1, :-1] < diagonal_matrix[1:, 1:])
        ):
            raise ValueError('diagonal_matrix must be real diagonal descending.')
        if not predicates.matrix_commutes(diagonal_matrix, symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('Given matrices must commute.')

    def similar_singular(i, j):
        return np.allclose(diagonal_matrix[i, i], diagonal_matrix[j, j], rtol=rtol)

    # Because the symmetric matrix commutes with the diagonal singulars matrix,
    # the symmetric matrix should be block-diagonal with a block boundary
    # wherever the singular values happen change. So we can use the singular
    # values to extract blocks that can be independently diagonalized.
    ranges = _contiguous_groups(diagonal_matrix.shape[0], similar_singular)

    # Build the overall diagonalization by diagonalizing each block.
    p = np.zeros(symmetric_matrix.shape, dtype=np.float64)
    for start, end in ranges:
        block = symmetric_matrix[start:end, start:end]
        p[start:end, start:end] = diagonalize_real_symmetric_matrix(
            block, rtol=rtol, atol=atol, check_preconditions=False
        )

    return p


def _svd_handling_empty(mat):
    if not mat.shape[0] * mat.shape[1]:
        z = np.zeros((0, 0), dtype=mat.dtype)
        return z, np.array([]), z

    return np.linalg.svd(mat)


def bidiagonalize_real_matrix_pair_with_symmetric_products(
    mat1: np.ndarray,
    mat2: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds orthogonal matrices that diagonalize both mat1 and mat2.

    Requires mat1 and mat2 to be real.
    Requires mat1.T @ mat2 to be symmetric.
    Requires mat1 @ mat2.T to be symmetric.

    Args:
        mat1: One of the real matrices.
        mat2: The other real matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the inputs are real, and that
            mat1.T @ mat2 and mat1 @ mat2.T are both symmetric. Defaults to set.

    Returns:
        A tuple (L, R) of two orthogonal matrices, such that both L @ mat1 @ R
        and L @ mat2 @ R are diagonal matrices.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if np.any(np.imag(mat1) != 0):
            raise ValueError('mat1 must be real.')
        if np.any(np.imag(mat2) != 0):
            raise ValueError('mat2 must be real.')
        if not predicates.is_hermitian(mat1.dot(mat2.T), rtol=rtol, atol=atol):
            raise ValueError('mat1 @ mat2.T must be symmetric.')
        if not predicates.is_hermitian(mat1.T.dot(mat2), rtol=rtol, atol=atol):
            raise ValueError('mat1.T @ mat2 must be symmetric.')

    # Use SVD to bi-diagonalize the first matrix.
    base_left, base_diag, base_right = _svd_handling_empty(np.real(mat1))
    base_diag = np.diag(base_diag)

    # Determine where we switch between diagonalization-fixup strategies.
    dim = base_diag.shape[0]
    rank = dim
    while rank > 0 and tolerance.all_near_zero(base_diag[rank - 1, rank - 1], atol=atol):
        rank -= 1
    base_diag = base_diag[:rank, :rank]

    # Try diagonalizing the second matrix with the same factors as the first.
    semi_corrected = base_left.T.dot(np.real(mat2)).dot(base_right.T)

    # Fix up the part of the second matrix's diagonalization that's matched
    # against non-zero diagonal entries in the first matrix's diagonalization
    # by performing simultaneous diagonalization.
    overlap = semi_corrected[:rank, :rank]
    overlap_adjust = diagonalize_real_symmetric_and_sorted_diagonal_matrices(
        overlap, base_diag, rtol=rtol, atol=atol, check_preconditions=check_preconditions
    )

    # Fix up the part of the second matrix's diagonalization that's matched
    # against zeros in the first matrix's diagonalization by performing an SVD.
    extra = semi_corrected[rank:, rank:]
    extra_left_adjust, _, extra_right_adjust = _svd_handling_empty(extra)

    # Merge the fixup factors into the initial diagonalization.
    left_adjust = combinators.block_diag(overlap_adjust, extra_left_adjust)
    right_adjust = combinators.block_diag(overlap_adjust.T, extra_right_adjust)
    left = left_adjust.T.dot(base_left.T)
    right = base_right.T.dot(right_adjust.T)

    return left, right


def bidiagonalize_unitary_with_special_orthogonals(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
) -> Tuple[np.ndarray, np.array, np.ndarray]:
    """Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

    Args:
        mat: A unitary matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input is a unitary matrix
            (to the given tolerances). Defaults to set.

    Returns:
        A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
        be orthogonal matrices with determinant equal to 1.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if not predicates.is_unitary(mat, rtol=rtol, atol=atol):
            raise ValueError('matrix must be unitary.')

    # Note: Because mat is unitary, setting A = real(mat) and B = imag(mat)
    # guarantees that both A @ B.T and A.T @ B are Hermitian.
    left, right = bidiagonalize_real_matrix_pair_with_symmetric_products(
        np.real(mat), np.imag(mat), rtol=rtol, atol=atol, check_preconditions=check_preconditions
    )

    # Convert to special orthogonal w/o breaking diagonalization.
    if np.linalg.det(left) < 0:
        left[0, :] *= -1
    if np.linalg.det(right) < 0:
        right[:, 0] *= -1

    diag = combinators.dot(left, mat, right)

    return left, np.diag(diag), right
