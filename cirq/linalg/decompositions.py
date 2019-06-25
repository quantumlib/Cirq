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


"""Utility methods for breaking matrices into useful pieces."""

from typing import Set, NamedTuple, Union  # pylint: disable=unused-import
from typing import Callable, List, Tuple, TypeVar

import math
import cmath
import numpy as np

from cirq import value
from cirq._compat import proper_repr
from cirq.linalg import combinators, diagonalize, predicates


T = TypeVar('T')
MAGIC = np.array([[1, 0, 0, 1j],
                  [0, 1j, 1, 0],
                  [0, 1j, -1, 0],
                  [1, 0, 0, -1j]]) * np.sqrt(0.5)
MAGIC_CONJ_T = np.conj(MAGIC.T)


def _phase_matrix(angle: float) -> np.ndarray:
    return np.diag([1, np.exp(1j * angle)])


def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def deconstruct_single_qubit_matrix_into_angles(
        mat: np.ndarray) -> Tuple[float, float, float]:
    """Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

    Args:
        mat: The 2x2 unitary matrix to break down.

    Returns:
        A tuple containing the amount to phase around Z, then rotate around Y,
        then phase around Z (all in radians).
    """
    # Anti-cancel left-vs-right phase along top row.
    right_phase = cmath.phase(mat[0, 1] * np.conj(mat[0, 0])) + math.pi
    mat = np.dot(mat, _phase_matrix(-right_phase))

    # Cancel top-vs-bottom phase along left column.
    bottom_phase = cmath.phase(mat[1, 0] * np.conj(mat[0, 0]))
    mat = np.dot(_phase_matrix(-bottom_phase), mat)

    # Lined up for a rotation. Clear the off-diagonal cells with one.
    rotation = math.atan2(abs(mat[1, 0]), abs(mat[0, 0]))
    mat = np.dot(_rotation_matrix(-rotation), mat)

    # Cancel top-left-vs-bottom-right phase.
    diagonal_phase = cmath.phase(mat[1, 1] * np.conj(mat[0, 0]))

    # Note: Ignoring global phase.
    return right_phase + diagonal_phase, rotation * 2, bottom_phase


def _group_similar(items: List[T],
                   comparer: Callable[[T, T], bool]) -> List[List[T]]:
    """Combines similar items into groups.

  Args:
    items: The list of items to group.
    comparer: Determines if two items are similar.

  Returns:
    A list of groups of items.
  """
    groups = []  # type: List[List[T]]
    used = set()  # type: Set[int]
    for i in range(len(items)):
        if i not in used:
            group = [items[i]]
            for j in range(i + 1, len(items)):
                if j not in used and comparer(items[i], items[j]):
                    used.add(j)
                    group.append(items[j])
            groups.append(group)
    return groups


def _perp_eigendecompose(matrix: np.ndarray,
                         rtol: float = 1e-5,
                         atol: float = 1e-8,
                         ) -> Tuple[np.array, List[np.ndarray]]:
    """An eigendecomposition that ensures eigenvectors are perpendicular.

    numpy.linalg.eig doesn't guarantee that eigenvectors from the same
    eigenspace will be perpendicular. This method uses Gram-Schmidt to recover
    a perpendicular set. It further checks that all eigenvectors are
    perpendicular and raises an ArithmeticError otherwise.

    Args:
        matrix: The matrix to decompose.
        rtol: Relative threshold for determining whether eigenvalues are from
              the same eigenspace and whether eigenvectors are perpendicular.
        atol: Absolute threshold for determining whether eigenvalues are from
              the same eigenspace and whether eigenvectors are perpendicular.

    Returns:
        The eigenvalues and column eigenvectors. The i'th eigenvalue is
        associated with the i'th column eigenvector.

    Raises:
        ArithmeticError: Failed to find perpendicular eigenvectors.
    """
    vals, cols = np.linalg.eig(matrix)
    vecs = [cols[:, i] for i in range(len(cols))]

    # Convert list of row arrays to list of column arrays.
    for i in range(len(vecs)):
        vecs[i] = np.reshape(vecs[i], (len(vecs[i]), vecs[i].ndim))

    # Group by similar eigenvalue.
    n = len(vecs)
    groups = _group_similar(
        list(range(n)),
        lambda k1, k2: np.allclose(vals[k1], vals[k2], rtol=rtol))

    # Remove overlap between eigenvectors with the same eigenvalue.
    for g in groups:
        q, _ = np.linalg.qr(np.hstack([vecs[i] for i in g]))
        for i in range(len(g)):
            vecs[g[i]] = q[:, i]

    return vals, vecs


def map_eigenvalues(
        matrix: np.ndarray,
        func: Callable[[complex], complex],
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8) -> np.ndarray:
    """Applies a function to the eigenvalues of a matrix.

    Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

    Args:
        matrix: The matrix to modify with the function.
        func: The function to apply to the eigenvalues of the matrix.
        rtol: Relative threshold used when separating eigenspaces.
        atol: Absolute threshold used when separating eigenspaces.

    Returns:
        The transformed matrix.
    """
    vals, vecs = _perp_eigendecompose(matrix,
                                      rtol=rtol,
                                      atol=atol)
    pieces = [np.outer(vec, np.conj(vec.T)) for vec in vecs]
    out_vals = np.vectorize(func)(vals.astype(complex))

    total = np.zeros(shape=matrix.shape)
    for piece, val in zip(pieces, out_vals):
        total = np.add(total, piece * val)
    return total


def kron_factor_4x4_to_2x2s(
        matrix: np.ndarray,
) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.
    Giving an incorrect matrix will cause garbage output.

    Args:
        matrix: The 4x4 unitary matrix to factor.

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
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
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

    return g, f1, f2


def so4_to_magic_su2s(
        mat: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        check_preconditions: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

    Mag is the magic basis matrix:

        1  0  0  i
        0  i  1  0
        0  i -1  0     (times sqrt(0.5) to normalize)
        1  0  0 -i

    Args:
        mat: A real 4x4 orthogonal matrix.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: When set, the code verifies that the given
            matrix is from SO(4). Defaults to set.

    Returns:
        A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
        is approximately equal to the given matrix.

    Raises:
        ValueError: Bad matrix.
        """
    if check_preconditions:
        if mat.shape != (4, 4) or not predicates.is_special_orthogonal(
                mat, atol=atol, rtol=rtol):
            raise ValueError('mat must be 4x4 special orthogonal.')

    ab = combinators.dot(MAGIC, mat, MAGIC_CONJ_T)
    _, a, b = kron_factor_4x4_to_2x2s(ab)

    return a, b


@value.value_equality(approximate=True)
class AxisAngleDecomposition:
    """Represents a unitary operation as an axis, angle, and global phase.

    The unitary $U$ is decomposed as follows:

        $$U = g e^{-i \theta/2 (xX + yY + zZ)}$$

    where \theta is the rotation angle, (x, y, z) is a unit vector along the
    rotation axis, and g is the global phase.
    """

    def __init__(self, *, angle: float, axis: Tuple[float, float, float],
                 global_phase: Union[int, float, complex]):
        if not np.isclose(np.linalg.norm(axis, 2), 1, atol=1e-8):
            raise ValueError('Axis vector must be normalized.')
        self.global_phase = complex(global_phase)
        self.axis = tuple(axis)
        self.angle = float(angle)

    def canonicalize(self, atol: float = 1e-8) -> 'AxisAngleDecomposition':
        """Returns a standardized AxisAngleDecomposition with the same unitary.

        Ensures the axis (x, y, z) satisfies x+y+z >= 0.
        Ensures the angle theta satisfies -pi + atol < theta <= pi + atol.

        Args:
            atol: Absolute tolerance for errors in the representation and the
                canonicalization. Determines how much larger a value needs to
                be than pi before it wraps into the negative range (so that
                approximation errors less than the tolerance do not cause sign
                instabilities).

        Returns:
            The canonicalized AxisAngleDecomposition.
        """
        assert 0 <= atol < np.pi

        angle = self.angle
        x, y, z = self.axis
        p = self.global_phase

        # Prefer axes that point positive-ward.
        if x + y + z < 0:
            x = -x
            y = -y
            z = -z
            angle = -angle

        # Prefer angle in (-π, π].
        if abs(angle) >= np.pi * 2:
            angle %= np.pi * 4
        while angle <= -np.pi + atol:
            angle += np.pi * 2
            p = -p
        while angle > np.pi + atol:
            angle -= np.pi * 2
            p = -p

        return AxisAngleDecomposition(axis=(x, y, z),
                                      angle=angle,
                                      global_phase=p)

    def _value_equality_values_(self):
        v = self.canonicalize(atol=0)
        return (value.PeriodicValue(v.angle,
                                    period=math.pi * 2), v.axis, v.global_phase)

    def _unitary_(self):
        x, y, z = self.axis
        xm = np.array([[0, 1], [1, 0]])
        ym = np.array([[0, -1j], [1j, 0]])
        zm = np.diag([1, -1])
        i = np.eye(2)
        c = math.cos(-self.angle / 2)
        s = math.sin(-self.angle / 2)
        return (c * i + 1j * s * (x * xm + y * ym + z * zm)) * self.global_phase

    def __str__(self):
        axis_terms = '+'.join('{:.3g}*{}'.format(e, a) if e < 0.9999 else a
                              for e, a in zip(self.axis, ['X', 'Y', 'Z'])
                              if abs(e) >= 1e-8).replace('+-', '-')
        return '{:.3g}*π around {}'.format(
            self.angle / np.pi,
            axis_terms,
        )

    def __repr__(self):
        return ('cirq.AxisAngleDecomposition('
                'angle={!r}, axis={!r}, global_phase={!r})'.format(
                    self.angle, self.axis, self.global_phase))


def axis_angle(single_qubit_unitary: np.ndarray) -> AxisAngleDecomposition:
    """Decomposes a single-qubit unitary into axis, angle, and global phase.

    Args:
        single_qubit_unitary: The unitary of the single-qubit operation to
            decompose.

    Returns:
        An AxisAngleDecomposition equivalent to the given unitary.
    """
    u = single_qubit_unitary
    assert u.shape == (2, 2)
    assert predicates.is_unitary(single_qubit_unitary, atol=1e-8)

    # Extract phased quaternion components.
    [a, b], [c, d] = u
    wp = (a + d) / 2
    xp = (b + c) / 2j
    yp = (b - c) / 2
    zp = (a - d) / 2j

    # Extract global phase factor from largest component.
    p = max(wp, xp, yp, zp, key=abs)
    p /= abs(p)

    # Cancel global phase factor, pushing components onto the real line.
    w = min(1, max(-1, np.real(wp / p)))
    x = np.real(xp / p)
    y = np.real(yp / p)
    z = np.real(zp / p)
    angle = -2 * math.acos(w)

    # Normalize axis.
    n = math.sqrt(x * x + y * y + z * z)
    if n < 0.0000001:
        # There's an axis singularity near θ=0.
        # Default to no rotation around the X axis.
        return AxisAngleDecomposition(global_phase=p, angle=0, axis=(1, 0, 0))
    x /= n
    y /= n
    z /= n

    return AxisAngleDecomposition(axis=(x, y, z), angle=angle,
                                  global_phase=p).canonicalize()


@value.value_equality
class KakDecomposition:
    """A convenient description of an arbitrary two-qubit operation.

    Any two qubit operation U can be decomposed into the form

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

    This class stores g, (b0, b1), (x, y, z), and (a0, a1).

    Attributes:
        global_phase: g from the above equation.
        single_qubit_operations_before: b0, b1 from the above equation.
        interaction_coefficients: x, y, z from the above equation.
        single_qubit_operations_after: a0, a1 from the above equation.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """

    def __init__(self,
                 *,
                 global_phase: complex,
                 single_qubit_operations_before: Tuple[np.ndarray, np.ndarray],
                 interaction_coefficients: Tuple[float, float, float],
                 single_qubit_operations_after: Tuple[np.ndarray, np.ndarray]):
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            interaction_coefficients: x, y, z from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase = global_phase
        self.single_qubit_operations_before = single_qubit_operations_before
        self.interaction_coefficients = interaction_coefficients
        self.single_qubit_operations_after = single_qubit_operations_after

    def _value_equality_values_(self):
        def flatten(x):
            return tuple(tuple(e.flat) for e in x)

        return (type(KakDecomposition),
                self.global_phase,
                tuple(self.interaction_coefficients),
                flatten(self.single_qubit_operations_before),
                flatten(self.single_qubit_operations_after))

    def __str__(self):
        return ('KAK {{\n'
                '    xyz*(4/π): {:.3g}, {:.3g}, {:.3g}\n'
                '    before: ({}) ⊗ ({})\n'
                '    after: ({}) ⊗ ({})\n'
                '}}').format(self.interaction_coefficients[0] * 4 / np.pi,
                             self.interaction_coefficients[1] * 4 / np.pi,
                             self.interaction_coefficients[2] * 4 / np.pi,
                             axis_angle(self.single_qubit_operations_before[0]),
                             axis_angle(self.single_qubit_operations_before[1]),
                             axis_angle(self.single_qubit_operations_after[0]),
                             axis_angle(self.single_qubit_operations_after[1]))

    def __repr__(self):
        return (
            'cirq.KakDecomposition(\n'
            '    interaction_coefficients={!r},\n'
            '    single_qubit_operations_before=(\n'
            '        {},\n'
            '        {},\n'
            '    ),\n'
            '    single_qubit_operations_after=(\n'
            '        {},\n'
            '        {},\n'
            '    ),\n'
            '    global_phase={!r})'
        ).format(
            self.interaction_coefficients,
            proper_repr(self.single_qubit_operations_before[0]),
            proper_repr(self.single_qubit_operations_before[1]),
            proper_repr(self.single_qubit_operations_after[0]),
            proper_repr(self.single_qubit_operations_after[1]),
            self.global_phase,
        )

    def _unitary_(self):
        """Returns the decomposition's two-qubit unitary matrix.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)
        """
        before = np.kron(*self.single_qubit_operations_before)
        after = np.kron(*self.single_qubit_operations_after)

        def interaction_matrix(m: np.ndarray, c: float) -> np.ndarray:
            return map_eigenvalues(np.kron(m, m),
                                   lambda v: np.exp(1j * v * c))

        x, y, z = self.interaction_coefficients
        x_mat = np.array([[0, 1], [1, 0]])
        y_mat = np.array([[0, -1j], [1j, 0]])
        z_mat = np.array([[1, 0], [0, -1]])

        return self.global_phase * combinators.dot(
            after,
            interaction_matrix(z_mat, z),
            interaction_matrix(y_mat, y),
            interaction_matrix(x_mat, x),
            before)


def kak_canonicalize_vector(x: float, y: float, z: float,
                            atol: float = 1e-9) -> KakDecomposition:
    """Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

    Args:
        x: The strength of the XX interaction.
        y: The strength of the YY interaction.
        z: The strength of the ZZ interaction.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

        Guarantees that the implied output matrix:

            g · (a1 ⊗ a0) · exp(i·(x2·XX + y2·YY + z2·ZZ)) · (b1 ⊗ b0)

        is approximately equal to the implied input matrix:

            exp(i·(x·XX + y·YY + z·ZZ))
    """

    phase = [complex(1)]  # Accumulated global phase.
    left = [np.eye(2)] * 2  # Per-qubit left factors.
    right = [np.eye(2)] * 2  # Per-qubit right factors.
    v = [x, y, z]  # Remaining XX/YY/ZZ interaction vector.

    # These special-unitary matrices flip the X, Y, and Z axes respectively.
    flippers = [
        np.array([[0, 1], [1, 0]]) * 1j,
        np.array([[0, -1j], [1j, 0]]) * 1j,
        np.array([[1, 0], [0, -1]]) * 1j
    ]

    # Each of these special-unitary matrices swaps two the roles of two axes.
    # The matrix at index k swaps the *other two* axes (e.g. swappers[1] is a
    # Hadamard operation that swaps X and Z).
    swappers = [
        np.array([[1, -1j], [1j, -1]]) * 1j * np.sqrt(0.5),
        np.array([[1, 1], [1, -1]]) * 1j * np.sqrt(0.5),
        np.array([[0, 1 - 1j], [1 + 1j, 0]]) * 1j * np.sqrt(0.5)
    ]

    # Shifting strength by ½π is equivalent to local ops (e.g. exp(i½π XX)∝XX).
    def shift(k, step):
        v[k] += step * np.pi / 2
        phase[0] *= 1j ** step
        right[0] = combinators.dot(flippers[k] ** (step % 4), right[0])
        right[1] = combinators.dot(flippers[k] ** (step % 4), right[1])

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

    # If x = π/4, force z to be positive
    if v[0] > np.pi / 4 - atol and v[2] < 0:
        shift(0, -1)
        negate(0, 2)

    return KakDecomposition(
        global_phase=phase[0],
        single_qubit_operations_after=(left[1], left[0]),
        interaction_coefficients=(v[0], v[1], v[2]),
        single_qubit_operations_before=(right[1], right[0]))


def kak_decomposition(
        mat: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8) -> KakDecomposition:
    """Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

    Args:
        mat: The 4x4 unitary matrix to decompose.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.

    Returns:
        A `cirq.KakDecomposition` canonicalized such that the interaction
        coefficients x, y, z satisfy:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

    Raises:
        ValueError: Bad matrix.
        ArithmeticError: Failed to perform the decomposition.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    magic = np.array([[1, 0, 0, 1j],
                      [0, 1j, 1, 0],
                      [0, 1j, -1, 0],
                      [1, 0, 0, -1j]]) * np.sqrt(0.5)
    gamma = np.array([[1, 1, 1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, 1],
                      [1, -1, -1, 1]]) * 0.25

    # Diagonalize in magic basis.
    left, d, right = diagonalize.bidiagonalize_unitary_with_special_orthogonals(
        combinators.dot(np.conj(magic.T), mat, magic),
        atol=atol,
        rtol=rtol,
        check_preconditions=False)

    # Recover pieces.
    a1, a0 = so4_to_magic_su2s(left.T,
                               atol=atol,
                               rtol=rtol,
                               check_preconditions=False)
    b1, b0 = so4_to_magic_su2s(right.T,
                               atol=atol,
                               rtol=rtol,
                               check_preconditions=False)
    w, x, y, z = gamma.dot(np.vstack(np.angle(d))).flatten()
    g = np.exp(1j * w)

    # Canonicalize.
    inner_cannon = kak_canonicalize_vector(x, y, z)

    b1 = np.dot(inner_cannon.single_qubit_operations_before[0], b1)
    b0 = np.dot(inner_cannon.single_qubit_operations_before[1], b0)
    a1 = np.dot(a1, inner_cannon.single_qubit_operations_after[0])
    a0 = np.dot(a0, inner_cannon.single_qubit_operations_after[1])
    return KakDecomposition(
        interaction_coefficients=inner_cannon.interaction_coefficients,
        global_phase=g * inner_cannon.global_phase,
        single_qubit_operations_before=(b1, b0),
        single_qubit_operations_after=(a1, a0))
