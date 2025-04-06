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

import cmath
import math
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np

# this is for older systems with matplotlib <3.2 otherwise 3d projections fail
from mpl_toolkits import mplot3d

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations

linalg = LazyLoader("linalg", globals(), "scipy.linalg")


if TYPE_CHECKING:
    import cirq

T = TypeVar('T')
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) * np.sqrt(0.5)

MAGIC_CONJ_T = np.conj(MAGIC.T)

# yapf: disable
YY = np.array([[0, 0, 0, -1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [-1, 0, 0, 0]])
# yapf: enable


def _phase_matrix(angle: float) -> np.ndarray:
    return np.diag([1, np.exp(1j * angle)])


def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def deconstruct_single_qubit_matrix_into_angles(mat: np.ndarray) -> Tuple[float, float, float]:
    r"""Breaks down a 2x2 unitary into ZYZ angle parameters.

    Given a unitary U, this function returns three angles: $\phi_0, \phi_1, \phi_2$,
    such that:  $U = Z^{\phi_2 / \pi} Y^{\phi_1 / \pi} Z^{\phi_0/ \pi}$
    for the Pauli matrices Y and Z.  That is, phasing around Z by $\phi_0$ radians,
    then rotating around Y by $\phi_1$ radians, and then phasing again by
    $\phi_2$ radians will produce the same effect as the original unitary.
    (Note that the matrices are applied right to left.)

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


def unitary_eig(
    matrix: np.ndarray, check_preconditions: bool = True, atol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Gives the guaranteed unitary eigendecomposition of a normal matrix.

    All hermitian and unitary matrices are normal matrices. This method was
    introduced as for certain classes of unitary matrices (where the eigenvalues
    are close to each other) the eigenvectors returned by `numpy.linalg.eig` are
    not guaranteed to be orthogonal.
    For more information, see https://github.com/numpy/numpy/issues/15461.

    Args:
        matrix: A normal matrix. If not normal, this method is not
            guaranteed to return correct eigenvalues.  A normal matrix
            is one where $A A^\dagger = A^\dagger A$.
        check_preconditions: When true and matrix is not unitary,
            a `ValueError` is raised when the matrix is not normal.
        atol: The absolute tolerance when checking whether the original matrix
            was unitary.

    Returns:
        A Tuple of
            eigvals: The eigenvalues of `matrix`.
            V: The unitary matrix with the eigenvectors as columns.

    Raises:
        ValueError: if the input matrix is not normal.
    """
    if check_preconditions and not predicates.is_normal(matrix, atol=atol):
        raise ValueError(f'Input must correspond to a normal matrix .Received input:\n{matrix}')

    R, V = linalg.schur(matrix, output="complex")
    return R.diagonal(), V


# pylint: enable=missing-raises-doc
def map_eigenvalues(
    matrix: np.ndarray, func: Callable[[complex], complex], *, atol: float = 1e-8
) -> np.ndarray:
    """Applies a function to the eigenvalues of a matrix.

    Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

    Args:
        matrix: The matrix to modify with the function.
        func: The function to apply to the eigenvalues of the matrix.
        atol: Absolute threshold used when separating eigenspaces.

    Returns:
        The transformed matrix.
    """
    vals, vecs = unitary_eig(matrix, atol=atol)
    pieces = [np.outer(vec, np.conj(vec.T)) for vec in vecs.T]
    out_vals = np.vectorize(func)(vals.astype(complex))

    total = np.zeros(shape=matrix.shape)
    for piece, val in zip(pieces, out_vals):
        total = np.add(total, piece * val)
    return total


def kron_factor_4x4_to_2x2s(
    matrix: np.ndarray, rtol=1e-5, atol=1e-8
) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.

    Args:
        matrix: The 4x4 unitary matrix to factor.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(matrix[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 /= np.sqrt(np.linalg.det(f1)) or 1
        f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    if not np.allclose(matrix, g * np.kron(f1, f2), rtol=rtol, atol=atol):
        raise ValueError("Invalid 4x4 kronecker product.")

    return g, f1, f2


def so4_to_magic_su2s(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
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
        if mat.shape != (4, 4) or not predicates.is_special_orthogonal(mat, atol=atol, rtol=rtol):
            raise ValueError('mat must be 4x4 special orthogonal.')

    ab = combinators.dot(MAGIC, mat, MAGIC_CONJ_T)
    _, a, b = kron_factor_4x4_to_2x2s(ab, rtol, atol)

    return a, b


@value.value_equality(approximate=True)
class AxisAngleDecomposition:
    """Represents a unitary operation as an axis, angle, and global phase.

    The unitary $U$ is decomposed as follows:

        $$U = g e^{-i \theta/2 (xX + yY + zZ)}$$

    where \theta is the rotation angle, (x, y, z) is a unit vector along the
    rotation axis, and g is the global phase.
    """

    def __init__(self, *, angle: float, axis: Tuple[float, float, float], global_phase: complex):
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

        return AxisAngleDecomposition(axis=(x, y, z), angle=angle, global_phase=p)

    def _value_equality_values_(self) -> Any:
        v = self.canonicalize(atol=0)
        return (value.PeriodicValue(v.angle, period=math.pi * 2), v.axis, v.global_phase)

    def _unitary_(self) -> np.ndarray:
        x, y, z = self.axis
        xm = np.array([[0, 1], [1, 0]])
        ym = np.array([[0, -1j], [1j, 0]])
        zm = np.diag([1, -1])
        i = np.eye(2)
        c = math.cos(-self.angle / 2)
        s = math.sin(-self.angle / 2)
        return (c * i + 1j * s * (x * xm + y * ym + z * zm)) * self.global_phase

    def __str__(self) -> str:
        axis_terms = '+'.join(
            f'{e:.3g}*{a}' if e < 0.9999 else a
            for e, a in zip(self.axis, ['X', 'Y', 'Z'])
            if abs(e) >= 1e-8
        ).replace('+-', '-')
        half_turns = self.angle / np.pi
        return f'{half_turns:.3g}*π around {axis_terms}'

    def __repr__(self) -> str:
        return (
            f'cirq.AxisAngleDecomposition(angle={self.angle!r}, '
            f'axis={self.axis!r}, global_phase={self.global_phase!r})'
        )


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

    return AxisAngleDecomposition(axis=(x, y, z), angle=angle, global_phase=p).canonicalize()


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

    def __init__(
        self,
        *,
        global_phase: complex = complex(1),
        single_qubit_operations_before: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        interaction_coefficients: Tuple[float, float, float],
        single_qubit_operations_after: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            interaction_coefficients: x, y, z from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase: complex = global_phase
        self.single_qubit_operations_before: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_before
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )
        self.interaction_coefficients = interaction_coefficients
        self.single_qubit_operations_after: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_after
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )

    def _value_equality_values_(self) -> Any:
        def flatten(x):
            return tuple(tuple(e.flat) for e in x)

        return (
            self.global_phase,
            tuple(self.interaction_coefficients),
            flatten(self.single_qubit_operations_before),
            flatten(self.single_qubit_operations_after),
        )

    def __str__(self) -> str:
        xx = self.interaction_coefficients[0] * 4 / np.pi
        yy = self.interaction_coefficients[1] * 4 / np.pi
        zz = self.interaction_coefficients[2] * 4 / np.pi
        before0 = axis_angle(self.single_qubit_operations_before[0])
        before1 = axis_angle(self.single_qubit_operations_before[1])
        after0 = axis_angle(self.single_qubit_operations_after[0])
        after1 = axis_angle(self.single_qubit_operations_after[1])
        return (
            'KAK {\n'
            f'    xyz*(4/π): {xx:.3g}, {yy:.3g}, {zz:.3g}\n'
            f'    before: ({before0}) ⊗ ({before1})\n'
            f'    after: ({after0}) ⊗ ({after1})\n'
            '}'
        )

    def __repr__(self) -> str:
        before0 = proper_repr(self.single_qubit_operations_before[0])
        before1 = proper_repr(self.single_qubit_operations_before[1])
        after0 = proper_repr(self.single_qubit_operations_after[0])
        after1 = proper_repr(self.single_qubit_operations_after[1])
        return (
            'cirq.KakDecomposition(\n'
            f'    interaction_coefficients={self.interaction_coefficients!r},\n'
            '    single_qubit_operations_before=(\n'
            f'        {before0},\n'
            f'        {before1},\n'
            '    ),\n'
            '    single_qubit_operations_after=(\n'
            f'        {after0},\n'
            f'        {after1},\n'
            '    ),\n'
            f'    global_phase={self.global_phase!r})'
        )

    def _unitary_(self) -> np.ndarray:
        """Returns the decomposition's two-qubit unitary matrix.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)
        """
        before = np.kron(*self.single_qubit_operations_before)
        after = np.kron(*self.single_qubit_operations_after)

        def interaction_matrix(m: np.ndarray, c: float) -> np.ndarray:
            return map_eigenvalues(np.kron(m, m), lambda v: np.exp(1j * v * c))

        x, y, z = self.interaction_coefficients
        x_mat = np.array([[0, 1], [1, 0]])
        y_mat = np.array([[0, -1j], [1j, 0]])
        z_mat = np.array([[1, 0], [0, -1]])

        return self.global_phase * combinators.dot(
            after,
            interaction_matrix(z_mat, z),
            interaction_matrix(y_mat, y),
            interaction_matrix(x_mat, x),
            before,
        )

    def _decompose_(self, qubits):
        from cirq import ops

        a, b = qubits
        return [
            ops.global_phase_operation(self.global_phase),
            ops.MatrixGate(self.single_qubit_operations_before[0]).on(a),
            ops.MatrixGate(self.single_qubit_operations_before[1]).on(b),
            np.exp(1j * ops.X(a) * ops.X(b) * self.interaction_coefficients[0]),
            np.exp(1j * ops.Y(a) * ops.Y(b) * self.interaction_coefficients[1]),
            np.exp(1j * ops.Z(a) * ops.Z(b) * self.interaction_coefficients[2]),
            ops.MatrixGate(self.single_qubit_operations_after[0]).on(a),
            ops.MatrixGate(self.single_qubit_operations_after[1]).on(b),
        ]


def scatter_plot_normalized_kak_interaction_coefficients(
    interactions: Iterable[Union[np.ndarray, 'cirq.SupportsUnitary', 'KakDecomposition']],
    *,
    include_frame: bool = True,
    ax: Optional[mplot3d.axes3d.Axes3D] = None,
    **kwargs,
):
    r"""Plots the interaction coefficients of many two-qubit operations.

    Plots:
        A point for the (x, y, z) normalized interaction coefficients of
        each interaction from the given interactions. The (x, y, z) coordinates
        are normalized so that the maximum value is at 1 instead of at pi/4.

        If `include_frame` is set to True, then a black wireframe outline of the
        canonicalized normalized KAK coefficient space. The space is defined by
        the following two constraints:

            0 <= abs(z) <= y <= x <= 1
            if x = 1 then z >= 0

        The wireframe includes lines along the surface of the space at z=0.

        The space is a prism with the identity at the origin, a crease along
        y=z=0 leading to the CZ/CNOT at x=1 and a vertical triangular face that
        contains the iswap at x=y=1,z=0 and the swap at x=y=z=1:

                                 (x=1,y=1,z=0)
                             swap___iswap___swap (x=1,y=1,z=+-1)
                               _/\    |    /
                             _/   \   |   /
                           _/      \  |  /
                         _/         \ | /
                       _/            \|/
        (x=0,y=0,z=0) I---------------CZ (x=1,y=0,z=0)

    Args:
        interactions: An iterable of two qubit unitary interactions. Each
            interaction can be specified as a raw 4x4 unitary matrix, or an
            object with a 4x4 unitary matrix according to `cirq.unitary` (
            (e.g. `cirq.CZ` or a `cirq.KakDecomposition` or a `cirq.Circuit`
            over two qubits).
        include_frame: Determines whether or not to draw the kak space
            wireframe. Defaults to `True`.
        ax: A matplotlib 3d axes object to plot into. If not specified, a new
            figure is created, plotted, and shown.

        **kwargs: Arguments forwarded into the call to `scatter` that plots the
            points. Working arguments include color `c='blue'`, scale `s=2`,
            labelling `label="theta=pi/4"`, etc. For reference see the
            `matplotlib.pyplot.scatter` documentation:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

    Returns:
        The matplotlib 3d axes object that was plotted into.

    Examples:
        >>> ax = None
        >>> for y in np.linspace(0, 0.5, 4):
        ...     a, b = cirq.LineQubit.range(2)
        ...     circuits = [
        ...         cirq.Circuit(
        ...             cirq.CZ(a, b)**0.5,
        ...             cirq.X(a)**y, cirq.X(b)**x,
        ...             cirq.CZ(a, b)**0.5,
        ...             cirq.X(a)**x, cirq.X(b)**y,
        ...             cirq.CZ(a, b) ** 0.5,
        ...         )
        ...         for x in np.linspace(0, 1, 25)
        ...     ]
        ...     ax = cirq.scatter_plot_normalized_kak_interaction_coefficients(
        ...         circuits,
        ...         include_frame=ax is None,
        ...         ax=ax,
        ...         s=1,
        ...         label=f'y={y:0.2f}')
        >>> _ = ax.legend()
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
    """
    show_plot = not ax
    if ax is None:
        fig = plt.figure()
        ax = cast(mplot3d.axes3d.Axes3D, fig.add_subplot(1, 1, 1, projection='3d'))

    def coord_transform(
        pts: Union[List[Tuple[int, int, int]], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts) == 0:
            return np.array([]), np.array([]), np.array([])
        xs, ys, zs = np.transpose(pts)
        return xs, zs, ys

    if include_frame:
        envelope = [
            (0, 0, 0),
            (1, 1, 1),
            (1, 1, -1),
            (0, 0, 0),
            (1, 1, 1),
            (1, 0, 0),
            (0, 0, 0),
            (1, 1, -1),
            (1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 0, 0),
        ]
        ax.plot(*coord_transform(envelope), c='black', linewidth=1)

    # parse input and extract KAK vector
    if not isinstance(interactions, np.ndarray):
        interactions_extracted: List[np.ndarray] = [
            a if isinstance(a, np.ndarray) else protocols.unitary(a) for a in interactions
        ]
    else:
        interactions_extracted = [interactions]

    points = kak_vector(interactions_extracted) * 4 / np.pi

    ax.scatter(*coord_transform(points), **kwargs)
    ax.set_xlim(0, +1)
    ax.set_ylim(-1, +1)
    ax.set_zlim(0, +1)

    if show_plot:
        fig.show()

    return ax


def kak_canonicalize_vector(x: float, y: float, z: float, atol: float = 1e-9) -> KakDecomposition:
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
        np.array([[1, 0], [0, -1]]) * 1j,
    ]

    # Each of these special-unitary matrices swaps two the roles of two axes.
    # The matrix at index k swaps the *other two* axes (e.g. swappers[1] is a
    # Hadamard operation that swaps X and Z).
    swappers = [
        np.array([[1, -1j], [1j, -1]]) * 1j * np.sqrt(0.5),
        np.array([[1, 1], [1, -1]]) * 1j * np.sqrt(0.5),
        np.array([[0, 1 - 1j], [1 + 1j, 0]]) * 1j * np.sqrt(0.5),
    ]

    # Shifting strength by ½π is equivalent to local ops (e.g. exp(i½π XX)∝XX).
    def shift(k, step):
        v[k] += step * np.pi / 2
        phase[0] *= 1j**step
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
        single_qubit_operations_before=(right[1], right[0]),
    )


# yapf: disable
KAK_MAGIC = np.array([[1, 0, 0, 1j],
                      [0, 1j, 1, 0],
                      [0, 1j, -1, 0],
                      [1, 0, 0, -1j]]) * np.sqrt(0.5)

KAK_MAGIC_DAG = np.conjugate(np.transpose(KAK_MAGIC))
KAK_GAMMA = np.array([[1, 1, 1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, 1],
                      [1, -1, -1, 1]]) * 0.25


# yapf: enable


def kak_decomposition(
    unitary_object: Union[
        np.ndarray, 'cirq.SupportsUnitary', 'cirq.Gate', 'cirq.Operation', KakDecomposition
    ],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> KakDecomposition:
    """Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

    Args:
        unitary_object: The value to decompose. Can either be a 4x4 unitary
            matrix, or an object that has a 4x4 unitary matrix (via the
            `cirq.SupportsUnitary` protocol).
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: If set, verifies that the input corresponds to a
            4x4 unitary before decomposing.

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
    if isinstance(unitary_object, KakDecomposition):
        return unitary_object
    if isinstance(unitary_object, np.ndarray):
        mat = unitary_object
    else:
        mat = protocols.unitary(unitary_object)
    if check_preconditions and (
        mat.shape != (4, 4) or not predicates.is_unitary(mat, rtol=rtol, atol=atol)
    ):
        raise ValueError(f'Input must correspond to a 4x4 unitary matrix. Received matrix:\n{mat}')

    # Diagonalize in magic basis.
    left, d, right = diagonalize.bidiagonalize_unitary_with_special_orthogonals(
        KAK_MAGIC_DAG @ mat @ KAK_MAGIC, atol=atol, rtol=rtol, check_preconditions=False
    )

    # Recover pieces.
    a1, a0 = so4_to_magic_su2s(left.T, atol=atol, rtol=rtol, check_preconditions=False)
    b1, b0 = so4_to_magic_su2s(right.T, atol=atol, rtol=rtol, check_preconditions=False)
    w, x, y, z = (KAK_GAMMA @ np.angle(d).reshape(-1, 1)).flatten()
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
        single_qubit_operations_after=(a1, a0),
    )


def kak_vector(
    unitary: Union[Iterable[np.ndarray], np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> np.ndarray:
    r"""Compute the KAK vectors of one or more two qubit unitaries.

    Any 2 qubit unitary may be expressed as

    $$ U = k_l A k_r $$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$ A= \exp \left(i \sum_{s=x,y,z} k_s \sigma_{s}^{(0)} \sigma_{s}^{(1)}
                 \right) $$

    The vector entries are ordered such that
        $$ 0 ≤ |k_z| ≤ k_y ≤ k_x ≤ π/4 $$
    if $k_x$ = π/4, $k_z \geq 0$.

    References:
        The appendix section of "Lower bounds on the complexity of simulating
        quantum gates".
        http://arxiv.org/abs/quant-ph/0307190v1

    Examples:
        >>> cirq.kak_vector(np.eye(4))
        array([0., 0., 0.])
        >>> unitaries = [cirq.unitary(cirq.CZ),cirq.unitary(cirq.ISWAP)]
        >>> cirq.kak_vector(unitaries) * 4 / np.pi
        array([[ 1.,  0., -0.],
               [ 1.,  1.,  0.]])

    Args:
        unitary: A unitary matrix, or a multi-dimensional array of unitary
            matrices. Must have shape (..., 4, 4), where the last two axes are
            for the unitary matrix and other axes are for broadcasting the kak
            vector computation.
        rtol: Per-matrix-entry relative tolerance on equality. Used in unitarity
            check of input.
        atol: Per-matrix-entry absolute tolerance on equality. Used in unitarity
            check of input. This also determines how close $k_x$ must be to π/4
            to guarantee $k_z$ ≥ 0. Must be non-negative.
        check_preconditions: When set to False, skips verifying that the input
            is unitary in order to increase performance.

    Returns:
        The KAK vector of the given unitary or unitaries. The output shape is
        the same as the input shape, except the two unitary matrix axes are
        replaced by the kak vector axis (i.e. the output has shape
        `unitary.shape[:-2] + (3,)`).

    Raises:
        ValueError: If `atol` is negative or if the unitary has the wrong shape.
    """
    unitary = np.asarray(unitary)
    if len(unitary) == 0:
        return np.zeros(shape=(0, 3), dtype=np.float64)

    if unitary.ndim < 2 or unitary.shape[-2:] != (4, 4):
        raise ValueError(
            f'Expected input unitary to have shape (...,4,4), but got {unitary.shape}.'
        )

    if atol < 0:
        raise ValueError(f'Input atol must be positive, got {atol}.')

    if check_preconditions:
        actual = np.einsum('...ba,...bc', unitary.conj(), unitary) - np.eye(4)
        if not np.allclose(actual, np.zeros_like(actual), rtol=rtol, atol=atol):
            raise ValueError(
                'Input must correspond to a 4x4 unitary matrix or tensor of '
                f'unitary matrices. Received input:\n{unitary}'
            )

    UB = np.einsum('...ab,...bc,...cd', MAGIC_CONJ_T, unitary, MAGIC)

    m = np.einsum('...ab,...cb', UB, UB)

    evals, _ = np.linalg.eig(m)

    # The algorithm in the appendix mentioned above is slightly incorrect in
    # that it only works for elements of SU(4). A phase correction must be
    # added to deal with U(4).
    with np.errstate(divide="ignore", invalid="ignore"):
        phases = np.log(-1j * np.linalg.det(unitary)).imag + np.pi / 2
    evals *= np.exp(-1j * phases / 2)[..., np.newaxis]

    # The following steps follow the appendix exactly.
    S2 = np.log(-1j * evals).imag + np.pi / 2
    S2 = np.sort(S2, axis=-1)[..., ::-1]

    n_shifted = (np.round(S2.sum(axis=-1) / (2 * np.pi))).astype(int)
    for n in range(1, 5):
        S2[n_shifted == n, :n] -= 2 * np.pi

    # Fix pathological case of SWAP gate
    S2[n_shifted == -1, :3] += 2 * np.pi

    k_vec = (np.einsum('ab,...b', KAK_GAMMA, S2))[..., 1:] / 2

    return _canonicalize_kak_vector(k_vec, atol)


def _canonicalize_kak_vector(k_vec: np.ndarray, atol: float) -> np.ndarray:
    r"""Map a KAK vector into its Weyl chamber equivalent vector.

    This implementation is vectorized but does not produce the single qubit
    unitaries required to bring the KAK vector into canonical form.

    Args:
        k_vec: The KAK vector to be canonicalized. This input may be vectorized,
            with shape (...,3), where the final axis denotes the k_vector and
            all other axes are broadcast.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0.

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0
        The output is vectorized, with shape k_vec.shape[:-1] + (3,).
    """

    # Get all strengths to (-¼π, ¼π]
    k_vec = np.mod(k_vec + np.pi / 4, np.pi / 2) - np.pi / 4

    # Sort in descending order with respect to absolute value.
    order = np.argsort(np.abs(k_vec), axis=-1)
    k_vec = np.take_along_axis(k_vec, order, axis=-1)[..., ::-1]

    # Multiply x,z and y,z components by -1 to fix x,y sign.
    x_negative = k_vec[..., 0] < 0
    k_vec[x_negative, 0] *= -1
    k_vec[x_negative, 2] *= -1
    y_negative = k_vec[..., 1] < 0
    k_vec[y_negative, 1] *= -1
    k_vec[y_negative, 2] *= -1

    # If x = π/4, force z to be positive.
    x_is_pi_over_4 = np.isclose(k_vec[..., 0], np.pi / 4, atol=atol)
    z_is_negative = k_vec[..., 2] < 0
    need_diff = np.logical_and(x_is_pi_over_4, z_is_negative)
    # -1 to x and z components, then shift x up by pi/2. Since x is pi/4, we
    # actually do nothing to that index.
    k_vec[need_diff, 2] *= -1

    return k_vec


def num_cnots_required(u: np.ndarray, atol: float = 1e-8) -> int:
    """Returns the min number of CNOT/CZ gates required by a two-qubit unitary.

    See Proposition III.1, III.2, III.3 in Shende et al. “Recognizing Small-
    Circuit Structure in Two-Qubit Operators and Timing Hamiltonians to Compute
    Controlled-Not Gates”.  https://arxiv.org/abs/quant-ph/0308045

    Args:
        u: A two-qubit unitary.
        atol: The absolute tolerance used to make this judgement.

    Returns:
        The number of CNOT or CZ gates required to implement the unitary.

    Raises:
        ValueError: If the shape of `u` is not 4 by 4.
    """
    if u.shape != (4, 4):
        raise ValueError(f"Expected unitary of shape (4,4), instead got {u.shape}")
    g = _gamma(transformations.to_special(u))
    # see Fadeev-LeVerrier formula
    a3 = -np.trace(g)
    # no need to check a2 = 6, as a3 = +-4 only happens if the eigenvalues are
    # either all +1 or -1, which unambiguously implies that a2 = 6
    if np.abs(a3 - 4) < atol or np.abs(a3 + 4) < atol:
        return 0
    # see Fadeev-LeVerrier formula
    a2 = (a3 * a3 - np.trace(g @ g)) / 2
    if np.abs(a3) < atol and np.abs(a2 - 2) < atol:
        return 1
    if np.abs(a3.imag) < atol:
        return 2
    return 3


def _gamma(u: np.ndarray) -> np.ndarray:
    """Gamma function to convert u to the magic basis.

    See Definition IV.1 in Shende et al. "Minimal Universal Two-Qubit CNOT-based
    Circuits." https://arxiv.org/abs/quant-ph/0308033

    Args:
        u: a member of SU(4)
    Returns:
        u @ yy @ u.T @ yy, where yy = Y ⊗ Y
    """
    return u @ YY @ u.T @ YY


def extract_right_diag(u: np.ndarray) -> np.ndarray:
    """Extract a diagonal unitary from a 3-CNOT two-qubit unitary.

    Returns a 2-CNOT unitary D that is diagonal, so that U @ D needs only
    two CNOT gates in case the original unitary is a 3-CNOT unitary.

    See Proposition V.2 in Minimal Universal Two-Qubit CNOT-based Circuits.
    https://arxiv.org/abs/quant-ph/0308033

    Args:
        u: three-CNOT two-qubit unitary
    Returns:
        diagonal extracted from U
    """
    t = _gamma(transformations.to_special(u).T).diagonal()
    k = np.real(t[0] + t[3] - t[1] - t[2])
    psi = np.arctan2(np.imag(np.sum(t)), k)
    f = np.exp(1j * psi)
    return np.diag([1, f, f, 1])
