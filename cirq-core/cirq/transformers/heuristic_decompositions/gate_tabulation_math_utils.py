# pylint: disable=wrong-or-nonexistent-copyright-notice
import itertools
from typing import Union, Sequence, Optional

import numpy as np
from cirq.value import random_state

_RealArraylike = Union[np.ndarray, float]


def _single_qubit_unitary(
    theta: _RealArraylike, phi_d: _RealArraylike, phi_o: _RealArraylike
) -> np.ndarray:
    """Single qubit unitary matrix.

    Args:
        theta: cos(theta) is magnitude of 00 matrix element. May be a scalar
           or real ndarray (for broadcasting).
        phi_d: exp(i phi_d) is the phase of 00 matrix element. May be a scalar
           or real ndarray (for broadcasting).
        phi_o: i exp(i phi_o) is the phase of 10 matrix element. May be a scalar
           or real ndarray (for broadcasting).


    Notes:
        The output is vectorized with respect to the angles. I.e, if the angles
        are (broadcastable) arraylike objects whose sum would have shape (...),
        the output is an array of shape (...,2,2), where the final two indices
        correspond to unitary matrices.
    """

    U00 = np.cos(theta) * np.exp(1j * np.asarray(phi_d))
    U10 = 1j * np.sin(theta) * np.exp(1j * np.asarray(phi_o))

    # This implementation is agnostic to the shapes of the angles, as long
    # as they can be broadcast together.
    Udiag = np.array([[U00, np.zeros_like(U00)], [np.zeros_like(U00), U00.conj()]])
    Udiag = np.moveaxis(Udiag, [0, 1], [-2, -1])
    Uoff = np.array([[np.zeros_like(U10), -U10.conj()], [U10, np.zeros_like(U10)]])
    Uoff = np.moveaxis(Uoff, [0, 1], [-2, -1])
    return Udiag + Uoff


def random_qubit_unitary(
    shape: Sequence[int] = (),
    randomize_global_phase: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Random qubit unitary distributed over the Haar measure.

    The implementation is vectorized for speed.

    Args:
        shape: The broadcasted shape of the output. This is used to generate
            a tensor of random unitaries with dimensions tuple(shape) + (2,2).
        randomize_global_phase: (Default False) If True, a global phase is also
            sampled randomly. This corresponds to sampling over U(2) instead of
            SU(2).
        rng: Random number generator to be used in sampling. Default is
            numpy.random.
    """
    real_rng = random_state.parse_random_state(rng)

    theta = np.arcsin(np.sqrt(real_rng.rand(*shape)))
    phi_d = real_rng.rand(*shape) * np.pi * 2
    phi_o = real_rng.rand(*shape) * np.pi * 2

    out = _single_qubit_unitary(theta, phi_d, phi_o)

    if randomize_global_phase:
        out = np.moveaxis(out, (-2, -1), (0, 1))
        out *= np.exp(1j * np.pi * 2 * real_rng.rand(*shape))
        out = np.moveaxis(out, (0, 1), (-2, -1))
    return out


def vector_kron(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized implementation of kron for square matrices."""
    s_0, s_1 = first.shape[-2:], second.shape[-2:]
    assert s_0[0] == s_0[1]
    assert s_1[0] == s_1[1]
    out = np.einsum('...ab,...cd->...acbd', first, second)
    s_v = out.shape[:-4]
    return out.reshape(s_v + (s_0[0] * s_1[0],) * 2)


# Encode all possible local operations that produce equivalent KAK vectors
# and which can also be detected by the entanglement fidelity function
# These operations can be decomposed as s_x^a s_y^b s_z^c n_j p, where
# s_j denotes a pi/2 shift in index j (a,b,c are 0 or 1), n_j is a pi rotation
# about the j axis, and p is a permutation of the three indices.

# all permutations of (1,2,3)
_perms_123 = np.zeros((6, 3, 3), int)
for ind, perm in enumerate(itertools.permutations((0, 1, 2))):
    _perms_123[ind, (0, 1, 2), perm] = 1

_negations = np.zeros((4, 3, 3), int)
_negations[0, (0, 1, 2), (0, 1, 2)] = 1
_negations[1, (0, 1, 2), (0, 1, 2)] = (1, -1, -1)
_negations[2, (0, 1, 2), (0, 1, 2)] = (-1, 1, -1)
_negations[3, (0, 1, 2), (0, 1, 2)] = (-1, -1, 1)

_offsets = np.zeros((8, 3))
_offsets[1, 0] = np.pi / 2
_offsets[2, 1] = np.pi / 2
_offsets[3, 2] = np.pi / 2
_offsets[4, (1, 2)] = np.pi / 2
_offsets[5, (0, 2)] = np.pi / 2
_offsets[6, (0, 1)] = np.pi / 2
_offsets[7, (0, 1, 2)] = np.pi / 2


def _kak_equivalent_vectors(kak_vec) -> np.ndarray:
    """Generates all KAK vectors equivalent under single qubit unitaries."""

    # Technically this is not all equivalent vectors, but a subset of vectors
    # which are not guaranteed to give the same answer under the infidelity
    # formula.

    kak_vec = np.asarray(kak_vec, dtype=float)

    # Apply all permutations, then all negations, then all shifts.

    out = np.einsum('pab,...b->...pa', _perms_123, kak_vec)  # (...,6,3)
    out = np.einsum('nab,...b->...na', _negations, out)  # (...,6,4,3)

    # (...,8,6,4,3)
    out = out[..., np.newaxis, :, :, :] + _offsets[:, np.newaxis, np.newaxis, :]

    # Merge indices
    return np.reshape(out, out.shape[:-4] + (192, 3))


def kak_vector_infidelity(
    k_vec_a: np.ndarray, k_vec_b: np.ndarray, ignore_equivalent_vectors: bool = False
) -> np.ndarray:
    r"""The locally invariant infidelity between two KAK vectors.

    This is the quantity

    $$
    \min 1 - F_e( \exp(i k_a · (XX,YY,ZZ)) kL \exp(i k_b · (XX,YY,ZZ)) kR)
    $$

    where $F_e$ is the entanglement (process) fidelity and the minimum is taken
    over all 1-local unitaries kL, kR.

    Args:
        k_vec_a: A 3-vector or tensor of 3-vectors with shape (...,3).
        k_vec_b: A 3-vector or tensor of 3-vectors with shape (...,3). If both
            k_vec_a and k_vec_b are tensors, their shapes must be compatible
            for broadcasting.
        ignore_equivalent_vectors: If True, the calculation ignores any other
            KAK vectors that are equivalent to the inputs under local unitaries.
            The resulting infidelity is then only an upper bound to the true
            infidelity.

    Returns:
        An ndarray storing the locally invariant infidelity between the inputs.
        If k_vec_a or k_vec_b is a tensor, the result is vectorized.
    """
    k_vec_a, k_vec_b = np.asarray(k_vec_a), np.asarray(k_vec_b)

    if ignore_equivalent_vectors:
        k_diff = k_vec_a - k_vec_b
        out = 1 - np.prod(np.cos(k_diff), axis=-1) ** 2
        out -= np.prod(np.sin(k_diff), axis=-1) ** 2
        return out

    # We must take the minimum infidelity over all possible locally equivalent
    # and nontrivial KAK vectors. We need only consider equivalent vectors
    # of one input.

    # Ensure we consider equivalent vectors for only the smallest input.
    if k_vec_a.size < k_vec_b.size:
        k_vec_a, k_vec_b = k_vec_b, k_vec_a  # coverage: ignore

    k_vec_a = k_vec_a[..., np.newaxis, :]  # (...,1,3)
    k_vec_b = _kak_equivalent_vectors(k_vec_b)  # (...,192,3)

    k_diff = k_vec_a - k_vec_b

    out = 1 - np.prod(np.cos(k_diff), axis=-1) ** 2
    out -= np.prod(np.sin(k_diff), axis=-1) ** 2  # (...,192)

    return out.min(axis=-1)


def in_weyl_chamber(kak_vec: np.ndarray) -> np.ndarray:
    """Whether a given collection of coordinates is within the Weyl chamber.

    Args:
        kak_vec: A numpy.ndarray tensor encoding a KAK 3-vector. Input may be
            broadcastable with shape (...,3).

    Returns:
        np.ndarray of boolean values denoting whether the given coordinates
        are in the Weyl chamber.
    """
    kak_vec = np.asarray(kak_vec)
    assert kak_vec.shape[-1] == 3, 'Last index of input must represent a 3-vector.'
    # For convenience
    xp, yp, zp = kak_vec[..., 0], kak_vec[..., 1], kak_vec[..., 2]

    pi_4 = np.pi / 4

    x_inside = np.logical_and(0 <= xp, xp <= pi_4)

    y_inside = np.logical_and(0 <= yp, yp <= pi_4)
    y_inside = np.logical_and(y_inside, xp >= yp)

    z_inside = np.abs(zp) <= yp

    return np.logical_and.reduce((x_inside, y_inside, z_inside))


def weyl_chamber_mesh(spacing: float) -> np.ndarray:
    """Cubic mesh of points in the Weyl chamber.

    Args:
        spacing: Euclidean distance between neighboring KAK vectors.

    Returns:
        np.ndarray of shape (N,3) corresponding to the points in the Weyl
        chamber.

    Raises:
        ValueError: If the spacing is so small (less than 1e-3) that this
            would build a mesh of size about 1GB.
    """
    if spacing < 1e-3:  # memory required ~ 1 GB
        raise ValueError(f'Generating a mesh with spacing {spacing} may cause system to crash.')

    # Uniform mesh
    disps = np.arange(-np.pi / 4, np.pi / 4, step=spacing)
    mesh_points = np.array([a.ravel() for a in np.array(np.meshgrid(*(disps,) * 3))])
    mesh_points = np.moveaxis(mesh_points, 0, -1)

    # Reduce to points within Weyl chamber
    return mesh_points[in_weyl_chamber(mesh_points)]


_XX = np.zeros((4, 4))
_XX[(0, 1, 2, 3), (3, 2, 1, 0)] = 1
_ZZ = np.diag([1, -1, -1, 1])
_YY = -_XX @ _ZZ
_kak_gens = np.array([_XX, _YY, _ZZ])


def kak_vector_to_unitary(vector: np.ndarray) -> np.ndarray:
    r"""Convert a KAK vector to its unitary matrix equivalent.

    Args:
        vector: A KAK vector shape (..., 3). (Input may be vectorized).

    Returns:
        unitary: Corresponding 2-qubit unitary, of the form
           $exp( i k_x \sigma_x \sigma_x + i k_y \sigma_y \sigma_y
                + i k_z \sigma_z \sigma_z)$.
           matrix or tensor of matrices of shape (..., 4,4).
    """
    vector = np.asarray(vector)
    gens = np.einsum('...a,abc->...bc', vector, _kak_gens)
    evals, evecs = np.linalg.eigh(gens)

    return np.einsum('...ab,...b,...cb', evecs, np.exp(1j * evals), evecs.conj())


def unitary_entanglement_fidelity(U_actual: np.ndarray, U_ideal: np.ndarray) -> np.ndarray:
    r"""Entanglement fidelity between two unitaries.

    For unitary matrices, this is related to the average unitary fidelity F
    as

    :math:`F = \frac{F_e d + 1}{d + 1}`
    where d is the matrix dimension.

    Args:
        U_actual : Matrix whose fidelity to U_ideal will be computed. This may
            be a non-unitary matrix, i.e. the projection of a larger unitary
            matrix into the computational subspace.
        U_ideal : Unitary matrix to which U_actual will be compared.

    Both arguments may be vectorized, in that their shapes may be of the form
    (...,M,M) (as long as both shapes can be broadcast together).

    Returns:
        The entanglement fidelity between the two unitaries. For inputs with
        shape (...,M,M), the output has shape (...).

    """
    U_actual = np.asarray(U_actual)
    U_ideal = np.asarray(U_ideal)
    assert (
        U_actual.shape[-1] == U_actual.shape[-2]
    ), "Inputs' trailing dimensions must be equal (square)."

    dim = U_ideal.shape[-1]

    prod_trace = np.einsum('...ba,...ba->...', U_actual.conj(), U_ideal)

    return np.real((np.abs(prod_trace)) / dim) ** 2
