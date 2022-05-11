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

"""Utility methods for transforming matrices or vectors."""

from typing import Tuple, Optional, Sequence, List, Union

import numpy as np

from cirq import protocols
from cirq.linalg import predicates

# This is a special indicator value used by the `sub_state_vector` method to
# determine whether or not the caller provided a 'default' argument. It must be
# of type np.ndarray to ensure the method has the correct type signature in that
# case. It is checked for using `is`, so it won't have a false positive if the
# user provides a different np.array([]) value.
RaiseValueErrorIfNotProvided: np.ndarray = np.array([])


def reflection_matrix_pow(reflection_matrix: np.ndarray, exponent: float):
    """Raises a matrix with two opposing eigenvalues to a power.

    Args:
        reflection_matrix: The matrix to raise to a power.
        exponent: The power to raise the matrix to.

    Returns:
        The given matrix raised to the given power.
    """

    # The eigenvalues are x and -x for some complex unit x. Determine x.
    squared_phase = np.dot(reflection_matrix[:, 0], reflection_matrix[0, :])
    phase = complex(np.sqrt(squared_phase))

    # Extract +x and -x eigencomponents of the matrix.
    i = np.eye(reflection_matrix.shape[0]) * phase
    pos_part = (i + reflection_matrix) * 0.5
    neg_part = (i - reflection_matrix) * 0.5

    # Raise the matrix to a power by raising its eigencomponents to that power.
    pos_factor = phase ** (exponent - 1)
    neg_factor = pos_factor * complex(-1) ** exponent
    pos_part_raised = pos_factor * pos_part
    neg_part_raised = neg_part * neg_factor
    return pos_part_raised + neg_part_raised


def match_global_phase(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape or a.size == 0:
        return np.copy(a), np.copy(b)

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def targeted_left_multiply(
    left_matrix: np.ndarray,
    right_target: np.ndarray,
    target_axes: Sequence[int],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Left-multiplies the given axes of the target tensor by the given matrix.

    Note that the matrix must have a compatible tensor structure.

    For example, if you have an 6-qubit state vector `input_state` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation `op` with shape
    (2, 2, 2, 2), and you want to apply `op` to the 5'th and 3'rd qubits
    within `input_state`, then the output state vector is computed as follows:

        output_state = cirq.targeted_left_multiply(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is `old_effect`, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows:

        new_effect = cirq.targeted_left_multiply(
            left_matrix=cirq.unitary(cirq.CNOT).reshape((2, 2, 2, 2)),
            right_target=old_effect,
            target_axes=[1, 4])

    Args:
        left_matrix: What to left-multiply the target tensor by.
        right_target: A tensor to carefully broadcast a left-multiply over.
        target_axes: Which axes of the target are being operated on.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as right_target.

    Returns:
        The output tensor.

    Raises:
        ValueError: If `out` is either `right_target` or `left_matrix`.
    """
    if out is right_target or out is left_matrix:
        raise ValueError('out is right_target or out is left_matrix')

    k = len(target_axes)
    d = len(right_target.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_axes)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_axes):
        output_indices[t] = w

    all_indices = set(input_indices + data_indices + tuple(output_indices))

    return np.einsum(
        left_matrix,
        input_indices,
        right_target,
        data_indices,
        output_indices,
        # We would prefer to omit 'optimize=' (it's faster),
        # but this is a workaround for a bug in numpy:
        #     https://github.com/numpy/numpy/issues/10926
        optimize=len(all_indices) >= 26,
        # And this is workaround for *another* bug!
        # Supposed to be able to just say 'old=old'.
        **({'out': out} if out is not None else {}),
    )


def targeted_conjugate_about(
    tensor: np.ndarray,
    target: np.ndarray,
    indices: Sequence[int],
    conj_indices: Sequence[int] = None,
    buffer: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Conjugates the given tensor about the target tensor.

    This method computes a target tensor conjugated by another tensor.
    Here conjugate is used in the sense of conjugating by a matrix, i.a.
    A conjugated about B is $A B A^\dagger$ where $\dagger$ represents the
    conjugate transpose.

    Abstractly this compute $A \cdot B \cdot A^\dagger$ where A and B are
    multi-dimensional arrays, and instead of matrix multiplication $\cdot$
    is a contraction between the given indices (indices for first $\cdot$,
    conj_indices for second $\cdot$).

    More specifically this computes
        $\sum tensor_{i_0,...,i_{r-1},j_0,...,j_{r-1}} *
        target_{k_0,...,k_{r-1},l_0,...,l_{r-1}} *
        tensor_{m_0,...,m_{r-1},n_0,...,n_{r-1}}^*$

    where the sum is over indices where $j_s$ = $k_s$ and $s$ is in `indices`
    and $l_s$ = $m_s$ and s is in `conj_indices`.

    Args:
        tensor: The tensor that will be conjugated about the target tensor.
        target: The tensor that will receive the conjugation.
        indices: The indices which will be contracted between the tensor and
            target.
        conj_indices: The indices which will be contracted between the
            complex conjugate of the tensor and the target. If this is None,
            then these will be the values in indices plus half the number
            of dimensions of the target (`ndim`). This is the most common case
            and corresponds to the case where the target is an operator on
            a n-dimensional tensor product space (here `n` would be `ndim`).
        buffer: A buffer to store partial results in.  If not specified or None,
            a new buffer is used.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as target.

    Returns:
        The result the conjugation.
    """
    conj_indices = conj_indices or [i + target.ndim // 2 for i in indices]
    first_multiply = targeted_left_multiply(tensor, target, indices, out=buffer)
    return targeted_left_multiply(np.conjugate(tensor), first_multiply, conj_indices, out=out)


_TSliceAtom = Union[int, slice, 'ellipsis']
_TSlice = Union[_TSliceAtom, Sequence[_TSliceAtom]]


def apply_matrix_to_slices(
    target: np.ndarray,
    matrix: np.ndarray,
    slices: Sequence[_TSlice],
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Left-multiplies an NxN matrix onto N slices of a numpy array.

    Example:
        The 4x4 matrix of a fractional SWAP gate can be expressed as

           [ 1       ]
           [   X**t  ]
           [       1 ]

        Where X is the 2x2 Pauli X gate and t is the power of the swap with t=1
        being a full swap. X**t is a power of the Pauli X gate's matrix.
        Applying the fractional swap is equivalent to applying a fractional X
        within the inner 2x2 subspace; the rest of the matrix is identity. This
        can be expressed using `apply_matrix_to_slices` as follows:

            def fractional_swap(target):
                assert target.shape == (4,)
                return apply_matrix_to_slices(
                    target=target,
                    matrix=cirq.unitary(cirq.X**t),
                    slices=[1, 2]
                )

    Args:
        target: The input array with slices that need to be left-multiplied.
        matrix: The linear operation to apply to the subspace defined by the
            slices.
        slices: The parts of the tensor that correspond to the "vector entries"
            that the matrix should operate on. May be integers or complicated
            multi-dimensional slices into a tensor. The slices must refer to
            non-overlapping sections of the input all with the same shape.
        out: Where to write the output. If not specified, a new numpy array is
            created, with the same shape and dtype as the target, to store the
            output.

    Returns:
        The transformed array.

    Raises:
        ValueError: If `out` is `target` , or the matrix shaped does not match
            `slices`.
    """
    # Validate arguments.
    if out is target:
        raise ValueError("Can't write output over the input.")
    if matrix.shape != (len(slices), len(slices)):
        raise ValueError("matrix.shape != (len(slices), len(slices))")

    # Fill in default values and prepare space.
    if out is None:
        out = np.copy(target)
    else:
        out[...] = target[...]

    # Apply operation.
    for i, s_i in enumerate(slices):
        out[s_i] *= matrix[i, i]
        for j, s_j in enumerate(slices):
            if i != j:
                out[s_i] += target[s_j] * matrix[i, j]

    return out


def partial_trace(tensor: np.ndarray, keep_indices: Sequence[int]) -> np.ndarray:
    """Takes the partial trace of a given tensor.

    The input tensor must have shape `(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
    The trace is done over all indices that are not in keep_indices. The
    resulting tensor has shape `(d_{i_0}, ..., d_{i_r}, d_{i_0}, ..., d_{i_r})`
    where `i_j` is the `j`th element of `keep_indices`.

    Args:
        tensor: The tensor to sum over. This tensor must have a shape
            `(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
        keep_indices: Which indices to not sum over. These are only the indices
            of the first half of the tensors indices (i.e. all elements must
            be between `0` and `tensor.ndims / 2 - 1` inclusive).

    Raises:
        ValueError: if the tensor is not of the correct shape or the indices
            are not from the first half of valid indices for the tensor.
    """
    ndim = tensor.ndim // 2
    if not all(tensor.shape[i] == tensor.shape[i + ndim] for i in range(ndim)):
        raise ValueError(
            'Tensors must have shape (d_0,...,d_{{k-1}},d_0,...,'
            'd_{{k-1}}) but had shape ({}).'.format(tensor.shape)
        )
    if not all(i < ndim for i in keep_indices):
        raise ValueError(
            'keep_indices were {} but must be in first half, '
            'i.e. have index less that {}.'.format(keep_indices, ndim)
        )
    keep_set = set(keep_indices)
    keep_map = dict(zip(keep_indices, sorted(keep_indices)))
    left_indices = [keep_map[i] if i in keep_set else i for i in range(ndim)]
    right_indices = [ndim + i if i in keep_set else i for i in left_indices]
    return np.einsum(tensor, left_indices + right_indices)


class EntangledStateError(ValueError):
    """Raised when a product state is expected, but an entangled state is provided."""


def partial_trace_of_state_vector_as_mixture(
    state_vector: np.ndarray, keep_indices: List[int], *, atol: Union[int, float] = 1e-8
) -> Tuple[Tuple[float, np.ndarray], ...]:
    """Returns a mixture representing a state vector with only some qubits kept.

    The input state vector can have any shape, but if it is one-dimensional it
    will be interpreted as qubits, since that is the most common case, and fail
    if the dimension is not size `2 ** n`. States in the output mixture will
    retain the same type of shape as the input state vector.

    If the state vector cannot be factored into a pure state over `keep_indices`
    then eigendecomposition is used and the output mixture will not be unique.

    Args:
        state_vector: The state vector to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
        atol: The tolerance for determining that a factored state is pure.

    Returns:
        A single-component mixture in which the factored state vector has
        probability '1' if the partially traced state is pure, or else a
        mixture of the default eigendecomposition of the mixed state's
        partial trace.

    Raises:
        ValueError: If the input `state_vector` is one dimension, but that
            dimension size is not a power of two.
        IndexError: If any indexes are out of range.
    """

    if state_vector.ndim == 1:
        dims = int(np.log2(state_vector.size))
        if 2**dims != state_vector.size:
            raise ValueError(f'Cannot infer underlying shape of {state_vector.shape}.')
        state_vector = state_vector.reshape((2,) * dims)
        ret_shape: Tuple[int, ...] = (2 ** len(keep_indices),)
    else:
        ret_shape = tuple(state_vector.shape[i] for i in keep_indices)

    # Attempt to do efficient state factoring.
    try:
        state, _ = factor_state_vector(state_vector, keep_indices, atol=atol)
        return ((1.0, state.reshape(ret_shape)),)
    except EntangledStateError:
        pass

    # Fall back to a (non-unique) mixture representation.
    rho = np.outer(state_vector, np.conj(state_vector)).reshape(state_vector.shape * 2)
    keep_rho = partial_trace(rho, keep_indices).reshape((np.prod(ret_shape),) * 2)
    eigvals, eigvecs = np.linalg.eigh(keep_rho)
    mixture = tuple(zip(eigvals, [vec.reshape(ret_shape) for vec in eigvecs.T]))
    return tuple([(float(p[0]), p[1]) for p in mixture if not protocols.approx_eq(p[0], 0.0)])


def sub_state_vector(
    state_vector: np.ndarray,
    keep_indices: List[int],
    *,
    default: np.ndarray = RaiseValueErrorIfNotProvided,
    atol: Union[int, float] = 1e-6,
) -> np.ndarray:
    r"""Attempts to factor a state vector into two parts and return one of them.

    The input `state_vector` must have shape ``(2,) * n`` or ``(2 ** n)`` where
    `state_vector` is expressed over n qubits. The returned array will retain
    the same type of shape as the input state vector, either ``(2 ** k)`` or
    ``(2,) * k`` where k is the number of qubits kept.

    If a state vector $|\psi\rangle$ defined on n qubits is an outer product
    of kets like  $|\psi\rangle$ = $|x\rangle \otimes |y\rangle$, and
    $|x\rangle$ is defined over the subset ``keep_indices`` of k qubits, then
    this method will factor $|\psi\rangle$ into $|x\rangle$ and $|y\rangle$ and
    return $|x\rangle$. Note that $|x\rangle$ is not unique, because scalar
    multiplication may be absorbed by any factor of a tensor product,
    $e^{i \theta} |y\rangle \otimes |x\rangle =
    |y\rangle \otimes e^{i \theta} |x\rangle$

    This method randomizes the global phase of $|x\rangle$ in order to avoid
    accidental reliance on the global phase being some specific value.

    If the provided `state_vector` cannot be factored into a pure state over
    `keep_indices`, the method will fall back to return `default`. If `default`
    is not provided, the method will fail and raise `ValueError`.

    Args:
        state_vector: The target state_vector.
        keep_indices: Which indices to attempt to get the separable part of the
            `state_vector` on.
        default: Determines the fallback behavior when `state_vector` doesn't
            have a pure state factorization. If the factored state is not pure
            and `default` is not set, a ValueError is raised. If default is set
            to a value, that value is returned.
        atol: The minimum tolerance for comparing the output state's coherence
            measure to 1.

    Returns:
        The state vector expressed over the desired subset of qubits.

    Raises:
        ValueError: If the `state_vector` is not of the correct shape or the
            indices are not a valid subset of the input `state_vector`'s
            indices.
        IndexError: If any indexes are out of range.
        EntangledStateError: If the result of factoring is not a pure state and
            `default` is not provided.

    """

    if not np.log2(state_vector.size).is_integer():
        raise ValueError(
            "Input state_vector of size {} does not represent a "
            "state over qubits.".format(state_vector.size)
        )

    n_qubits = int(np.log2(state_vector.size))
    keep_dims = 1 << len(keep_indices)
    ret_shape: Union[Tuple[int], Tuple[int, ...]]
    if state_vector.shape == (state_vector.size,):
        ret_shape = (keep_dims,)
        state_vector = state_vector.reshape((2,) * n_qubits)
    elif state_vector.shape == (2,) * n_qubits:
        ret_shape = tuple(2 for _ in range(len(keep_indices)))
    else:
        raise ValueError("Input state_vector must be shaped like (2 ** n,) or (2,) * n")

    keep_dims = 1 << len(keep_indices)
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError("Input state must be normalized.")
    if len(set(keep_indices)) != len(keep_indices):
        raise ValueError(f"keep_indices were {keep_indices} but must be unique.")
    if any([ind >= n_qubits for ind in keep_indices]):
        raise ValueError("keep_indices {} are an invalid subset of the input state vector.")

    other_qubits = sorted(set(range(n_qubits)) - set(keep_indices))
    candidates = [
        state_vector[predicates.slice_for_qubits_equal_to(other_qubits, k)].reshape(keep_dims)
        for k in range(1 << len(other_qubits))
    ]
    # The coherence measure is computed using unnormalized candidates.
    best_candidate = max(candidates, key=lambda c: np.linalg.norm(c, 2))
    best_candidate = best_candidate / np.linalg.norm(best_candidate)
    left = np.conj(best_candidate.reshape((keep_dims,))).T
    coherence_measure = sum([abs(np.dot(left, c.reshape((keep_dims,)))) ** 2 for c in candidates])

    if protocols.approx_eq(coherence_measure, 1, atol=atol):
        return np.exp(2j * np.pi * np.random.random()) * best_candidate.reshape(ret_shape)

    # Method did not yield a pure state. Fall back to `default` argument.
    if default is not RaiseValueErrorIfNotProvided:
        return default

    raise EntangledStateError(
        "Input state vector could not be factored into pure state over "
        "indices {}".format(keep_indices)
    )


def to_special(u: np.ndarray) -> np.ndarray:
    """Converts a unitary matrix to a special unitary matrix.

    All unitary matrices u have |det(u)| = 1.
    Also for all d dimensional unitary matrix u, and scalar s:
        det(u * s) = det(u) * s^(d)
    To find a special unitary matrix from u:
        u * det(u)^{-1/d}

    Args:
        u: the unitary matrix
    Returns:
        the special unitary matrix
    """
    return u * (np.linalg.det(u) ** (-1 / len(u)))


def state_vector_kronecker_product(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merges two state vectors into a single unified state vector.

    The resulting vector's shape will be `t1.shape + t2.shape`.

    Args:
        t1: The first state vector.
        t2: The second state vector.
    Returns:
        A new state vector representing the unified state.
    """
    return np.outer(t1, t2).reshape(t1.shape + t2.shape)


def density_matrix_kronecker_product(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merges two density matrices into a single unified density matrix.

    The resulting matrix's shape will be `(t1.shape/2 + t2.shape/2) * 2`. In
    other words, if t1 has shape [A,B,C,A,B,C] and t2 has shape [X,Y,Z,X,Y,Z],
    the resulting matrix will have shape [A,B,C,X,Y,Z,A,B,C,X,Y,Z].

    Args:
        t1: The first density matrix.
        t2: The second density matrix.
    Returns:
        A density matrix representing the unified state.
    """
    t = state_vector_kronecker_product(t1, t2)
    t1_len = len(t1.shape)
    t1_dim = int(t1_len / 2)
    t2_len = len(t2.shape)
    t2_dim = int(t2_len / 2)
    shape = t1.shape[:t1_dim] + t2.shape[:t2_dim]
    return np.moveaxis(t, range(t1_len, t1_len + t2_dim), range(t1_dim, t1_dim + t2_dim)).reshape(
        shape * 2
    )


def factor_state_vector(
    t: np.ndarray, axes: Sequence[int], *, validate=True, atol=1e-07
) -> Tuple[np.ndarray, np.ndarray]:
    """Factors a state vector into two independent state vectors.

    This function should only be called on state vectors that are known to be
    separable, such as immediately after a measurement or reset operation. It
    does not verify that the provided state vector is indeed separable, and
    will return nonsense results for vectors representing entangled states.

    Args:
        t: The state vector to factor.
        axes: The axes to factor out.
        validate: Perform a validation that the density matrix factors cleanly.
        atol: The absolute tolerance for the validation.

    Returns:
        A tuple with the `(extracted, remainder)` state vectors, where
        `extracted` means the sub-state vector which corresponds to the axes
        requested, and with the axes in the requested order, and where
        `remainder` means the sub-state vector on the remaining axes, in the
        same order as the original state vector.

    Raises:
        EntangledStateError: If the tensor is already in entangled state, and
            the validate flag is set.
        ValueError: If the tensor factorization fails for any other reason.
    """
    n_axes = len(axes)
    t1 = np.moveaxis(t, axes, range(n_axes))
    pivot = np.unravel_index(np.abs(t1).argmax(), t1.shape)
    slices1 = (slice(None),) * n_axes + pivot[n_axes:]
    slices2 = pivot[:n_axes] + (slice(None),) * (t1.ndim - n_axes)
    extracted = t1[slices1]
    extracted = extracted / np.sum(abs(extracted) ** 2) ** 0.5
    remainder = t1[slices2]
    remainder = remainder / np.sum(abs(remainder) ** 2) ** 0.5
    if validate:
        t2 = state_vector_kronecker_product(extracted, remainder)
        if not predicates.allclose_up_to_global_phase(t2, t1, atol=atol):
            if not np.isclose(np.linalg.norm(t1), 1):
                raise ValueError('Input state must be normalized.')
            raise EntangledStateError('The tensor cannot be factored by the requested axes')
    return extracted, remainder


def factor_density_matrix(
    t: np.ndarray, axes: Sequence[int], *, validate=True, atol=1e-07
) -> Tuple[np.ndarray, np.ndarray]:
    """Factors a density matrix into two independent density matrices.

    This function should only be called on density matrices that are known to
    be separable, such as immediately after a measurement or reset operation.
    It does not verify that the provided density matrix is indeed separable,
    and will return nonsense results for matrices representing entangled
    states.

    Args:
        t: The density matrix to factor.
        axes: The axes to factor out. Only the left axes should be provided.
            For example, to extract [C,A] from density matrix of shape
            [A,B,C,D,A,B,C,D], `axes` should be [2,0], and the return value
            will be two density matrices ([C,A,C,A], [B,D,B,D]).
        validate: Perform a validation that the density matrix factors cleanly.
        atol: The absolute tolerance for the validation.

    Returns:
        A tuple with the `(extracted, remainder)` density matrices, where
        `extracted` means the sub-matrix which corresponds to the axes
        requested, and with the axes in the requested order, and where
        `remainder` means the sub-matrix on the remaining axes, in the same
        order as the original density matrix.

    Raises:
        ValueError: If the tensor cannot be factored along the given aces.
    """
    extracted = partial_trace(t, axes)
    remaining_axes = [i for i in range(t.ndim // 2) if i not in axes]
    remainder = partial_trace(t, remaining_axes)
    if validate:
        t1 = density_matrix_kronecker_product(extracted, remainder)
        product_axes = list(axes) + remaining_axes
        t2 = transpose_density_matrix_to_axis_order(t1, product_axes)
        if not np.allclose(t2, t, atol=atol):
            raise ValueError('The tensor cannot be factored by the requested axes')
    return extracted, remainder


def transpose_state_vector_to_axis_order(t: np.ndarray, axes: Sequence[int]):
    """Transposes the axes of a state vector to a specified order.

    Args:
        t: The state vector to transpose.
        axes: The desired axis order.
    Returns:
        The transposed state vector.
    """
    assert set(axes) == set(range(int(t.ndim))), "All axes must be provided."
    return np.moveaxis(t, axes, range(len(axes)))


def transpose_density_matrix_to_axis_order(t: np.ndarray, axes: Sequence[int]):
    """Transposes the axes of a density matrix to a specified order.

    Args:
        t: The density matrix to transpose.
        axes: The desired axis order. Only the left axes should be provided.
            For example, to transpose [A,B,C,A,B,C] to [C,B,A,C,B,A], `axes`
            should be [2,1,0].
    Returns:
        The transposed density matrix.
    """
    axes = list(axes) + [i + len(axes) for i in axes]
    return transpose_state_vector_to_axis_order(t, axes)
