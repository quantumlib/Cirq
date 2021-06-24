"""Utility methods for decomposing two-qubit unitaries into sqrt-iSWAP gates.

References:
    Towards ultra-high fidelity quantum operations: SQiSW gate as a native
    two-qubit gate
    https://arxiv.org/abs/2105.06074
"""

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, protocols
from cirq.optimizers import decompositions

if TYPE_CHECKING:
    import cirq


def two_qubit_matrix_to_sqrt_iswap_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    mat: np.ndarray,
    *,
    required_sqrt_iswap_count: Optional[int] = None,
    use_sqrt_iswap_inv: bool = False,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> Sequence['cirq.Operation']:
    """Decomposes a two-qubit operation into ZPow/XPow/YPow/sqrt-iSWAP gates.

    This method uses the KAK decomposition of the matrix to determine how many
    sqrt-iSWAP gates are needed and which single-qubit gates to use in between
    each sqrt-iSWAP.

    All operations can be synthesized with exactly three sqrt-iSWAP gates and
    about 79% of operations (randomly chosen under the Harr measure) can also be
    synthesized with two sqrt-iSWAP gates.  Only special cases locally
    equivalent to identity or sqrt-iSWAP can be synthesized with zero or one
    sqrt-iSWAP gates respectively.  Unless ``required_sqrt_iswap_count`` is
    specified, the fewest possible number of sqrt-iSWAP will be used.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        required_sqrt_iswap_count: When specified, exactly this many sqrt-iSWAP
            gates will be used even if fewer is possible (maximum 3).  Raises
            ``ValueError`` if impossible.
        use_sqrt_iswap_inv: If True, returns a decomposition using
            ``SQRT_ISWAP_INV`` gates instead of ``SQRT_ISWAP``.  This
            decomposition is identical except for the addition of single-qubit
            Z gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        check_preconditions: If set, verifies that the input corresponds to a
            4x4 unitary before decomposing.

    Returns:
        A list of operations implementing the matrix including at most three
        ``SQRT_ISWAP`` (sqrt-iSWAP) gates and ZPow, XPow, and YPow single-qubit
        gates.

    Raises:
        ValueError:
            If ``required_sqrt_iswap_count`` is specified, the minimum number of
            sqrt-iSWAP gates needed to decompose the given matrix is greater
            than ``required_sqrt_iswap_count``.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    kak = linalg.kak_decomposition(
        mat, atol=atol / 10, rtol=0, check_preconditions=check_preconditions
    )
    operations = _kak_decomposition_to_sqrt_iswap_operations(
        q0, q1, kak, required_sqrt_iswap_count, use_sqrt_iswap_inv, atol=atol
    )
    return operations


def _kak_decomposition_to_sqrt_iswap_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    kak: linalg.KakDecomposition,
    required_sqrt_iswap_count: Optional[int] = None,
    use_sqrt_iswap_inv: bool = False,
    atol: float = 1e-8,
) -> Sequence['cirq.Operation']:
    single_qubit_operations, _ = _single_qubit_matrices_with_sqrt_iswap(
        kak, required_sqrt_iswap_count, atol=atol
    )
    if use_sqrt_iswap_inv:
        z_unitary = protocols.unitary(ops.Z)
        return _decomp_to_operations(
            q0,
            q1,
            ops.SQRT_ISWAP_INV,
            single_qubit_operations,
            u0_before=z_unitary,
            u0_after=z_unitary,
            atol=atol,
        )
    return _decomp_to_operations(
        q0,
        q1,
        ops.SQRT_ISWAP,
        single_qubit_operations,
        atol=atol,
    )


def _decomp_to_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    two_qubit_gate: 'cirq.Gate',
    single_qubit_operations: Sequence[Tuple[np.ndarray, np.ndarray]],
    u0_before: np.ndarray = np.eye(2),
    u0_after: np.ndarray = np.eye(2),
    atol: float = 1e-8,
) -> Sequence['cirq.Operation']:
    """Converts a sequence of single-qubit unitary matrices on two qubits into a
    list of operations with interleaved two-qubit gates."""
    two_qubit_op = two_qubit_gate(q0, q1)
    operations = []

    prev_commute = 1

    def append(matrix0, matrix1, final_layer=False):
        """Appends the decomposed single-qubit operations for matrix0 and
        matrix1.

        The cleanup logic, specific to sqrt-iSWAP, commutes the final Z**a gate
        and any whole X or Y gate on q1 through the following sqrt-iSWAP.

        Commutation rules:
        - Z(q0)**a, Z(q1)**a together commute with sqrt-iSWAP for all a
        - X(q0), X(q0) together commute with sqrt-iSWAP
        - Y(q0), Y(q0) together commute with sqrt-iSWAP
        """
        nonlocal prev_commute
        # Commute previous Z(q0)**a, Z(q1)**a through earlier sqrt-iSWAP
        rots1 = list(
            decompositions.single_qubit_matrix_to_pauli_rotations(
                np.dot(matrix1, prev_commute), atol=atol
            )
        )
        new_commute = np.eye(2, dtype=matrix0.dtype)
        if not final_layer:
            # Commute rightmost Z(q0)**b, Z(q1)**b through next sqrt-iSWAP
            if len(rots1) > 0 and rots1[-1][0] == ops.Z:
                _, prev_z = rots1.pop()
                z_unitary = protocols.unitary(ops.Z ** prev_z)
                new_commute = new_commute @ z_unitary
                matrix0 = z_unitary.T.conj() @ matrix0
            # Commute rightmost whole X(q0), X(q0) or Y, Y through next sqrt-iSWAP
            if len(rots1) > 0 and linalg.tolerance.near_zero_mod(rots1[-1][1], 1, atol=atol):
                pauli, half_turns = rots1.pop()
                p_unitary = protocols.unitary(pauli ** half_turns)
                new_commute = new_commute @ p_unitary
                matrix0 = p_unitary.T.conj() @ matrix0
        rots0 = list(
            decompositions.single_qubit_matrix_to_pauli_rotations(
                np.dot(matrix0, prev_commute), atol=atol
            )
        )
        # Append single qubit ops
        operations.extend((pauli ** half_turns).on(q0) for pauli, half_turns in rots0)
        operations.extend((pauli ** half_turns).on(q1) for pauli, half_turns in rots1)
        prev_commute = new_commute

    single_ops = list(single_qubit_operations)
    if len(single_ops) <= 1:  # Handle zero sqrt-iSWAP case separately
        for matrix0, matrix1 in single_ops:  # Only entry, if any
            append(matrix0, matrix1, final_layer=True)  # Append only pair of single qubit gates
        return operations
    for matrix0, matrix1 in single_ops[:1]:  # First entry
        append(u0_before @ matrix0, matrix1)  # Append pair of single qubit gates
        operations.append(two_qubit_op)  # Append two-qubit gate between each pair
    for matrix0, matrix1 in single_ops[1:-1]:  # All middle entries
        append(u0_before @ matrix0 @ u0_after, matrix1)  # Append pair of single qubit gates
        operations.append(two_qubit_op)  # Append two-qubit gate between each pair
    for matrix0, matrix1 in single_ops[-1:]:  # Last entry
        # Append final pair of single qubit gates
        append(matrix0 @ u0_after, matrix1, final_layer=True)
    return operations


def _single_qubit_matrices_with_sqrt_iswap(
    kak: 'cirq.KakDecomposition',
    required_sqrt_iswap_count: Optional[int] = None,
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Computes the sequence of interleaved single-qubit unitary matrices in the
    sqrt-iSWAP decomposition."""
    decomposers = [
        (_in_0_region, _decomp_0_matrices),
        (_in_1sqrt_iswap_region, _decomp_1sqrt_iswap_matrices),
        (_in_2sqrt_iswap_region, _decomp_2sqrt_iswap_matrices),
        (_in_3sqrt_iswap_region, _decomp_3sqrt_iswap_matrices),
    ]
    if required_sqrt_iswap_count is not None:
        if not 0 <= required_sqrt_iswap_count <= 3:
            raise ValueError('the argument `required_sqrt_iswap_count` must be 0, 1, 2, or 3.')
        can_decompose, decomposer = decomposers[required_sqrt_iswap_count]
        if not can_decompose(kak.interaction_coefficients, weyl_tol=atol / 10):
            raise ValueError(
                f'the given gate cannot be decomposed into exactly '
                f'{required_sqrt_iswap_count} sqrt-iSWAP gates.'
            )
        return decomposer(kak, atol=atol)
    for can_decompose, decomposer in decomposers:
        if can_decompose(kak.interaction_coefficients, weyl_tol=atol / 10):
            return decomposer(kak, atol)
    assert False, 'The final can_decompose should always returns True'


def _in_0_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Tests if (x, y, z) ~= (0, 0, 0) assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x) <= weyl_tol and abs(y) <= weyl_tol and abs(z) <= weyl_tol


def _in_1sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Tests if (x, y, z) ~= (π/8, π/8, 0), assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x - np.pi / 8) <= weyl_tol and abs(y - np.pi / 8) <= weyl_tol and abs(z) <= weyl_tol


def _in_2sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Tests if (x, y, z) is inside or within weyl_tol of the volume
    x >= y + |z| assuming x, y, z are canonical.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = interaction_coefficients
    # Lemma 1 of the paper
    return x + weyl_tol >= y + abs(z)


def _in_3sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Any two-qubit operation is decomposable into three SQRT_ISWAP gates.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    return True


def _decomp_0_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 0-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (0, 0, 0) within tolerance.
    """
    # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
    # Only a single pair of single-qubit unitaries is returned here so
    # _decomp_to_operations will not insert any sqrt-iSWAP gates in between
    return [
        (
            kak.single_qubit_operations_after[0] @ kak.single_qubit_operations_before[0],
            kak.single_qubit_operations_after[1] @ kak.single_qubit_operations_before[1],
        )
    ], kak.global_phase


def _decomp_1sqrt_iswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 1-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (π/8, π/8, 0) within tolerance.
    """
    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        kak.single_qubit_operations_before,
        kak.single_qubit_operations_after,
    ], kak.global_phase


def _decomp_2sqrt_iswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 2-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and x >= y + |z| within tolerance.  For x, y, z
    that violate this inequality, three sqrt-iSWAP gates are required.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = kak.interaction_coefficients
    a0, a1 = kak.single_qubit_operations_before
    c0, c1 = kak.single_qubit_operations_after

    # Computed gate parameters: Eq. 4, 6, 7, 8 of the paper
    # range limits added for robustness to numerical error
    def safe_arccos(v):
        return np.arccos(np.clip(v, -1, 1))

    def nonzero_sign(v):
        return -1 if v < 0 else 1

    _c = np.clip(
        np.sin(x + y - z) * np.sin(x - y + z) * np.sin(-x - y - z) * np.sin(-x + y + z), 0, 1
    )
    alpha = safe_arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) + 2 * np.sqrt(_c))
    beta = safe_arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) - 2 * np.sqrt(_c))
    _4ccs = 4 * (np.cos(x) * np.cos(z) * np.sin(y)) ** 2  # Intermediate value
    gamma = safe_arccos(
        nonzero_sign(z)
        * np.sqrt(_4ccs / (_4ccs + np.clip(np.cos(2 * x) * np.cos(2 * y) * np.cos(2 * z), 0, 1)))
    )

    # Inner single-qubit gates: Fig. 4 of the paper
    b0 = (
        protocols.unitary(ops.rz(-gamma))
        @ protocols.unitary(ops.rx(-alpha))
        @ protocols.unitary(ops.rz(-gamma))
    )
    b1 = protocols.unitary(ops.rx(-beta))

    # Compute KAK on the decomposition to determine outer single-qubit gates
    # There is no known closed form solution for these gates
    u_sqrt_iswap = protocols.unitary(ops.SQRT_ISWAP)
    u = u_sqrt_iswap @ np.kron(b0, b1) @ u_sqrt_iswap  # Unitary of decomposition
    kak_fix = linalg.kak_decomposition(u, atol=atol / 10, rtol=0, check_preconditions=False)
    d0, d1 = kak_fix.single_qubit_operations_before
    e0, e1 = kak_fix.single_qubit_operations_after

    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        (d0.T.conj() @ a0, d1.T.conj() @ a1),
        (b0, b1),
        (c0 @ e0.T.conj(), c1 @ e1.T.conj()),
    ], kak.global_phase / kak_fix.global_phase


def _decomp_3sqrt_iswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 3-SQRT_ISWAP decomposition.

    Assumes any canonical x, y, z.  Three sqrt-iSWAP gates are only needed if
    x < y + |z|.  Only two are needed for other gates (most cases).

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = kak.interaction_coefficients
    f0, f1 = kak.single_qubit_operations_before
    g0, g1 = kak.single_qubit_operations_after

    # Find x1, y1, z1, x2, y2, z2
    # such that x1+x2=x, y1+y2=y, z1+z2=z
    # where x1, y1, z1 are implementable by one sqrt-iSWAP gate
    # and x2, y2, z2 implementable by two sqrt-iSWAP gates
    # No error tolerance needed
    ieq1 = y > np.pi / 8
    ieq2 = z < 0
    if ieq1:
        if ieq2:
            # Non-canonical Weyl coordinates for the single sqrt-iSWAP
            x1, y1, z1 = 0, np.pi / 8, -np.pi / 8
        else:
            x1, y1, z1 = 0, np.pi / 8, np.pi / 8
    else:
        x1, y1, z1 = -np.pi / 8, np.pi / 8, 0
    # Non-canonical Weyl coordinates for the two sqrt-iSWAP decomposition
    x2, y2, z2 = x - x1, y - y1, z - z1

    # Find fixup single-qubit gates for the canonical decompositions
    kak1 = linalg.kak_canonicalize_vector(x1, y1, z1, atol)
    kak2 = linalg.kak_canonicalize_vector(x2, y2, z2, atol)

    # Compute sub-decompositions
    ((a0, a1), (b0, b1)), phase1 = _decomp_1sqrt_iswap_matrices(kak1, atol)
    ((c0, c1), (d0, d1), (e0, e1)), phase2 = _decomp_2sqrt_iswap_matrices(kak2, atol)

    # There are two valid solutions: kak1 before kak2 or kak2 before kak1
    # Arbitrarily pick kak1 before kak2
    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        (a0 @ f0, a1 @ f1),
        (c0 @ b0, c1 @ b1),
        (d0, d1),
        (g0 @ e0, g1 @ e1),
    ], kak.global_phase * phase1 * phase2
