"""Utility methods for decomposing two-qubit unitaries into SQISWAP gates.

References:
    Towards ultra-high fidelity quantum operations: SQiSW gate as a native
    two-qubit gate
    https://arxiv.org/abs/2105.06074
"""

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, protocols
from cirq.optimizers import two_qubit_decompositions

if TYPE_CHECKING:
    import cirq


def two_qubit_matrix_to_sqiswap_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    mat: np.ndarray,
    *,
    required_sqiswap_count: Optional[int] = None,
    atol: float = 1e-8,
    clean_operations: bool = True,
) -> Sequence['cirq.Operation']:
    """Decomposes a two-qubit operation into Z/XY/SQISWAP gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        required_sqiswap_count: When specified, exactly this many SQISWAP gates
            will be used even if fewer is possible (maximum 3).
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the matrix including at most three
        SQISWAP (sqrt-iSWAP) gates, single-qubit gates, and a global phase gate.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    kak = linalg.kak_decomposition(mat, atol=atol, rtol=0)
    operations = _kak_decomposition_to_sqiswap_operations(
        q0, q1, kak, required_sqiswap_count, include_global_phase=not clean_operations, atol=atol
    )
    if clean_operations:
        return two_qubit_decompositions._cleanup_operations(operations)
    return operations


def _kak_decomposition_to_sqiswap_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    kak: linalg.KakDecomposition,
    required_sqiswap_count: Optional[int] = None,
    include_global_phase: bool = False,
    atol: float = 1e-8,
) -> Sequence['cirq.Operation']:
    """Computes the list of operations in the SQISWAP decomposition."""
    single_qubit_operations, global_phase = _single_qubit_matrices_with_sqiswap(
        kak, required_sqiswap_count, atol=atol
    )
    return _decomp_to_operations(
        q0,
        q1,
        ops.SQISWAP,
        single_qubit_operations,
        global_phase if include_global_phase else None,
        atol=atol,
    )


def _decomp_to_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    two_qubit_gate: 'cirq.Gate',
    single_qubit_operations: Sequence[Tuple[np.ndarray, np.ndarray]],
    global_phase: Optional[complex] = None,
    atol: float = 1e-8,
) -> Sequence['cirq.Operation']:
    """Converts a sequence of single-qubit unitary matrices on two qubits into a
    list of operations with interleaved two-qubit gates."""
    two_qubit_op = two_qubit_gate(q0, q1)
    operations = []
    if global_phase is not None:
        operations.append(ops.GlobalPhaseOperation(global_phase, atol))

    def append(matrix0, matrix1):
        operations.append(ops.MatrixGate(matrix0).on(q0))
        operations.append(ops.MatrixGate(matrix1).on(q1))

    iter_ops = iter(single_qubit_operations)
    for matrix0, matrix1 in iter_ops:
        append(matrix0, matrix1)  # Append first pair of single qubit gates
        break
    for matrix0, matrix1 in iter_ops:
        operations.append(two_qubit_op)  # Append two-qubit gate between each pair
        append(matrix0, matrix1)  # Append other pairs of single qubit gates
    return operations


def _single_qubit_matrices_with_sqiswap(
    kak: 'cirq.KakDecomposition',
    required_sqiswap_count: Optional[int] = None,
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Computes the sequence of interleaved single-qubit unitary matrices in the
    SQISWAP decomposition."""
    if required_sqiswap_count is not None:
        if not 0 <= required_sqiswap_count <= 3:
            raise ValueError('the argument `required_sqiswap_count` must be 0, 1, 2, or 3.')
        if not [_in_0_region, _in_1sqiswap_region, _in_2sqiswap_region, _in_3sqiswap_region][
            required_sqiswap_count
        ](kak.interaction_coefficients, weyl_tol=atol):
            raise ValueError(
                f'the given gate cannot be decomposed into exactly '
                f'{required_sqiswap_count} SQISWAP gates.'
            )
        return [
            _decomp_0_matrices,
            _decomp_1sqiswap_matrices,
            _decomp_2sqiswap_matrices,
            _decomp_3sqiswap_matrices,
        ][required_sqiswap_count](kak, atol=atol)
    if _in_0_region(kak.interaction_coefficients, weyl_tol=atol):
        return _decomp_0_matrices(kak, atol)
    elif _in_1sqiswap_region(kak.interaction_coefficients, weyl_tol=atol):
        return _decomp_1sqiswap_matrices(kak, atol)
    elif _in_2sqiswap_region(kak.interaction_coefficients, weyl_tol=atol):
        return _decomp_2sqiswap_matrices(kak, atol)
    return _decomp_3sqiswap_matrices(kak, atol)


def _in_0_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Tests if (x, y, z) ~= (0, 0, 0) assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x) <= weyl_tol and abs(y) <= weyl_tol and abs(z) <= weyl_tol


def _in_1sqiswap_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Tests if (x, y, z) ~= (π/8, π/8, 0), assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x - np.pi / 8) <= weyl_tol and abs(y - np.pi / 8) <= weyl_tol and abs(z) <= weyl_tol


def _in_2sqiswap_region(
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


def _in_3sqiswap_region(
    interaction_coefficients: Tuple[float, float, float],
    weyl_tol: float = 1e-8,
) -> bool:
    """Any two-qubit operation is decomposable into three SQISWAP gates.

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
    """Returns the single-qubit matrices for the 0-SQISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (0, 0, 0) within tolerance.
    """
    return [
        (
            kak.single_qubit_operations_after[0] @ kak.single_qubit_operations_before[0],
            kak.single_qubit_operations_after[1] @ kak.single_qubit_operations_before[1],
        )
    ], kak.global_phase


def _decomp_1sqiswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 1-SQISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (π/8, π/8, 0) within tolerance.
    """
    return [  # Pairs of single-qubit unitaries, SQISWAP between each is implied
        kak.single_qubit_operations_before,
        kak.single_qubit_operations_after,
    ], kak.global_phase


def _decomp_2sqiswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 2-SQISWAP decomposition.

    Assumes canonical x, y, z and x >= y + |z| within tolerance.  For x, y, z
    that violate this inequality, three SQISWAP gates are required.

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
    u_sqiswap = protocols.unitary(ops.SQISWAP)
    u = u_sqiswap @ np.kron(b0, b1) @ u_sqiswap  # Unitary of decomposition
    kak_fix = linalg.kak_decomposition(u, atol=atol, rtol=0)
    d0, d1 = kak_fix.single_qubit_operations_before
    e0, e1 = kak_fix.single_qubit_operations_after

    return [  # Pairs of single-qubit unitaries, SQISWAP between each is implied
        (d0.T.conj() @ a0, d1.T.conj() @ a1),
        (b0, b1),
        (c0 @ e0.T.conj(), c1 @ e1.T.conj()),
    ], kak.global_phase / kak_fix.global_phase


def _decomp_3sqiswap_matrices(
    kak: 'cirq.KakDecomposition',
    atol: float = 1e-8,
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 3-SQISWAP decomposition.

    Assumes any canonical x, y, z.  Three SQISWAP gates are only needed if
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
    # where x1, y1, z1 are implementable by one SQiSWAP gate
    # and x2, y2, z2 implementable by two SQiSWAP gates
    # No error tolerance needed
    ieq1 = y > np.pi / 8
    ieq2 = z < 0
    if ieq1:
        if ieq2:
            # Non-canonical Weyl coordinates for the single SQiSWAP
            x1, y1, z1 = 0, np.pi / 8, -np.pi / 8
        else:
            x1, y1, z1 = 0, np.pi / 8, np.pi / 8
    else:
        x1, y1, z1 = -np.pi / 8, np.pi / 8, 0
    # Non-canonical Weyl coordinates for the two-SQiSWAP decomposition
    x2, y2, z2 = x - x1, y - y1, z - z1

    # Find fixup single-qubit gates for the canonical decompositions
    kak1 = linalg.kak_canonicalize_vector(x1, y1, z1, atol)
    kak2 = linalg.kak_canonicalize_vector(x2, y2, z2, atol)

    # Compute sub-decompositions
    ((a0, a1), (b0, b1)), phase1 = _decomp_1sqiswap_matrices(kak1, atol)
    ((c0, c1), (d0, d1), (e0, e1)), phase2 = _decomp_2sqiswap_matrices(kak2, atol)

    # There are two valid solutions: kak1 before kak2 or kak2 before kak1
    # Arbitrarily pick kak1 before kak2
    return [  # Pairs of single-qubit unitaries, SQISWAP between each is implied
        (a0 @ f0, a1 @ f1),
        (c0 @ b0, c1 @ b1),
        (d0, d1),
        (g0 @ e0, g1 @ e1),
    ], kak.global_phase * phase1 * phase2
