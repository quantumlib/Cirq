# Copyright 2022 The Cirq Developers
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

"""Utility methods for decomposing two-qubit unitaries into sqrt-iSWAP gates.

References:
    Towards ultra-high fidelity quantum operations: SQiSW gate as a native
    two-qubit gate
    https://arxiv.org/abs/2105.06074
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import sympy

from cirq import circuits, linalg, ops, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz

if TYPE_CHECKING:
    import cirq


def parameterized_2q_op_to_sqrt_iswap_operations(
    op: cirq.Operation, *, use_sqrt_iswap_inv: bool = False
) -> protocols.decompose_protocol.DecomposeResult:
    """Tries to decompose a parameterized 2q operation into √iSWAP's + parameterized 1q rotations.

    Currently only supports decomposing the following gates:
        a) `cirq.CZPowGate`
        b) `cirq.SwapPowGate`
        c) `cirq.ISwapPowGate`
        d) `cirq.FSimGate`

    Args:
        op: Parameterized two qubit operation to be decomposed into sqrt-iswaps.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used as the target 2q gate, instead
            of `cirq.SQRT_ISWAP`.

    Returns:
        A parameterized `cirq.OP_TREE` implementing `op` using only `cirq.SQRT_ISWAP`
        (or `cirq.SQRT_ISWAP_INV`) and parameterized single qubit rotations OR
        None or NotImplemented if decomposition of `op` is not known.
    """
    gate = op.gate
    q0, q1 = op.qubits

    if isinstance(gate, ops.CZPowGate):
        return _cphase_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.SwapPowGate):
        return _swap_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.ISwapPowGate):
        return _iswap_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.FSimGate):
        return _fsim_symbols_to_sqrt_iswap(q0, q1, gate.theta, gate.phi, use_sqrt_iswap_inv)
    return NotImplemented


def _sqrt_iswap_inv(a: cirq.Qid, b: cirq.Qid, use_sqrt_iswap_inv: bool = True) -> cirq.OP_TREE:
    """Optree implementing `cirq.SQRT_ISWAP_INV(a, b)` using √iSWAPs.

    Args:
        a: The first qubit.
        b: The second qubit.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Returns:
        `cirq.SQRT_ISWAP_INV(a, b)` or equivalent unitary implemented using `cirq.SQRT_ISWAP`.
    """
    return (
        ops.SQRT_ISWAP_INV(a, b)
        if use_sqrt_iswap_inv
        else [ops.Z(a), ops.SQRT_ISWAP(a, b), ops.Z(a)]
    )


def _cphase_symbols_to_sqrt_iswap(
    a: cirq.Qid, b: cirq.Qid, turns: cirq.TParamVal, use_sqrt_iswap_inv: bool = True
):
    """Implements `cirq.CZ(a, b) ** turns` using two √iSWAPs and single qubit rotations.

    Output unitary:
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, g]]
    where:
        g = exp(i·π·t).

    Args:
        a: The first qubit.
        b: The second qubit.
        turns: The rotational angle (t) that specifies the gate, where
            g = exp(i·π·t/2).
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    theta = sympy.Mod(turns, 2.0) * sympy.pi

    # -1 if theta > pi.  Adds a hacky fudge factor so theta=pi is not 0
    sign = sympy.sign(sympy.pi - theta + 1e-9)

    # For sign = 1: theta. For sign = -1, 2pi-theta
    theta_prime = (sympy.pi - sign * sympy.pi) + sign * theta

    phi = sympy.asin(np.sqrt(2) * sympy.sin(theta_prime / 4))
    xi = sympy.atan(sympy.tan(phi) / np.sqrt(2))

    yield ops.rz(sign * 0.5 * theta_prime).on(a)
    yield ops.rz(sign * 0.5 * theta_prime).on(b)
    yield ops.rx(xi).on(a)
    yield ops.X(b) ** (-sign * 0.5)
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.rx(-2 * phi).on(a)
    yield ops.Z(a)
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a)
    yield ops.rx(xi).on(a)
    yield ops.X(b) ** (sign * 0.5)


def _swap_symbols_to_sqrt_iswap(
    a: cirq.Qid, b: cirq.Qid, turns: cirq.TParamVal, use_sqrt_iswap_inv: bool = True
):
    """Implements `cirq.SWAP(a, b) ** turns` using two √iSWAPs and single qubit rotations.

    Output unitary:
        [[1, 0,        0,     0],
         [0, g·c,    -i·g·s,  0],
         [0, -i·g·s,  g·c,    0],
         [0,   0,      0,     1]]
    where:
        c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).

    Args:
        a: The first qubit.
        b: The second qubit.
        turns: The rotational angle (t) that specifies the gate, where
            c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    yield ops.Z(a) ** 1.25
    yield ops.Z(b) ** -0.25
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a) ** (-turns / 2 + 1)
    yield ops.Z(b) ** (turns / 2)
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a) ** (turns / 2 - 0.25)
    yield ops.Z(b) ** (turns / 2 + 0.25)
    yield _cphase_symbols_to_sqrt_iswap(a, b, -turns, use_sqrt_iswap_inv)


def _iswap_symbols_to_sqrt_iswap(
    a: cirq.Qid, b: cirq.Qid, turns: cirq.TParamVal, use_sqrt_iswap_inv: bool = True
):
    """Implements `cirq.ISWAP(a, b) ** turns` using two √iSWAPs and single qubit rotations.

    Output unitary:
       [[1   0   0   0],
        [0   c  is   0],
        [0  is   c   0],
        [0   0   0   1]]
    where c = cos(π·t/2), s = sin(π·t/2).

    Args:
        a: The first qubit.
        b: The second qubit.
        turns: The rotational angle (t) that specifies the gate, where
            c = cos(π·t/2), s = sin(π·t/2).
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    yield ops.Z(a) ** 0.75
    yield ops.Z(b) ** 0.25
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a) ** (-turns / 2 + 1)
    yield ops.Z(b) ** (turns / 2)
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a) ** 0.25
    yield ops.Z(b) ** -0.25


def _fsim_symbols_to_sqrt_iswap(
    a: cirq.Qid,
    b: cirq.Qid,
    theta: cirq.TParamVal,
    phi: cirq.TParamVal,
    use_sqrt_iswap_inv: bool = True,
):
    """Implements `cirq.FSimGate(theta, phi)(a, b)` using two √iSWAPs and single qubit rotations.

    FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π)

    Args:
        a: The first qubit.
        b: The second qubit.
        theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
        phi: Controlled phase angle, in radians.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    if theta != 0.0:
        yield _iswap_symbols_to_sqrt_iswap(a, b, -2 * theta / np.pi, use_sqrt_iswap_inv)
    if phi != 0.0:
        yield _cphase_symbols_to_sqrt_iswap(a, b, -phi / np.pi, use_sqrt_iswap_inv)


def two_qubit_matrix_to_sqrt_iswap_operations(
    q0: cirq.Qid,
    q1: cirq.Qid,
    mat: np.ndarray,
    *,
    required_sqrt_iswap_count: Optional[int] = None,
    use_sqrt_iswap_inv: bool = False,
    atol: float = 1e-8,
    check_preconditions: bool = True,
    clean_operations: bool = False,
) -> Sequence[cirq.Operation]:
    """Decomposes a two-qubit operation into ZPow/XPow/YPow/sqrt-iSWAP gates.

    This method uses the KAK decomposition of the matrix to determine how many
    sqrt-iSWAP gates are needed and which single-qubit gates to use in between
    each sqrt-iSWAP.

    All operations can be synthesized with exactly three sqrt-iSWAP gates and
    about 79% of operations (randomly chosen under the Haar measure) can also be
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
        clean_operations: Merges runs of single qubit gates to a single `cirq.PhasedXZGate` in
            the resulting operations list.

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
    return (
        [*merge_single_qubit_gates_to_phxz(circuits.Circuit(operations)).all_operations()]
        if clean_operations
        else operations
    )


def _kak_decomposition_to_sqrt_iswap_operations(
    q0: cirq.Qid,
    q1: cirq.Qid,
    kak: linalg.KakDecomposition,
    required_sqrt_iswap_count: Optional[int] = None,
    use_sqrt_iswap_inv: bool = False,
    atol: float = 1e-8,
) -> Sequence[cirq.Operation]:
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
    return _decomp_to_operations(q0, q1, ops.SQRT_ISWAP, single_qubit_operations, atol=atol)


def _decomp_to_operations(
    q0: cirq.Qid,
    q1: cirq.Qid,
    two_qubit_gate: cirq.Gate,
    single_qubit_operations: Sequence[Tuple[np.ndarray, np.ndarray]],
    u0_before: np.ndarray = np.eye(2),
    u0_after: np.ndarray = np.eye(2),
    atol: float = 1e-8,
) -> Sequence[cirq.Operation]:
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
            single_qubit_decompositions.single_qubit_matrix_to_pauli_rotations(
                np.dot(matrix1, prev_commute), atol=atol
            )
        )
        new_commute = np.eye(2, dtype=matrix0.dtype)
        if not final_layer:
            # Commute rightmost Z(q0)**b, Z(q1)**b through next sqrt-iSWAP
            if len(rots1) > 0 and rots1[-1][0] == ops.Z:
                _, prev_z = rots1.pop()
                z_unitary = protocols.unitary(ops.Z**prev_z)
                new_commute = new_commute @ z_unitary
                matrix0 = z_unitary.T.conj() @ matrix0
            # Commute rightmost whole X(q0), X(q0) or Y, Y through next sqrt-iSWAP
            if len(rots1) > 0 and linalg.tolerance.near_zero_mod(rots1[-1][1], 1, atol=atol):
                pauli, half_turns = rots1.pop()
                p_unitary = protocols.unitary(pauli**half_turns)
                new_commute = new_commute @ p_unitary
                matrix0 = p_unitary.T.conj() @ matrix0
        rots0 = list(
            single_qubit_decompositions.single_qubit_matrix_to_pauli_rotations(
                np.dot(matrix0, prev_commute), atol=atol
            )
        )
        # Append single qubit ops
        operations.extend((pauli**half_turns).on(q0) for pauli, half_turns in rots0)
        operations.extend((pauli**half_turns).on(q1) for pauli, half_turns in rots1)
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
    kak: cirq.KakDecomposition, required_sqrt_iswap_count: Optional[int] = None, atol: float = 1e-8
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
    assert False, 'The final can_decompose should always returns True'  # pragma: no cover


def _in_0_region(
    interaction_coefficients: Tuple[float, float, float], weyl_tol: float = 1e-8
) -> bool:
    """Tests if (x, y, z) ~= (0, 0, 0) assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x) <= weyl_tol and abs(y) <= weyl_tol and abs(z) <= weyl_tol


def _in_1sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float], weyl_tol: float = 1e-8
) -> bool:
    """Tests if (x, y, z) ~= (π/8, π/8, 0), assuming x, y, z are canonical."""
    x, y, z = interaction_coefficients
    return abs(x - np.pi / 8) <= weyl_tol and abs(y - np.pi / 8) <= weyl_tol and abs(z) <= weyl_tol


def _in_2sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float], weyl_tol: float = 1e-8
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
    # The other constraint in Lemma 1 simply asserts x, y, z are canonical
    return x + weyl_tol >= y + abs(z)


def _in_3sqrt_iswap_region(
    interaction_coefficients: Tuple[float, float, float], weyl_tol: float = 1e-8
) -> bool:
    """Any two-qubit operation is decomposable into three SQRT_ISWAP gates.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    return True


def _decomp_0_matrices(
    kak: cirq.KakDecomposition, atol: float = 1e-8
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
    kak: cirq.KakDecomposition, atol: float = 1e-8
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 1-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and (x, y, z) = (π/8, π/8, 0) within tolerance.
    """
    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        kak.single_qubit_operations_before,
        kak.single_qubit_operations_after,
    ], kak.global_phase


def _decomp_2sqrt_iswap_matrices(
    kak: cirq.KakDecomposition, atol: float = 1e-8
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 2-SQRT_ISWAP decomposition.

    Assumes canonical x, y, z and x >= y + |z| within tolerance.  For x, y, z
    that violate this inequality, three sqrt-iSWAP gates are required.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    # Follows the if-branch of procedure DECOMP(U) in Algorithm 1 of the paper
    x, y, z = kak.interaction_coefficients
    b0, b1 = kak.single_qubit_operations_before
    a0, a1 = kak.single_qubit_operations_after

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
    # Don't need to limit this value because it will always be positive and the clip in the
    # following `safe_arccos` handles the cases where this could be slightly greater than 1.
    _4ccs = 4 * (np.cos(x) * np.cos(z) * np.sin(y)) ** 2  # Intermediate value
    gamma = safe_arccos(
        nonzero_sign(z)
        * np.sqrt(_4ccs / (_4ccs + np.clip(np.cos(2 * x) * np.cos(2 * y) * np.cos(2 * z), 0, 1)))
    )

    # Inner single-qubit gates: Fig. 4 of the paper
    # Gate angles here are multiplied by -2 to adjust for non-standard gate definitions in the paper
    c0 = (
        protocols.unitary(ops.rz(-gamma))
        @ protocols.unitary(ops.rx(-alpha))
        @ protocols.unitary(ops.rz(-gamma))
    )
    c1 = protocols.unitary(ops.rx(-beta))

    # Compute KAK on the decomposition to determine outer single-qubit gates
    # There is no known closed form solution for these gates
    u_sqrt_iswap = protocols.unitary(ops.SQRT_ISWAP)
    u = u_sqrt_iswap @ np.kron(c0, c1) @ u_sqrt_iswap  # Unitary of decomposition
    kak_fix = linalg.kak_decomposition(u, atol=atol / 10, rtol=0, check_preconditions=False)
    e0, e1 = kak_fix.single_qubit_operations_before
    d0, d1 = kak_fix.single_qubit_operations_after

    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        (e0.T.conj() @ b0, e1.T.conj() @ b1),
        (c0, c1),
        (a0 @ d0.T.conj(), a1 @ d1.T.conj()),
    ], kak.global_phase / kak_fix.global_phase


def _decomp_3sqrt_iswap_matrices(
    kak: cirq.KakDecomposition, atol: float = 1e-8
) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Returns the single-qubit matrices for the 3-SQRT_ISWAP decomposition.

    Assumes any canonical x, y, z.  Three sqrt-iSWAP gates are only needed if
    x < y + |z|.  Only two are needed for other gates (most cases).

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    # This somewhat follows the else-branch of procedure DECOMP(U) in Algorithm 1 of the paper.
    # However the canonicalization conditions are different from the paper and allow any Weyl
    # coordinate to be synthesized with 3 sqrt-iSWAPs.
    #
    # This method breaks the 3-sqrt-iSWAP synthesis problem into a sum of the 1-sqrt-iSWAP and
    # 2-sqrt-iSWAP problems.  This works because given two 2-qubit unitaries, U and V, with (not
    # necessarily canonical) Weyl coordinates (x1, y1, z1) and (x2, y2, z2), both products U*V and
    # V*U will have (non-canonical) Weyl coordinates (x1+x2, y1+y2, z1+z2) if both U and V are
    # diagonal in the magic basis (i.e. all single-qubit operations of the pre-canonicalized KAK
    # decomposition are identity).
    x, y, z = kak.interaction_coefficients
    b0, b1 = kak.single_qubit_operations_before
    a0, a1 = kak.single_qubit_operations_after

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
            x1, y1, z1 = 0.0, np.pi / 8, -np.pi / 8
        else:
            x1, y1, z1 = 0.0, np.pi / 8, np.pi / 8
    else:
        x1, y1, z1 = -np.pi / 8, np.pi / 8, 0.0
    # Non-canonical Weyl coordinates for the two sqrt-iSWAP decomposition
    x2, y2, z2 = x - x1, y - y1, z - z1

    # Find fixup single-qubit gates for the canonical (i.e. diagonal in the magic basis)
    # decompositions
    kak1 = linalg.kak_canonicalize_vector(x1, y1, z1, atol)
    kak2 = linalg.kak_canonicalize_vector(x2, y2, z2, atol)

    # Compute sub-decompositions
    # F0 and F1 from Algorithm 1 of the paper are not needed
    ((h0, h1), (g0, g1)), phase1 = _decomp_1sqrt_iswap_matrices(kak1, atol)
    ((e0, e1), (c0, c1), (d0, d1)), phase2 = _decomp_2sqrt_iswap_matrices(kak2, atol)

    # There are two valid solutions at this point: kak1 before kak2 or kak2 before kak1
    # Arbitrarily pick kak1 before kak2
    return [  # Pairs of single-qubit unitaries, SQRT_ISWAP between each is implied
        (h0 @ b0, h1 @ b1),
        (e0 @ g0, e1 @ g1),
        (c0, c1),
        (a0 @ d0, a1 @ d1),
    ], kak.global_phase * phase1 * phase2
