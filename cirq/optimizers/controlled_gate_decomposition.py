# Copyright 2020 The Cirq Developers
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

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary

if TYPE_CHECKING:
    import cirq


def _unitary_power(matrix: np.ndarray, power: float) -> np.ndarray:
    return map_eigenvalues(matrix, lambda e: e**power)


def _is_identity(matrix):
    """Checks whether M is identity."""
    return np.allclose(matrix, np.eye(matrix.shape[0]))


def _flatten(x):
    return sum(x, [])


def _decompose_abc(matrix: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decomposes 2x2 unitary matrix.

    Returns 2x2 special unitary matrices A, B, C and phase delta, such that:
    * ABC = I.
    * AXBXC * exp(1j*delta) = matrix.

    See [1], chapter 4.
    """
    assert matrix.shape == (2, 2)
    delta = np.angle(np.linalg.det(matrix)) * 0.5
    alpha = np.angle(matrix[0, 0]) + np.angle(matrix[0, 1]) - 2 * delta
    beta = np.angle(matrix[0, 0]) - np.angle(matrix[0, 1])

    m00_abs = np.abs(matrix[0, 0])
    if np.abs(m00_abs - 1.0) < 1e-9:
        m00_abs = 1
    theta = 2 * np.arccos(m00_abs)

    a = unitary(ops.rz(-alpha)) @ unitary(ops.ry(-theta / 2))
    b = unitary(ops.ry(theta / 2)) @ unitary(ops.rz((alpha + beta) / 2))
    c = unitary(ops.rz((alpha - beta) / 2))

    x = unitary(ops.X)
    assert np.allclose(a @ b @ c, np.eye(2), atol=1e-2)
    assert np.allclose((a @ x @ b @ x @ c) * np.exp(1j * delta),
                       matrix,
                       atol=1e-2)

    return a, b, c, delta


def _decompose_single_ctrl(matrix: np.ndarray, control: 'cirq.Qid',
                           target: 'cirq.Qid') -> List['cirq.Operation']:
    """Decomposes controlled gate with one control.

    See [1], chapter 5.1.
    """
    a, b, c, delta = _decompose_abc(matrix)

    result = [
        ops.ZPowGate(exponent=delta / np.pi).on(control),
        ops.MatrixGate(c).on(target),
        ops.CNOT.on(control, target),
        ops.MatrixGate(b).on(target),
        ops.CNOT.on(control, target),
        ops.MatrixGate(a).on(target),
    ]

    # Remove no-ops.
    result = [g for g in result if not _is_identity(unitary(g))]

    return result


def _ccnot_congruent(c0: 'cirq.Qid', c1: 'cirq.Qid',
                     target: 'cirq.Qid') -> List['cirq.Operation']:
    """Implements 3-qubit gate 'congruent' to CCNOT.

    Returns sequence of operations which is equivalent to applying
    CCNOT(c0, c1, target) and multiplying phase of |101> sate by -1.
    See lemma 6.2 in [1]."""
    return [
        ops.ry(-np.pi / 4).on(target),
        ops.CNOT(c1, target),
        ops.ry(-np.pi / 4).on(target),
        ops.CNOT(c0, target),
        ops.ry(np.pi / 4).on(target),
        ops.CNOT(c1, target),
        ops.ry(np.pi / 4).on(target),
    ]


def decompose_multi_controlled_x(controls: List['cirq.Qid'], target: 'cirq.Qid',
                                 free_qubits: List['cirq.Qid']
                                ) -> List['cirq.Operation']:
    """Implements action of multi-controlled Pauli X gate.

    Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
    gates.
    If `free_qubits` has at least 1 element, result has lengts
    O(len(controls)).

    Args:
        controls - control qubits.
        targets - target qubits.
        free_qubits - qubits which are neither controlled nor target. Can be
            modified by algorithm, but will end up in their initial state.
    """
    m = len(controls)
    if m == 0:
        return [ops.X.on(target)]
    elif m == 1:
        return [ops.CNOT.on(controls[0], target)]
    elif m == 2:
        return [ops.CCNOT.on(controls[0], controls[1], target)]

    m = len(controls)
    n = m + 1 + len(free_qubits)
    if (n >= 2 * m - 1) and (m >= 3):
        # See [1], Lemma 7.2.
        gates1 = [
            _ccnot_congruent(controls[m - 2 - i], free_qubits[m - 4 - i],
                             free_qubits[m - 3 - i]) for i in range(m - 3)
        ]
        gates2 = _ccnot_congruent(controls[0], controls[1], free_qubits[0])
        gates3 = _flatten(gates1) + gates2 + _flatten(gates1[::-1])
        first_ccnot = ops.CCNOT(controls[m - 1], free_qubits[m - 3], target)
        return [first_ccnot, *gates3, first_ccnot, *gates3]
    elif len(free_qubits) >= 1:
        # See [1], Lemma 7.3.
        m1 = n // 2
        free1 = controls[m1:] + [target] + free_qubits[1:]
        ctrl1 = controls[:m1]
        part1 = decompose_multi_controlled_x(ctrl1, free_qubits[0], free1)
        free2 = controls[:m1] + free_qubits[1:]
        ctrl2 = controls[m1:] + [free_qubits[0]]
        part2 = decompose_multi_controlled_x(ctrl2, target, free2)
        return [*part1, *part2, *part1, *part2]
    else:
        # No free qubits - must use general algorithm.
        # This will never happen if called from main algorithm and is added
        # only for completeness.
        return decompose_multi_controlled_rotation(unitary(ops.X), controls,
                                                   target)


def _decompose_su(matrix: np.ndarray, controls: List['cirq.Qid'],
                  target: 'cirq.Qid') -> List['cirq.Operation']:
    """Decomposes controlled special unitary gate into elementary gates.

    Result has O(len(controls)) operations.
    See [1], lemma 7.9.
    """
    assert matrix.shape == (2, 2)
    assert is_special_unitary(matrix)
    assert len(controls) >= 1

    a, b, c, _ = _decompose_abc(matrix)

    cnots = decompose_multi_controlled_x(controls[:-1], target, [controls[-1]])
    return [
        *_decompose_single_ctrl(c, controls[-1], target), *cnots,
        *_decompose_single_ctrl(b, controls[-1], target), *cnots,
        *_decompose_single_ctrl(a, controls[-1], target)
    ]


def _decompose_recursive(matrix: np.ndarray, power: float,
                         controls: List['cirq.Qid'], target: 'cirq.Qid',
                         free_qubits: List['cirq.Qid']
                        ) -> List['cirq.Operation']:
    """Decomposes controlled unitary gate into elementary gates.

    Result has O(len(controls)^2) operations.
    See [1], lemma 7.5.
    """
    if len(controls) == 1:
        return _decompose_single_ctrl(_unitary_power(matrix, power),
                                      controls[0], target)

    cnots = decompose_multi_controlled_x(controls[:-1], controls[-1],
                                         free_qubits + [target])
    return [
        *_decompose_single_ctrl(_unitary_power(matrix, 0.5 * power),
                                controls[-1], target), *cnots,
        *_decompose_single_ctrl(_unitary_power(matrix, -0.5 * power),
                                controls[-1], target), *cnots,
        *_decompose_recursive(matrix, 0.5 * power, controls[:-1], target,
                              [controls[-1]] + free_qubits)
    ]


def decompose_multi_controlled_rotation(matrix: np.ndarray,
                                        controls: List['cirq.Qid'],
                                        target: 'cirq.Qid'
                                       ) -> List['cirq.Operation']:
    """Implements action of multi-controlled unitary gate.

    Returns a sequence of operations, which is equivalent to applying
    single-qubit gate with matrix `matrix` on `target`, controlled by
    `controls`.

    Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
    gates.

    If matrix is special unitary, result has length `O(len(controls))`.
    Otherwise result has length `O(len(controls)**2)`.

    References:
        [1] Barenco, Bennett et al.
            Elementary gates for quantum computation. 1995.
            https://arxiv.org/pdf/quant-ph/9503016.pdf

    Args:
        matrix - 2x2 numpy unitary matrix (of real or complex dtype).
        controls - control qubits.
        targets - target qubits.

    Returns:
        A list of operations which, applied in a sequence, are equivalent to
        applying `MatrixGate(matrix).on(target).controlled_by(*controls)`.
    """
    assert is_unitary(matrix)
    assert matrix.shape == (2, 2)

    if len(controls) == 0:
        return [ops.MatrixGate(matrix).on(target)]
    elif len(controls) == 1:
        return _decompose_single_ctrl(matrix, controls[0], target)
    elif is_special_unitary(matrix):
        return _decompose_su(matrix, controls, target)
    else:
        return _decompose_recursive(matrix, 1.0, controls, target, [])
