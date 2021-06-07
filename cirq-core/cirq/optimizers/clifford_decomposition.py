# Copyright 2021 The Cirq Developers
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

"""Utility methods related to decompose clifford gate into circuits."""

from typing import Iterable, List, Tuple, Optional, cast, TYPE_CHECKING

import numpy as np
import functools
from cirq import ops, linalg, protocols, circuits, qis

if TYPE_CHECKING:
    import cirq


def _X(table, q, operations, qubits):
    table.rs[:] ^= table.zs[:, q]
    operations.append(ops.X(qubits[q]))


def _Z(table, q, operations, qubits):
    table.rs[:] ^= table.xs[:, q]
    operations.append(ops.Z(qubits[q]))


def _S(table, q, operations, qubits):
    table.rs[:] ^= table.xs[:, q] & table.zs[:, q]
    table.zs[:, q] ^= table.xs[:, q]
    operations.append(ops.S(qubits[q])**-1)


def _H(table, q, operations, qubits):
    (table.xs[:, q], table.zs[:, q]) = (table.zs[:, q].copy(), table.xs[:, q].copy())
    table.rs[:] ^= table.xs[:, q] & table.zs[:, q]
    operations.append(ops.H(qubits[q]))


def _CNOT(table, q1, q2, operations, qubits):
    table.rs[:] ^= table.xs[:, q1] & table.zs[:, q2] & (~(table.xs[:, q2] ^ table.zs[:, q1]))
    table.xs[:, q2] ^= table.xs[:, q1]
    table.zs[:, q1] ^= table.zs[:, q2]
    operations.append(ops.CNOT(qubits[q1], qubits[q2]))


def _SWAP(table, q1, q2, operations, qubits):
    table.xs[:, [q1, q2]] = table.xs[:, [q2, q1]]
    table.zs[:, [q1, q2]] = table.xs[:, [q2, q1]]
    operations.append(ops.SWAP(qubits[q1], qubits[q2]))


def decompose_clifford_tableau_to_operations(
    qubits: List['cirq.Qid'], clifford_tableau: qis.CliffordTableau
) -> List[ops.Operation]:
    """Decompose an n-qubit Clifford Tableau into one/two qubit operations.

    Args:
        qubits: The list of qubits being operated on.
        clifford_tableau: The Clifford Tableau used to decompose to the operations.

    Returns:
        A list of operations implementing the Clifford tableau
    """
    if len(qubits) != clifford_tableau.n:
        raise ValueError(
            f"The number of qubits must be the same as the number of Clifford Tableau."
        )
    assert (
        clifford_tableau._validate()
    ), "The provided clifford_tableau must satisfy the symplectic property."

    t: qis.CliffordTableau = clifford_tableau.copy()
    operations: List[ops.Operation] = []
    _X_with_ops = functools.partial(_X, operations=operations, qubits=qubits)
    _Z_with_ops = functools.partial(_Z, operations=operations, qubits=qubits)
    _H_with_ops = functools.partial(_H, operations=operations, qubits=qubits)
    _S_with_ops = functools.partial(_S, operations=operations, qubits=qubits)
    _CNOT_with_ops = functools.partial(_CNOT, operations=operations, qubits=qubits)
    _SWAP_with_ops = functools.partial(_SWAP, operations=operations, qubits=qubits)

    # The procedure is based on theorem 8 in
    # [1] S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
    #     Phys. Rev. A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196
    # with some modification by doing it row-by-row instead.

    # Suppose we have a Clifford Tableau
    #                  Xs   Zs
    # Destabilizers:  [ A | B ]
    # Stabilizers:    [ C | D ]
    for i in range(t.n):
        # Step 1: Make sure the Diagonal Elements are 1 by swapping.
        if not t.xs[i, i]:
            for j in range(i + 1, t.n):
                if t.xs[i, j]:
                    _SWAP_with_ops(t, i, j)
                    break
        # We may still not be able to find non-zero element in whole Xs row. In this case,
        # apply swap + Hadamard from zs. It is guaranteed to find one by lemma 5 in [1].
        if not t.xs[i, i]:
            for j in range(i, t.n):
                if t.zs[i, j]:
                    _H_with_ops(t, j)
                    if j != i:
                        _SWAP_with_ops(t, i, j)
                    break

        # Step 2: Gaussian Elimination of A By CNOT and phase gate (row style).
        # first i rows of destabilizers: [ I  0 | 0  0 ]
        _ = [_CNOT_with_ops(t, i, j) for j in range(i + 1, t.n) if t.xs[i, j]]
        if np.any(t.zs[i, i:]):
            if not t.zs[i, i]:
                _S_with_ops(t, i)
            _ = [_CNOT_with_ops(t, j, i) for j in range(i + 1, t.n) if t.zs[i, j]]
            _S_with_ops(t, i)

        # Step 3: Gaussian Elimination of D By CNOT and phase gate (row style).
        # first i rows of stabilizers: [ 0  0 | I  0 ]
        _ = [_CNOT_with_ops(t, j, i) for j in range(i + 1, t.n) if t.zs[i + t.n, j]]
        if np.any(t.xs[i + t.n, i:]):
            # Swap xs and zs
            _H_with_ops(t, i)
            _ = [_CNOT_with_ops(t, i, j) for j in range(i + 1, t.n) if t.xs[i + t.n, j]]
            if t.zs[i + t.n, i]:
                _S_with_ops(t, i)
            _H_with_ops(t, i)

    # Step 4: Correct the phase of tableau
    _ = [_Z_with_ops(t, i) for i, p in enumerate(t.rs[: t.n]) if p]
    _ = [_X_with_ops(t, i) for i, p in enumerate(t.rs[t.n :]) if p]

    # Step 5: invert the operations by reserver the orde: (AB)^{+} = B^{+} A^{+}.
    # Note only S gate is not self-adjoint.
    print(t.matrix(), t.rs)
    return operations[::-1]
