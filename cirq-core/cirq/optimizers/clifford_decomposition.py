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

"""Utility methods to decompose Clifford gates into circuits."""

from typing import List, TYPE_CHECKING
import functools

import numpy as np
from cirq import ops, protocols, qis, sim

if TYPE_CHECKING:
    import cirq


def _X(
    q: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    protocols.act_on(ops.X, args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.X(qubits[q]))


def _Z(
    q: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    protocols.act_on(ops.Z, args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.Z(qubits[q]))


def _Sdg(
    q: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    # Apply the tableau with S^\{dagger}
    protocols.act_on(ops.S ** -1, args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.S(qubits[q]))


def _H(
    q: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    protocols.act_on(ops.H, args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.H(qubits[q]))


def _CNOT(
    q1: int,
    q2: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    protocols.act_on(ops.CNOT, args, qubits=[qubits[q1], qubits[q2]], allow_decompose=False)
    operations.append(ops.CNOT(qubits[q1], qubits[q2]))


def _SWAP(
    q1: int,
    q2: int,
    args: sim.ActOnCliffordTableauArgs,
    operations: List[ops.Operation],
    qubits: List['cirq.Qid'],
):
    protocols.act_on(ops.SWAP, args, qubits=[qubits[q1], qubits[q2]], allow_decompose=False)
    operations.append(ops.SWAP(qubits[q1], qubits[q2]))


def decompose_clifford_tableau_to_operations(
    qubits: List['cirq.Qid'], clifford_tableau: qis.CliffordTableau
) -> List[ops.Operation]:
    """Decompose an n-qubit Clifford Tableau into a list of one/two qubit operations.

    The implementation is based on Theorem 8 in [1].
    [1] S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
        Phys. Rev. A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196

    Args:
        qubits: The list of qubits being operated on.
        clifford_tableau: The Clifford Tableau for decomposition.

    Returns:
        A list of operations reconstructs the same Clifford tableau.

    Raises:
        ValueError: The length of input qubit mismatch with the size of tableau.
    """
    if len(qubits) != clifford_tableau.n:
        raise ValueError("The number of qubits must be the same as the number of Clifford Tableau.")
    assert (
        clifford_tableau._validate()
    ), "The provided clifford_tableau must satisfy the symplectic property."

    t: qis.CliffordTableau = clifford_tableau.copy()
    operations: List[ops.Operation] = []
    args = sim.ActOnCliffordTableauArgs(
        tableau=t, qubits=qubits, prng=np.random.RandomState(), log_of_measurement_results={}
    )

    _X_with_ops = functools.partial(_X, args=args, operations=operations, qubits=qubits)
    _Z_with_ops = functools.partial(_Z, args=args, operations=operations, qubits=qubits)
    _H_with_ops = functools.partial(_H, args=args, operations=operations, qubits=qubits)
    _S_with_ops = functools.partial(_Sdg, args=args, operations=operations, qubits=qubits)
    _CNOT_with_ops = functools.partial(_CNOT, args=args, operations=operations, qubits=qubits)
    _SWAP_with_ops = functools.partial(_SWAP, args=args, operations=operations, qubits=qubits)

    # The procedure is based on Theorem 8 in
    # [1] S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
    #     Phys. Rev. A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196
    # with modification by doing it row-by-row instead.

    # Suppose we have a Clifford Tableau:
    #                   Xs  Zs
    # Destabilizers:  [ A | B ]
    # Stabilizers:    [ C | D ]
    for i in range(t.n):
        # Step 1a: Make the diagonal element of A equal to 1 by Hadamard gate if necessary.
        if not t.xs[i, i] and t.zs[i, i]:
            _H_with_ops(i)
        # Step 1b: Make the diagonal element of A equal to 1 by SWAP gate if necessary.
        if not t.xs[i, i]:
            for j in range(i + 1, t.n):
                if t.xs[i, j]:
                    _SWAP_with_ops(i, j)
                    break
        # Step 1c: We may still not be able to find non-zero element in whole Xs row. Then,
        # apply swap + Hadamard from zs. It is guaranteed to find one by lemma 5 in [1].
        if not t.xs[i, i]:
            for j in range(i + 1, t.n):
                if t.zs[i, j]:
                    _H_with_ops(j)
                    _SWAP_with_ops(i, j)
                    break

        # Step 2: Eliminate the elements in A By CNOT and phase gate (i-th row)
        # first i rows of destabilizers: [ I  0 | 0  0 ]
        _ = [_CNOT_with_ops(i, j) for j in range(i + 1, t.n) if t.xs[i, j]]
        if np.any(t.zs[i, i:]):
            if not t.zs[i, i]:
                _S_with_ops(i)
            _ = [_CNOT_with_ops(j, i) for j in range(i + 1, t.n) if t.zs[i, j]]
            _S_with_ops(i)

        # Step 3: Eliminate the elements in D By CNOT and phase gate (i-th row)
        # first i rows of stabilizers: [ 0  0 | I  0 ]
        _ = [_CNOT_with_ops(j, i) for j in range(i + 1, t.n) if t.zs[i + t.n, j]]
        if np.any(t.xs[i + t.n, i:]):
            # Swap xs and zs
            _H_with_ops(i)
            _ = [_CNOT_with_ops(i, j) for j in range(i + 1, t.n) if t.xs[i + t.n, j]]
            if t.zs[i + t.n, i]:
                _S_with_ops(i)
            _H_with_ops(i)

    # Step 4: Correct the phase of tableau
    _ = [_Z_with_ops(i) for i, p in enumerate(t.rs[: t.n]) if p]
    _ = [_X_with_ops(i) for i, p in enumerate(t.rs[t.n :]) if p]

    # Step 5: invert the operations by reversing the order: (AB)^{+} = B^{+} A^{+}.
    # Note only S gate is not self-adjoint.
    return operations[::-1]
