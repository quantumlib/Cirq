# Copyright 2023 The Cirq Developers
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

from typing import Any, FrozenSet, Sequence

import cirq
from cirq.protocols.decompose_protocol import DecomposeResult

_FREDKIN_GATESET = cirq.Gateset(cirq.FREDKIN, unroll_circuit_op=False)


def _fredkin(qubits: Sequence[cirq.Qid], context: cirq.DecompositionContext) -> cirq.OP_TREE:
    """Decomposition with 7 T and 10 clifford operations from https://arxiv.org/abs/1308.4134"""
    c, t1, t2 = qubits
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t1), cirq.H(t2)]
    yield [cirq.T(c), cirq.T(t1) ** -1, cirq.T(t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.CNOT(c, t2), cirq.T(t1)]
    yield [cirq.CNOT(c, t1), cirq.T(t2) ** -1]
    yield [cirq.T(t1) ** -1, cirq.CNOT(c, t2)]
    yield [cirq.CNOT(t2, t1)]
    yield [cirq.T(t1), cirq.H(t2)]
    yield [cirq.CNOT(t2, t1)]


def _try_decompose_from_known_decompositions(
    val: Any, context: cirq.DecompositionContext
) -> DecomposeResult:
    """Returns a flattened decomposition of the object into operations, if possible.

    Args:
        val: The object to decompose.
        context: Decomposition context storing common configurable options for `cirq.decompose`.

    Returns:
        A flattened decomposition of `val` if it's a gate or operation with a known decomposition.
    """
    if not isinstance(val, (cirq.Gate, cirq.Operation)):
        return None
    qubits = cirq.LineQid.for_gate(val) if isinstance(val, cirq.Gate) else val.qubits
    known_decompositions = [(_FREDKIN_GATESET, _fredkin)]

    classical_controls: FrozenSet[cirq.Condition] = frozenset()
    if isinstance(val, cirq.ClassicallyControlledOperation):
        classical_controls = val.classical_controls
        val = val.without_classical_controls()

    decomposition = None
    for gateset, decomposer in known_decompositions:
        if val in gateset:
            decomposition = cirq.flatten_to_ops(decomposer(qubits, context))
            break
    return (
        tuple(op.with_classical_controls(*classical_controls) for op in decomposition)
        if decomposition
        else None
    )


def _decompose_once_considering_known_decomposition(val: Any) -> DecomposeResult:
    """Decomposes a value into operations, if possible.

    Args:
        val: The value to decompose into operations.

    Returns:
        A tuple of operations if decomposition succeeds.
    """
    import uuid

    context = cirq.DecompositionContext(
        qubit_manager=cirq.GreedyQubitManager(prefix=f'_{uuid.uuid4()}', maximize_reuse=True)
    )

    decomposed = _try_decompose_from_known_decompositions(val, context)
    if decomposed is not None:
        return decomposed

    if isinstance(val, cirq.Gate):
        decomposed = cirq.decompose_once_with_qubits(
            val, cirq.LineQid.for_gate(val), context=context, flatten=False, default=None
        )
    else:
        decomposed = cirq.decompose_once(val, context=context, flatten=False, default=None)

    return [*cirq.flatten_to_ops(decomposed)] if decomposed is not None else None
