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

"""Transformer pass to merge connected components of k-qubit unitary operations."""

from typing import Callable, cast, Optional, TYPE_CHECKING

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


def _rewrite_merged_k_qubit_unitaries(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    k: int = 0,
    rewriter: Optional[Callable[['cirq.CircuitOperation'], 'cirq.OP_TREE']] = None,
    merged_circuit_op_tag: str = "_merged_k_qubit_unitaries_component",
) -> 'cirq.Circuit':
    deep = context.deep if context else False

    def map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        op_untagged = op.untagged
        if (
            deep
            and isinstance(op_untagged, circuits.CircuitOperation)
            and merged_circuit_op_tag not in op.tags
        ):
            return op_untagged.replace(
                circuit=_rewrite_merged_k_qubit_unitaries(
                    op_untagged.circuit,
                    context=context,
                    k=k,
                    rewriter=rewriter,
                    merged_circuit_op_tag=merged_circuit_op_tag,
                ).freeze()
            ).with_tags(*op.tags)
        if not (protocols.num_qubits(op) <= k and protocols.has_unitary(op)):
            return op
        if rewriter:
            return rewriter(
                cast(circuits.CircuitOperation, op_untagged)
                if merged_circuit_op_tag in op.tags
                else circuits.CircuitOperation(circuits.FrozenCircuit(op))
            )
        return ops.MatrixGate(protocols.unitary(op)).on(*op.qubits)

    return transformer_primitives.map_operations_and_unroll(
        circuit, map_func, tags_to_ignore=context.tags_to_ignore if context else ()
    ).unfreeze(copy=False)


@transformer_api.transformer
def merge_k_qubit_unitaries(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    k: int = 0,
    rewriter: Optional[Callable[['cirq.CircuitOperation'], 'cirq.OP_TREE']] = None,
) -> 'cirq.Circuit':
    """Merges connected components of unitary operations, acting on <= k qubits.

    Uses rewriter to convert a connected component of unitary operations acting on <= k-qubits
    into a more desirable form. If not specified, connected components are replaced by a single
    `cirq.MatrixGate` containing unitary matrix of the merged component.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        k: Connected components of unitary operations acting on <= k qubits are merged.
        rewriter: Callable type that takes a `cirq.CircuitOperation`, encapsulating a connected
            component of unitary operations acting on <= k qubits, and produces a `cirq.OP_TREE`.
            Specifies how to merge the connected component into a more desirable form.

    Returns:
        Copy of the transformed input circuit.

    Raises:
        ValueError: If k <= 0
    """
    if k <= 0:
        raise ValueError(f"k should be greater than or equal to 1. Found {k}.")
    merged_circuit_op_tag = "_merged_k_qubit_unitaries_component"
    circuit = transformer_primitives.merge_k_qubit_unitaries_to_circuit_op(
        circuit,
        k=k,
        tags_to_ignore=context.tags_to_ignore if context else (),
        merged_circuit_op_tag=merged_circuit_op_tag,
        deep=context.deep if context else False,
    )
    return _rewrite_merged_k_qubit_unitaries(
        circuit,
        context=context,
        k=k,
        rewriter=rewriter,
        merged_circuit_op_tag=merged_circuit_op_tag,
    )
