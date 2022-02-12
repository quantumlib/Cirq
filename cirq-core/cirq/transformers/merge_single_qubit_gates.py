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

"""Transformer passes to combine adjacent single-qubit rotations."""

from typing import Optional, Callable, List, TYPE_CHECKING

import numpy as np

from cirq import ops, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def merge_single_qubit_gates(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    rewriter: Optional[Callable[[List[ops.Operation]], ops.OP_TREE]] = None,
    synthesizer: Optional[Callable[[ops.Qid, np.ndarray], ops.OP_TREE]] = None,
) -> 'cirq.Circuit':
    """Merges adjacent single qubit unitaries in the given circuit.

    Uses rewriter or synthesiser to convert a connected component of single qubit unitaries
    into a more desirable form. At-most one of rewriter or synthesiser should be specified
    at a time. If both are not specified, connected components are replaced by a single
    `cirq.MatrixGate` containing unitary of the merged component.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        rewriter: Specifies how to merge runs of single-qubit operations into a more desirable
            form. Takes a list of operations and produces a list of operations.
        synthesizer: A special kind of rewriter that operates purely on the unitary matrix of the
            intended operation. Takes a qubit and a unitary matrix and returns a list of operations.

    Returns:
        Copy of the transformed input circuit.

    Raises:
        ValueError: If both rewriter and synthesizer were specified.
    """
    if rewriter is not None and synthesizer is not None:
        raise ValueError("Can't specify both rewriter and synthesizer.")
    merged_circuit_op_tag = "_merged_single_qubit_gates_component"

    def map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if not (protocols.num_qubits(op) == 1 and protocols.has_unitary(op)):
            return op
        op_untagged = op.untagged
        if rewriter:
            return rewriter(
                [*op_untagged.circuit.all_operations()]
                if isinstance(op_untagged, circuits.CircuitOperation)
                and merged_circuit_op_tag in op.tags
                else [op]
            )
        elif synthesizer:
            return synthesizer(op.qubits[0], protocols.unitary(op))
        else:
            return ops.MatrixGate(protocols.unitary(op)).on(*op.qubits)

    circuit = transformer_primitives.merge_k_qubit_unitaries_to_circuit_op(
        circuit,
        k=1,
        tags_to_ignore=context.tags_to_ignore if context else (),
        merged_circuit_op_tag=merged_circuit_op_tag,
    )
    return transformer_primitives.map_operations_and_unroll(
        circuit, map_func, tags_to_ignore=context.tags_to_ignore if context else ()
    ).unfreeze(copy=False)


@transformer_api.transformer
def merge_single_qubit_gates_to_phased_x_and_z(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with `cirq.PhasedXPowGate` and `cirq.ZPowGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedX operation followed by an optional Z operation.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    def synthesizer(q: 'cirq.Qid', mat: np.ndarray) -> 'cirq.OP_TREE':
        return [
            gate(q)
            for gate in single_qubit_decompositions.single_qubit_matrix_to_phased_x_z(mat, atol)
        ]

    return merge_single_qubit_gates(circuit, context=context, synthesizer=synthesizer)


@transformer_api.transformer
def merge_single_qubit_gates_to_phxz(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with a single optional `cirq.PhasedXZGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedXZ.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.
    Returns:
        Copy of the transformed input circuit.
    """

    def synthesizer(q: 'cirq.Qid', mat: np.ndarray) -> 'cirq.OP_TREE':
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat, atol)
        return gate(q) if gate else []

    return merge_single_qubit_gates(circuit, context=context, synthesizer=synthesizer)
