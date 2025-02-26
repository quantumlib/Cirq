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

import enum
import warnings
from typing import Optional, TYPE_CHECKING


from cirq import circuits, ops, protocols
from cirq.transformers import merge_k_qubit_gates, transformer_api, transformer_primitives
from cirq.study import sweepable
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


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

    def rewriter(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        u = protocols.unitary(op)
        if protocols.num_qubits(op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        return [
            g(op.qubits[0])
            for g in single_qubit_decompositions.single_qubit_matrix_to_phased_x_z(u, atol)
        ]

    return merge_k_qubit_gates.merge_k_qubit_unitaries(
        circuit, k=1, context=context, rewriter=rewriter
    )


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

    def rewriter(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        u = protocols.unitary(op)
        if protocols.num_qubits(op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol)
        return gate(op.qubits[0]) if gate else []

    return merge_k_qubit_gates.merge_k_qubit_unitaries(
        circuit, k=1, context=context, rewriter=rewriter
    )


@transformer_api.transformer
def merge_single_qubit_moments_to_phxz(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Merges adjacent moments with only 1-qubit rotations to a single moment with PhasedXZ gates.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """
    tags_to_ignore = set(context.tags_to_ignore) if context else set()

    def can_merge_moment(m: 'cirq.Moment'):
        return all(
            protocols.num_qubits(op) == 1
            and protocols.has_unitary(op)
            and tags_to_ignore.isdisjoint(op.tags)
            for op in m
        )

    def merge_func(m1: 'cirq.Moment', m2: 'cirq.Moment') -> Optional['cirq.Moment']:
        if not (can_merge_moment(m1) and can_merge_moment(m2)):
            return None
        ret_ops = []
        for q in m1.qubits | m2.qubits:
            op1, op2 = m1.operation_at(q), m2.operation_at(q)
            if op1 and op2:
                mat = protocols.unitary(op2) @ protocols.unitary(op1)
                gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat, atol)
                if gate:
                    ret_ops.append(gate(q))
            else:
                op = op1 or op2
                assert op is not None
                if isinstance(op.gate, ops.PhasedXZGate):
                    ret_ops.append(op)
                else:
                    gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(
                        protocols.unitary(op), atol
                    )
                    if gate:
                        ret_ops.append(gate(q))
        return circuits.Moment(ret_ops)

    return transformer_primitives.merge_moments(
        circuit,
        merge_func,
        deep=context.deep if context else False,
        tags_to_ignore=tuple(tags_to_ignore),
    ).unfreeze(copy=False)


@transformer_api.transformer
def merge_into_symbolized_phxz(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    sweeps: Optional['sweepable.Sweepable'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Merge consecutive single qubit gates into connected symbolized PhasedXZ gates.

    Specifically, if at least one of the consecutive gates is symbolized, then the merged gate
    will be a symbolized gate.

    e.g., X-Y-H-phxz(sa, sx, sz) ---transform---> phxz(sa, sx, sz)

    Note, we only consider merging non-parameterized gates to symbolized phxz with
     3 degrees of freedom, meaning that gates like Z^exp_symbol will be considered non-mergable.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        sweeps: Sweeps of the symbols in the input circuit, updated Sweeps will be returned
            based on the transformation.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    # TODO(#6994): support returning update sweeps when sweeps are provided.
    if sweeps is not None:
        raise NotImplementedError("To be supported in #6994.")

    if not protocols.is_parameterized(circuit):
        warnings.warn(
            "Expect parameterized circuits. "
            "Please use cirq.merge_single_qubit_gates_to_phxz instead.",
            UserWarning,
        )
        return merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol)

    # Merge all non parameterized single qubit gates first.
    circuit = merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol)

    def _merge_func(op1: 'cirq.Operation', op2: 'cirq.Operation'):

        class _MergeGateType(enum.Enum):
            MERAGABLE_NON_PARAMETERIZED = 0
            MERAGABLE_PARAMETERIZED_PHXZ = 1
            NON_MERGEABLE = 2

        def _categorize(op: 'cirq.Operation') -> _MergeGateType:
            if protocols.has_unitary(op) and protocols.num_qubits(op) == 1:
                return _MergeGateType.MERAGABLE_NON_PARAMETERIZED
            if isinstance(op.gate, ops.PhasedXZGate) and protocols.is_parameterized(op):
                return _MergeGateType.MERAGABLE_PARAMETERIZED_PHXZ
            return _MergeGateType.NON_MERGEABLE

        merge_type1 = _categorize(op1)
        merge_type2 = _categorize(op2)

        if (
            merge_type1 == _MergeGateType.NON_MERGEABLE
            or merge_type2 == _MergeGateType.NON_MERGEABLE
        ):
            return None

        # absorb the non-parameterized gate into the parameterized gate.
        if merge_type1 == _MergeGateType.MERAGABLE_PARAMETERIZED_PHXZ:
            return op1
        if merge_type2 == _MergeGateType.MERAGABLE_PARAMETERIZED_PHXZ:
            return op2

        return None  # pragma: no cover

    return transformer_primitives.merge_operations(
        circuit,
        _merge_func,
        deep=context.deep if context else False,
        tags_to_ignore=context.tags_to_ignore if context else (),
    ).unfreeze()
