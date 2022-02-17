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

"""Base class for creating custom target gatesets which can be used for compilation."""

from typing import Optional, List, Hashable, TYPE_CHECKING
import abc

from cirq import circuits, ops, protocols, _import
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import (
    merge_k_qubit_gates,
    merge_single_qubit_gates,
)

drop_empty_moments = _import.LazyLoader('drop_empty_moments', globals(), 'cirq.transformers')
drop_negligible = _import.LazyLoader('drop_negligible_operations', globals(), 'cirq.transformers')
expand_composite = _import.LazyLoader('expand_composite', globals(), 'cirq.transformers')

if TYPE_CHECKING:
    import numpy as np
    import cirq


def _create_transformer_with_kwargs(func: 'cirq.TRANSFORMER', **kwargs) -> 'cirq.TRANSFORMER':
    """Hack to capture additional keyword arguments to transformers while preserving mypy type."""

    def transformer(
        circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
    ) -> 'cirq.AbstractCircuit':
        return func(circuit, context=context, **kwargs)  # type: ignore

    return transformer


class CompilationTargetGateset(ops.Gateset, metaclass=abc.ABCMeta):
    """Abstract base class to create gatesets that can be used as targets for compilation.

    An instance of this type can be passed to transformers like `cirq.convert_to_target_gateset`,
    which can transform any given circuit to contain gates accepted by this gateset.
    """

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Maximum number of qubits on which a gate from this gateset can act upon."""

    @abc.abstractmethod
    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Method to rewrite the given operation using gates from this gateset.

        Args:
            op: `cirq.Operation` to be rewritten using gates from this gateset.
            moment_idx: Moment index where the given operation `op` occurs in a circuit.

        Returns:
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from this gateset.
            - `None` or `NotImplemented` if does not know how to decompose `op`.
        """

    @property
    def _intermediate_result_tag(self) -> Hashable:
        """A tag used to identify intermediate compilation results."""
        return "_default_merged_k_qubit_unitaries"

    @property
    def preprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run before decomposing individual operations."""
        return [
            _create_transformer_with_kwargs(
                expand_composite.expand_composite,
                no_decomp=lambda op: protocols.num_qubits(op) <= self.num_qubits,
            ),
            _create_transformer_with_kwargs(
                merge_k_qubit_gates.merge_k_qubit_unitaries,
                k=self.num_qubits,
                rewriter=lambda op: op.with_tags(self._intermediate_result_tag),
            ),
        ]

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [
            merge_single_qubit_gates.merge_single_qubit_moments_to_phxz,
            drop_negligible.drop_negligible_operations,
            drop_empty_moments.drop_empty_moments,
        ]


class TwoQubitAnalyticalCompilationTarget(CompilationTargetGateset):
    """Abstract base class to create two-qubit target gateset with known analytical decompositions.

    The class is useful to create 2-qubit target gatesets where an analytical method, like KAK
    Decomposition, can be used to decompose any arbitrary 2q unitary matrix into a sequence of
    gates from this gateset.

    Derived classes should simply implement `_decompose_two_qubit_matrix_to_operations` method.

    Note: The implementation assumes that any single qubit rotation is accepted by this gateset.
    """

    @property
    def num_qubits(self) -> int:
        return 2

    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        if self._intermediate_result_tag not in op.tags:
            return NotImplemented
        if protocols.num_qubits(op) == 1:
            return op
        op_untagged = op.untagged
        assert protocols.num_qubits(op) == 2
        assert isinstance(op_untagged, circuits.CircuitOperation)
        switch_to_new = any(op not in self for op in op_untagged.circuit.all_operations())
        old_2q_gate_count = len(
            [*op_untagged.circuit.findall_operations(lambda o: len(o.qubits) == 2)]
        )
        new_optree = self._decompose_two_qubit_matrix_to_operations(
            op.qubits[0], op.qubits[1], protocols.unitary(op)
        )
        new_2q_gate_count = len([o for o in ops.flatten_to_ops(new_optree) if len(o.qubits) == 2])
        switch_to_new |= new_2q_gate_count < old_2q_gate_count
        return new_optree if switch_to_new else [*op_untagged.circuit]

    @abc.abstractmethod
    def _decompose_two_qubit_matrix_to_operations(
        self, q0: 'cirq.Qid', q1: 'cirq.Qid', mat: 'np.ndarray'
    ) -> 'cirq.OP_TREE':
        """Decomposes the given 2-qubit unitary matrix into a sequence of gates from this gateset.

        Args:
            q0: The first qubit being operated on.
            q1: The other qubit being operated on.
            mat: Unitary matrix of two qubit operation to apply to the given pair of qubits.

        Returns:
            A `cirq.OP_TREE` implementing `cirq.MatrixGate(mat).on(q0, q1)` using gates from this
            gateset.
        """
