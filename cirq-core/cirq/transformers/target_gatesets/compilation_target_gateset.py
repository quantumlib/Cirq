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

from __future__ import annotations

import abc
from typing import Hashable, List, Optional, Type, TYPE_CHECKING, Union

from cirq import circuits, ops, protocols, transformers
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates

if TYPE_CHECKING:
    import cirq
    from cirq.protocols.decompose_protocol import DecomposeResult


def create_transformer_with_kwargs(transformer: cirq.TRANSFORMER, **kwargs) -> cirq.TRANSFORMER:
    """Method to capture additional keyword arguments to transformers while preserving mypy type.

    Returns a `cirq.TRANSFORMER` which, when called with a circuit and transformer context, is
    equivalent to calling `transformer(circuit, context=context, **kwargs)`. It is often useful to
    capture keyword arguments of a transformer before passing them as an argument to an API that
    expects `cirq.TRANSFORMER`. For example:

    >>> def run_transformers(transformers: list[cirq.TRANSFORMER]):
    ...     circuit = cirq.Circuit(cirq.X(cirq.q(0)))
    ...     context = cirq.TransformerContext()
    ...     for transformer in transformers:
    ...         transformer(circuit, context=context)
    ...
    >>> transformers: list[cirq.TRANSFORMER] = []
    >>> transformers.append(
    ...     cirq.create_transformer_with_kwargs(
    ...         cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= 2
    ...     )
    ... )
    >>> transformers.append(cirq.create_transformer_with_kwargs(cirq.merge_k_qubit_unitaries, k=2))
    >>> run_transformers(transformers)


    Args:
         transformer: A `cirq.TRANSFORMER` for which additional kwargs should be captured.
         **kwargs: The keyword arguments which should be captured and passed to `transformer`.

    Returns:
        A `cirq.TRANSFORMER` method `transformer_with_kwargs`, s.t. executing
        `transformer_with_kwargs(circuit, context=context)` is equivalent to executing
        `transformer(circuit, context=context, **kwargs)`.

    Raises:
        SyntaxError: if **kwargs contain a 'context'.
    """
    if 'context' in kwargs:
        raise SyntaxError('**kwargs to be captured must not contain `context`.')

    def transformer_with_kwargs(
        circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext] = None
    ) -> cirq.AbstractCircuit:
        return transformer(circuit, context=context, **kwargs)

    return transformer_with_kwargs


class CompilationTargetGateset(ops.Gateset, metaclass=abc.ABCMeta):
    """Abstract base class to create gatesets that can be used as targets for compilation.

    An instance of this type can be passed to transformers like `cirq.optimize_for_target_gateset`,
    which can transform any given circuit to contain gates accepted by this gateset.
    """

    def __init__(
        self,
        *gates: Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily],
        name: Optional[str] = None,
        unroll_circuit_op: bool = True,
        preserve_moment_structure: bool = True,
        reorder_operations: bool = False,
    ):
        """Initializes CompilationTargetGateset.

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
                `cirq.GateFamily` instances to initialize the Gateset.
            name: (Optional) Name for the Gateset. Useful for description.
            unroll_circuit_op: If True, `cirq.CircuitOperation` is recursively
                validated by validating the underlying `cirq.Circuit`.
            preserve_moment_structure: Whether to preserve the moment structure of the
                circuit during compilation or not.
            reorder_operations: Whether to attempt to reorder the operations in order to reduce
                circuit depth or not (can be True only if preserve_moment_structure=False).
        Raises:
            ValueError: If both reorder_operations and preserve_moment_structure are True.
        """
        if reorder_operations and preserve_moment_structure:
            raise ValueError(
                'reorder_operations and preserve_moment_structure can not both be True'
            )
        super().__init__(*gates, name=name, unroll_circuit_op=unroll_circuit_op)
        self._preserve_moment_structure = preserve_moment_structure
        self._reorder_operations = reorder_operations

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Maximum number of qubits on which a gate from this gateset can act upon."""

    @abc.abstractmethod
    def decompose_to_target_gateset(self, op: cirq.Operation, moment_idx: int) -> DecomposeResult:
        """Method to rewrite the given operation using gates from this gateset.

        Args:
            op: `cirq.Operation` to be rewritten using gates from this gateset.
            moment_idx: Moment index where the given operation `op` occurs in a circuit.

        Returns:
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from this gateset.
            - `None` or `NotImplemented` if does not know how to decompose `op`.
        """

    def _validate_operation(self, op: cirq.Operation) -> bool:
        """Validates whether the given `cirq.Operation` is contained in this Gateset.

        Overrides the method on the base gateset class to ensure that operations which created
        as intermediate compilation results are not accepted.
        For example, if a preprocessing `merge_k_qubit_unitaries` transformer merges connected
        component of 2q unitaries, it should not be accepted in the gateset so that so we can
        use `decompose_to_target_gateset` to determine how to expand this component.

        Args:
            op: The `cirq.Operation` instance to check containment for.

        Returns:
            Whether the given operation is contained in the gateset.
        """
        if self._intermediate_result_tag in op.tags:
            return False
        return super()._validate_operation(op)

    @property
    def _intermediate_result_tag(self) -> Hashable:
        """A tag used to identify intermediate compilation results."""
        return "_default_merged_k_qubit_unitaries"

    @property
    def preprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        """List of transformers which should be run before decomposing individual operations."""
        reorder_transfomers = (
            [transformers.insertion_sort_transformer] if self._reorder_operations else []
        )
        return [
            create_transformer_with_kwargs(
                transformers.expand_composite,
                no_decomp=lambda op: protocols.num_qubits(op) <= self.num_qubits,
            ),
            *reorder_transfomers,
            create_transformer_with_kwargs(
                merge_k_qubit_gates.merge_k_qubit_unitaries,
                k=self.num_qubits,
                rewriter=lambda op: op.with_tags(self._intermediate_result_tag),
            ),
        ]

    @property
    def postprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations."""
        processors: List[cirq.TRANSFORMER] = [
            merge_single_qubit_gates.merge_single_qubit_moments_to_phxz,
            transformers.drop_negligible_operations,
            transformers.drop_empty_moments,
        ]
        if not self._preserve_moment_structure:
            processors.append(transformers.stratified_circuit)
        return processors


class TwoQubitCompilationTargetGateset(CompilationTargetGateset):
    """Abstract base class to create two-qubit target gatesets.

    This base class can be used to create two-qubit compilation target gatesets. It automatically
    implements the logic to

        1. Apply `self.preprocess_transformers` to the input circuit, which by default will:
            a) Expand composite gates acting on > 2 qubits using `cirq.expand_composite`.
            b) Merge connected components of 1 & 2 qubit unitaries into tagged
                `cirq.CircuitOperation` using `cirq.merge_k_qubit_unitaries`.

        2. Apply `self.decompose_to_target_gateset` to rewrite each operation (including merged
        connected components from 1b) using gates from this gateset.
            a) Uses `self._decompose_single_qubit_operation`, `self._decompose_two_qubit_operation`
               and `self._decompose_multi_qubit_operation` to figure out how to rewrite (merged
               connected components of) operations using only gates from this gateset.
            b) A merged connected component containing only 1 & 2q gates from this gateset is
               replaced with a more efficient rewrite using `self._decompose_two_qubit_operation`
               iff the rewritten op-tree is lesser number of 2q interactions.

            Replace connected components with inefficient implementations (higher number of 2q
               interactions) with efficient rewrites to minimize total number of 2q interactions.

        3. Apply `self.postprocess_transformers` to the transformed circuit, which by default will:
            a) Apply `cirq.merge_single_qubit_moments_to_phxz` to preserve moment structure (eg:
               alternating layers of single/two qubit gates).
            b) Apply `cirq.drop_negligible_operations` and `cirq.drop_empty_moments` to minimize
               circuit depth.

    Derived classes should simply implement `self._decompose_two_qubit_operation` abstract method
    and provide analytical decomposition of any 2q unitary using gates from the target gateset.
    """

    @property
    def num_qubits(self) -> int:
        return 2

    def decompose_to_target_gateset(self, op: cirq.Operation, moment_idx: int) -> DecomposeResult:
        if not 1 <= protocols.num_qubits(op) <= 2:
            return self._decompose_multi_qubit_operation(op, moment_idx)
        if protocols.num_qubits(op) == 1:
            return self._decompose_single_qubit_operation(op, moment_idx)
        new_optree = self._decompose_two_qubit_operation(op, moment_idx)
        if new_optree is NotImplemented or new_optree is None:
            return new_optree
        new_optree = [*ops.flatten_to_ops_or_moments(new_optree)]
        op_untagged = op.untagged
        old_optree = (
            [*op_untagged.circuit]
            if isinstance(op_untagged, circuits.CircuitOperation)
            and self._intermediate_result_tag in op.tags
            else [op]
        )
        old_2q_gate_count = sum(1 for o in ops.flatten_to_ops(old_optree) if len(o.qubits) == 2)
        new_2q_gate_count = sum(1 for o in ops.flatten_to_ops(new_optree) if len(o.qubits) == 2)
        switch_to_new = (
            any(
                protocols.num_qubits(op) == 2 and op not in self
                for op in ops.flatten_to_ops(old_optree)
            )
            or new_2q_gate_count < old_2q_gate_count
        )
        if switch_to_new:
            return new_optree
        mapped_old_optree: List[cirq.OP_TREE] = []
        for old_op in ops.flatten_to_ops(old_optree):
            if old_op in self:
                mapped_old_optree.append(old_op)
            else:
                decomposed_op = self._decompose_single_qubit_operation(old_op, moment_idx)
                if decomposed_op is None or decomposed_op is NotImplemented:
                    return NotImplemented
                mapped_old_optree.append(decomposed_op)
        return mapped_old_optree

    def _decompose_single_qubit_operation(
        self, op: cirq.Operation, moment_idx: int
    ) -> DecomposeResult:
        """Decomposes (connected component of) 1-qubit operations using gates from this gateset.

        By default, rewrites every operation using a single `cirq.PhasedXZGate`.

        Args:
            op: A single-qubit operation (can be a tagged `cirq.CircuitOperation` wrapping
                a connected component of single qubit unitaries).
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
        return (
            ops.PhasedXZGate.from_matrix(protocols.unitary(op)).on(op.qubits[0])
            if protocols.has_unitary(op)
            else NotImplemented
        )

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, moment_idx: int
    ) -> DecomposeResult:
        """Decomposes operations acting on more than 2 qubits using gates from this gateset.

        Args:
            op: A multi qubit (>2q) operation.
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
        return NotImplemented

    @abc.abstractmethod
    def _decompose_two_qubit_operation(
        self, op: cirq.Operation, moment_idx: int
    ) -> DecomposeResult:
        """Decomposes (connected component of) 2-qubit operations using gates from this gateset.

        Args:
            op: A two-qubit operation (can be a tagged `cirq.CircuitOperation` wrapping
                a connected component of 1 & 2  qubit unitaries).
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
