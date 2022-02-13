# Copyright 2018 The Cirq Developers
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

"""An optimization pass that combines adjacent single-qubit rotations."""

from typing import Optional, Callable, List, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


class MergeSingleQubitGates(circuits.PointOptimizer):
    """Optimizes runs of adjacent unitary 1-qubit operations."""

    def __init__(
        self,
        *,
        rewriter: Optional[Callable[[List[ops.Operation]], Optional[ops.OP_TREE]]] = None,
        synthesizer: Optional[Callable[[ops.Qid, np.ndarray], Optional[ops.OP_TREE]]] = None,
    ):
        """Inits MergeSingleQubitGates.

        Args:
            rewriter: Specifies how to merge runs of single-qubit operations
                into a more desirable form. Takes a list of operations and
                produces a list of operations. The default rewriter computes the
                matrix of the run and returns a `cirq.SingleQubitMatrixGate`. If
                `rewriter` returns `None`, that means "do not rewrite the
                operations".
            synthesizer: A special kind of rewriter that operates purely on
                the unitary matrix of the intended operation. Takes a qubit
                and a unitary matrix and returns a list of operations. Can't
                be specified at the same time as `rewriter`. If `synthesizer`
                returns `None`, that means "do not rewrite the operations used
                to make this matrix".

        Raises:
            ValueError: If both a rewriter and synthesizer were specified.
        """
        super().__init__()
        if rewriter is not None and synthesizer is not None:
            raise ValueError("Can't specify both rewriter and synthesizer.")
        self._rewriter = rewriter
        self._synthesizer = synthesizer

    def _rewrite(self, operations: List[ops.Operation]) -> Optional[ops.OP_TREE]:
        if not operations:
            return None
        q = operations[0].qubits[0]

        # Custom rewriter?
        if self._rewriter is not None:
            return self._rewriter(operations)

        unitary = linalg.dot(*(protocols.unitary(op) for op in operations[::-1]))

        # Custom synthesizer?
        if self._synthesizer is not None:
            return self._synthesizer(q, unitary)

        # Just use the default.
        return ops.MatrixGate(unitary).on(q)

    def optimization_at(
        self, circuit: circuits.Circuit, index: int, op: ops.Operation
    ) -> Optional[circuits.PointOptimizationSummary]:
        if len(op.qubits) != 1:
            return None
        start = {op.qubits[0]: index}

        op_list = circuit.findall_operations_until_blocked(
            start,
            is_blocker=lambda next_op: len(next_op.qubits) != 1
            or not protocols.has_unitary(next_op),
        )
        operations = [op for idx, op in op_list]
        indices = [idx for idx, op in op_list]

        rewritten = self._rewrite(operations)

        if rewritten is None:
            return None
        return circuits.PointOptimizationSummary(
            clear_span=max(indices) + 1 - index, clear_qubits=op.qubits, new_operations=rewritten
        )


def merge_single_qubit_gates_into_phased_x_z(circuit: circuits.Circuit, atol: float = 1e-8) -> None:
    """Canonicalizes runs of single-qubit rotations in a circuit.

    Specifically, any run of non-parameterized single-qubit gates will be
    replaced by an optional PhasedX operation followed by an optional Z
    operation.

    Args:
        circuit: The circuit to rewrite. This value is mutated in-place.
        atol: Absolute tolerance to angle error. Larger values allow more
            negligible gates to be dropped, smaller values increase accuracy.
    """

    def synth(qubit: 'cirq.Qid', matrix: np.ndarray) -> List[ops.Operation]:
        out_gates = single_qubit_decompositions.single_qubit_matrix_to_phased_x_z(matrix, atol)
        return [gate(qubit) for gate in out_gates]

    MergeSingleQubitGates(synthesizer=synth).optimize_circuit(circuit)


def merge_single_qubit_gates_into_phxz(
    circuit: circuits.Circuit,
    atol: float = 1e-8,
) -> None:
    """Canonicalizes runs of single-qubit rotations in a circuit.

    Specifically, any run of non-parameterized single-qubit gates will be
    replaced by an optional PhasedXZ operation.

    Args:
        circuit: The circuit to rewrite. This value is mutated in-place.
        atol: Absolute tolerance to angle error. Larger values allow more
            negligible gates to be dropped, smaller values increase accuracy.
    """

    def synth(qubit: 'cirq.Qid', matrix: np.ndarray) -> List[ops.Operation]:
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(matrix, atol)
        return [gate(qubit)] if gate else []

    MergeSingleQubitGates(synthesizer=synth).optimize_circuit(circuit)
