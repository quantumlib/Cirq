# Copyright 2024 The Cirq Developers
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

"""Transformer that sorts commuting operations in increasing order of their `.qubits` tuple."""

from typing import Dict, List, Optional, TYPE_CHECKING

from cirq import circuits, protocols
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer(add_deep_support=True)
def insertion_sort_transformer(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> 'cirq.Circuit':
    """Sorts the operations using their sorted `.qubits` property as comparison key.

    Operations are swapped only if they commute.

    Args:
        circuit: input circuit.
        context: optional TransformerContext (not used),
    """
    final_operations: List['cirq.Operation'] = []
    qubit_index: Dict['cirq.Qid', int] = {
        q: idx for idx, q in enumerate(sorted(circuit.all_qubits()))
    }
    cached_qubit_indices: Dict[int, List[int]] = {}
    for pos, op in enumerate(circuit.all_operations()):
        # here `pos` is at the append position of final_operations
        if (op_qubit_indices := cached_qubit_indices.get(id(op))) is None:
            op_qubit_indices = cached_qubit_indices[id(op)] = sorted(
                qubit_index[q] for q in op.qubits
            )
        for tail_op in reversed(final_operations):
            tail_qubit_indices = cached_qubit_indices[id(tail_op)]
            if op_qubit_indices < tail_qubit_indices and (
                # special case for zero-qubit gates
                not op_qubit_indices
                # check if two sorted sequences are disjoint
                or op_qubit_indices[-1] < tail_qubit_indices[0]
                or set(op_qubit_indices).isdisjoint(tail_qubit_indices)
                # fallback to more expensive commutation check
                or protocols.commutes(op, tail_op, default=False)
            ):
                pos -= 1
                continue
            break
        final_operations.insert(pos, op)
    return circuits.Circuit(final_operations)
