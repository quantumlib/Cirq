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

from typing import Optional, TYPE_CHECKING, List, Tuple

from cirq import protocols, circuits
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


def _id(op: 'cirq.Operation') -> Tuple['cirq.Qid', ...]:
    return tuple(sorted(op.qubits))


@transformer_api.transformer(add_deep_support=True)
def insertion_sort_transformer(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> 'cirq.Circuit':
    """Sorts the operations using their `.qubits` property as comparison key.

    Operations are swapped only if they commute.

    Args:
        circuit: input circuit.
        context: optional TransformerContext (not used),
    """
    sorted_operations: List['cirq.Operation'] = []
    for op in circuit.all_operations():
        sorted_operations.append(op)
        j = len(sorted_operations) - 1
        while (
            j
            and _id(sorted_operations[j]) < _id(sorted_operations[j - 1])
            and protocols.commutes(sorted_operations[j], sorted_operations[j - 1], default=False)
        ):
            sorted_operations[j], sorted_operations[j - 1] = (
                sorted_operations[j - 1],
                sorted_operations[j],
            )
            j -= 1
    return circuits.Circuit(sorted_operations)