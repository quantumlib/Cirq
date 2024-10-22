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
    operations_with_key: List[Tuple[Tuple['cirq.Qid', ...], 'cirq.Operation']] = [
        (_id(op), op) for op in circuit.all_operations()
    ]
    for i in range(len(operations_with_key)):
        j = i
        while (
            j
            and operations_with_key[j][0] < operations_with_key[j - 1][0]
            and protocols.commutes(
                operations_with_key[j][1], operations_with_key[j - 1][1], default=False
            )
        ):
            operations_with_key[j], operations_with_key[j - 1] = (
                operations_with_key[j - 1],
                operations_with_key[j],
            )
            j -= 1
    return circuits.Circuit(op for _, op in operations_with_key)
