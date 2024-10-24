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

from typing import Optional, TYPE_CHECKING, List, Tuple, FrozenSet, Union

from cirq import protocols, circuits
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq

_MAX_QUBIT_COUNT_FOR_MASK = 64


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
    all_operations = [*circuit.all_operations()]
    relative_order = {
        qs: i for i, qs in enumerate(sorted(set(tuple(sorted(op.qubits)) for op in all_operations)))
    }
    if len(circuit.all_qubits()) <= _MAX_QUBIT_COUNT_FOR_MASK:
        # use bitmasks.
        q_index = {q: i for i, q in enumerate(circuit.all_qubits())}

        def _msk(qs: Tuple['cirq.Qid', ...]) -> int:
            msk = 0
            for q in qs:
                msk |= 1 << q_index[q]
            return msk

        operations_with_info: Union[
            List[Tuple['cirq.Operation', int, int]], List[Tuple['cirq.Operation', int, FrozenSet]]
        ] = [
            (op, relative_order[tuple(sorted(op.qubits))], _msk(op.qubits)) for op in all_operations
        ]
    else:
        # use sets.
        operations_with_info = [
            (op, relative_order[tuple(sorted(op.qubits))], frozenset(op.qubits))
            for op in all_operations
        ]
    sorted_info: Union[
        List[Tuple['cirq.Operation', int, int]], List[Tuple['cirq.Operation', int, FrozenSet]]
    ] = []
    for i in range(len(all_operations)):
        j = len(sorted_info)
        while (
            j
            and operations_with_info[i][1] < sorted_info[j - 1][1]
            and (
                not (operations_with_info[i][2] & sorted_info[j - 1][2])  # type: ignore[operator]
                or protocols.commutes(
                    operations_with_info[i][0], sorted_info[j - 1][0], default=False
                )
            )
        ):
            j -= 1
        sorted_info.insert(j, operations_with_info[i])  # type: ignore[arg-type]
    return circuits.Circuit(op for op, _, _ in sorted_info)
