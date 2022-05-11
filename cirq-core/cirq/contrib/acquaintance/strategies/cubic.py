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

import itertools
from typing import Iterable, Sequence, Tuple, TypeVar, TYPE_CHECKING

from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import LinearPermutationGate, SwapPermutationGate

if TYPE_CHECKING:
    import cirq

TItem = TypeVar('TItem')


def skip_and_wrap_around(items: Sequence[TItem]) -> Tuple[TItem, ...]:
    n_items = len(items)
    positions = {
        p: i
        for i, p in enumerate(itertools.chain(range(0, n_items, 2), reversed(range(1, n_items, 2))))
    }
    return tuple(items[positions[i]] for i in range(n_items))


def cubic_acquaintance_strategy(
    qubits: Iterable['cirq.Qid'], swap_gate: 'cirq.Gate' = ops.SWAP
) -> 'cirq.Circuit':
    """Acquaints every triple of qubits.

    Exploits the fact that in a simple linear swap network every pair of
    logical qubits that starts at distance two remains so (except temporarily
    near the edge), and that every third one `goes through` the pair at some
    point in the network. The strategy then iterates through a series of
    mappings in which qubits i and i + k are placed at distance two, for k = 1
    through n / 2. Linear swap networks are used in between to effect the
    permutation.
    """

    qubits = tuple(qubits)
    n_qubits = len(qubits)

    swap_gate = SwapPermutationGate(swap_gate)

    moments = []
    index_order = tuple(range(n_qubits))
    max_separation = max(((n_qubits - 1) // 2) + 1, 2)
    for separation in range(1, max_separation):
        stepped_indices_concatenated = tuple(
            itertools.chain(*(range(offset, n_qubits, separation) for offset in range(separation)))
        )
        new_index_order = skip_and_wrap_around(stepped_indices_concatenated)
        permutation = {i: new_index_order.index(j) for i, j in enumerate(index_order)}
        permutation_gate = LinearPermutationGate(n_qubits, permutation, swap_gate)
        moments.append(circuits.Moment([permutation_gate(*qubits)]))
        for i in range(n_qubits + 1):
            for offset in range(3):
                moment = circuits.Moment(
                    acquaint(*qubits[j : j + 3]) for j in range(offset, n_qubits - 2, 3)
                )
                moments.append(moment)
            if i < n_qubits:
                moment = circuits.Moment(
                    swap_gate(*qubits[j : j + 2]) for j in range(i % 2, n_qubits - 1, 2)
                )
                moments.append(moment)
        index_order = new_index_order[::-1]
    return circuits.Circuit(moments)
