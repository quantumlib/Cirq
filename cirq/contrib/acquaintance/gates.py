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

from functools import partial
from itertools import chain
from operator import indexOf
from typing import Iterable, Sequence, TYPE_CHECKING

from cirq import CompositeGate, TextDiagrammable
from cirq.ops import Operation, Gate, gate_features, QubitId, SWAP

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List, Tuple

class AcquaintanceOpportunityGate(Gate, TextDiagrammable):
    """Represents an acquaintance opportunity."""

    def __repr__(self):
        return 'Acq'

    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        if args.known_qubit_count is None:
            return NotImplemented
        wire_symbol = '█' if args.use_unicode_characters else 'Acq'
        wire_symbols = (wire_symbol,) * args.known_qubit_count
        return gate_features.TextDiagramInfo(
                wire_symbols=wire_symbols)

ACQUAINT = AcquaintanceOpportunityGate()

class PermutationGate(Gate):
    """Permutation gate."""
    pass

class CircularShiftGate(PermutationGate,
                        CompositeGate,
                        TextDiagrammable):
    """Swaps two sets of qubits.

    Args:
        shift: how many positions to circularly left shift the qubits.
        swap_gate: the gate to use when decomposing.
    """

    def __init__(self, 
                 shift: int,
                 swap_gate: Gate=SWAP) -> None:
        self.shift = shift
        self.swap_gate = swap_gate

    def __repr__(self):
        return 'CircularShiftGate'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return ((self.shift == other.shift) and
                (self.swap_gate == other.swap_gate))

    def default_decompose(self, qubits):
        n = len(qubits)
        left_shift = self.shift % n
        right_shift = n - left_shift
        mins = chain(range(left_shift - 1, 0, -1),
                     range(right_shift))
        maxs = chain(range(left_shift, n),
                     range(n - 1, right_shift, -1))
        for i, j in zip(mins, maxs):
            for k in range(i, j, 2):
                yield self.swap_gate(*qubits[k:k+2])

    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        if args.known_qubit_count is None:
            return NotImplemented
        direction_symbols = (
            ('╲', '╱') if args.use_unicode_characters else
            ('\\', '/'))
        wire_symbols = tuple(
                direction_symbols[int(i >= self.shift)] +
                str(i) +
                direction_symbols[int(i < self.shift)]
                for i in range(args.known_qubit_count))
        return gate_features.TextDiagramInfo(
                wire_symbols=wire_symbols)

CircularShiftGates = type('CircularShiftGates', (dict,), {'__missing__':
    (lambda self, key: self.setdefault(key, CircularShiftGate(key)))})
SHIFT = CircularShiftGates()

class SwapNetworkGate(Gate, CompositeGate, TextDiagrammable):
    """A single gate representing a generalized swap network.

    Args:
        part_lens: An iterable indicating the sizes of the parts in the
            partition defining the swap network.
        acquaintance_size: An int indicating the locality of the logical gates
            desired; used to keep track of this while nesting. If 0, no
            acquaintance gates are inserted.
    """

    def __init__(self, 
                 part_lens: Iterable[int], 
                 acquaintance_size: int=0) -> None:
        self.part_lens = tuple(part_lens)
        self.acquaintance_size = acquaintance_size

    def default_decompose(self, qubits):
        qubit_to_position = {q: i for i, q in enumerate(qubits)}
        parts = []
        q = 0
        for part_len in self.part_lens:
            parts.append(tuple(qubits[q: q + part_len]))
            q += part_len
        n_parts = len(parts)
        op_sort_key = (lambda op: 
                qubit_to_position[min(op.qubits, key=qubit_to_position.get)] %
                self.acquaintance_size)
        post_layer = []
        for layer_num in range(n_parts):
            pre_layer, layer, post_layer = post_layer, [], []
            for i in range(layer_num % 2, n_parts - 1, 2):
                left_part, right_part = parts[i:i+2]
                parts_qubits = left_part + right_part
                multiplicities = (len(left_part), len(right_part))
                shift = SHIFT[multiplicities[0]](*parts_qubits)
                if max(multiplicities) != self.acquaintance_size - 1:
                    layer.append(shift)
                elif self.acquaintance_size == 2:
                    pre_layer.append(ACQUAINT(*parts_qubits))
                    layer.append(shift)
                else:
                    if self.acquaintance_size > 3:
                        raise NotImplementedError()
                    layer.append(shift)
                    if multiplicities[0] == self.acquaintance_size - 1:
                        pre_qubits = parts_qubits[:self.acquaintance_size]
                        pre_layer.append(ACQUAINT(*pre_qubits))
                        post_qubits = parts_qubits[-self.acquaintance_size:]
                        post_layer.append(ACQUAINT(*post_qubits))
                    if multiplicities[1] == self.acquaintance_size - 1:
                        pre_qubits = parts_qubits[-self.acquaintance_size:]
                        pre_layer.append(ACQUAINT(*pre_qubits))
                        post_qubits = parts_qubits[:self.acquaintance_size]
                        post_layer.append(ACQUAINT(*post_qubits))
                
                parts[i] = parts_qubits[:multiplicities[1]]
                parts[i + 1] = parts_qubits[multiplicities[1]:]
            pre_layer.sort(key=op_sort_key)
            for op in pre_layer:
                yield op
            for op in layer:
                yield op
        post_layer.sort(key=op_sort_key)
        for op in post_layer:
            yield op
        for part in parts:
            if len(part) > 1:
                yield SwapNetworkGate((1,) * len(part))(*part)

    def text_diagram_info(self,
                          args: gate_features.TextDiagramInfoArgs):
        wire_symbol = ('×' if args.use_unicode_characters else 'swap')
        wire_symbols = tuple(
            wire_symbol + '({},{})'.format(part_index, qubit_index)
            for part_index, part_len in enumerate(self.part_lens)
            for qubit_index in range(part_len))
        return gate_features.TextDiagramInfo(
            wire_symbols=wire_symbols)

    @staticmethod
    def from_operations(qubit_order: Sequence[QubitId],
                        operations: Sequence[Operation],
                        acquaintance_size: int=0
                        ) -> 'SwapNetworkGate':
        qubit_sort_key = partial(indexOf, qubit_order)
        op_parts = [tuple(sorted(op.qubits,key=qubit_sort_key))
                    for op in operations]
        singletons = [(q,) for q in set(qubit_order).difference(*op_parts)
                     ] # type: List[Tuple[QubitId, ...]]
        part_sort_key = lambda p: min(qubit_sort_key(q) for q in p)
        parts = tuple(tuple(part) for part in 
                      sorted(singletons + op_parts, key=part_sort_key))
        part_sizes = tuple(len(part) for part in parts)

        assert sum(parts, ()) == tuple(qubit_order)

        return SwapNetworkGate(part_sizes, acquaintance_size)
