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
from math import ceil
from operator import indexOf
from typing import Sequence, TYPE_CHECKING, Dict

from cirq import CompositeGate, TextDiagrammable
from cirq.ops import (
        Operation, Gate, gate_features, QubitId, OP_TREE, SWAP, GateOperation)
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
        PermutationGate, SwapPermutationGate, LinearPermutationGate)

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

class SwapNetworkGate(CompositeGate, PermutationGate):
    """A single gate representing a generalized swap network.

    Args:
        part_lens: An sequence indicating the sizes of the parts in the
            partition defining the swap network.
        acquaintance_size: An int indicating the locality of the logical gates
            desired; used to keep track of this while nesting. If 0, no
            acquaintance gates are inserted.
    """

    def __init__(self,
                 part_lens: Sequence[int],
                 acquaintance_size: int=0,
                 swap_gate: Gate=SWAP
                 ) -> None:
        if len(part_lens) < 2:
            raise ValueError('part_lens must have length at least 2.')
        self.part_lens = tuple(part_lens)
        self.acquaintance_size = acquaintance_size
        self.swap_gate = swap_gate

    def acquaint_insides(self,
                         acquaintance_gate: Operation,
                         part: Sequence[QubitId],
                         max_reach: int,
                         acquaint_first: bool
                         ) -> OP_TREE:
        reaches = chain(range(1, max_reach + 1),
                        range(max_reach, -1, -1))
        offsets = (0, 1) * max_reach
        swap_gate = SwapPermutationGate(self.swap_gate)
        for offset, reach in zip(offsets, reaches):
            if offset != acquaint_first:
                yield acquaintance_gate
            for dr in range(offset, reach, 2):
                yield swap_gate(*part[dr:dr + 2])


    def default_decompose(self, qubits: Sequence[QubitId]) -> OP_TREE:
        qubit_to_position = {q: i for i, q in enumerate(qubits)}
        mapping = {q: i for i, q in enumerate(qubits)}
        parts = []
        q = 0
        for part_len in self.part_lens:
            parts.append(list(qubits[q: q + part_len]))
            q += part_len
        n_parts = len(parts)
        op_sort_key = (lambda op:
                qubit_to_position[min(op.qubits, key=qubit_to_position.get)] %
                self.acquaintance_size)
        posterior_interstitial_layer = [] # type: List[Operation]
        for layer_num in range(n_parts):
            prior_interstitial_layer = posterior_interstitial_layer
            pre_layer = []
            layer = []
            post_layer = []
            posterior_interstitial_layer = []
            for i in range(layer_num % 2, n_parts - 1, 2):
                left_part, right_part = parts[i:i+2]
                parts_qubits = list(left_part + right_part)
                multiplicities = (len(left_part), len(right_part))
                shift = CircularShiftGate(multiplicities[0],
                                          swap_gate=self.swap_gate)(
                                                  *parts_qubits)
                if max(multiplicities) != self.acquaintance_size - 1:
                    layer.append(shift)
                elif self.acquaintance_size == 2:
                    prior_interstitial_layer.append(ACQUAINT(*parts_qubits))
                    layer.append(shift)
                else:

                    # before
                    if multiplicities[0] == self.acquaintance_size - 1:
                        # right part
                        pre_qubits = parts_qubits[:self.acquaintance_size]
                        pre_acquaintance_gate = ACQUAINT(*pre_qubits)

                        prior_interstitial_layer.append(pre_acquaintance_gate)


                        max_reach = int(ceil(multiplicities[1] / 2)) - 1
                        pre_layer.append(
                                self.acquaint_insides(pre_acquaintance_gate,
                                    right_part, max_reach, False))
                        reached_qubits = right_part[:max_reach + 1]
                        positions = list(mapping[q] for q in reached_qubits)
                        mapping.update(zip(reached_qubits, reversed(positions)))

                    if multiplicities[1] == self.acquaintance_size - 1:
                        # left part
                        pre_qubits = parts_qubits[-self.acquaintance_size:]
                        pre_acquaintance_gate = ACQUAINT(*pre_qubits)

                        prior_interstitial_layer.append(pre_acquaintance_gate)

                        max_reach = int(ceil(multiplicities[0] / 2)) - 1
                        pre_layer.append(
                                self.acquaint_insides(pre_acquaintance_gate,
                                    left_part[::-1], max_reach, False))

                        reached_qubits = left_part[::-1][:max_reach + 1]
                        positions = list(mapping[q] for q in reached_qubits)
                        mapping.update(zip(reached_qubits, reversed(positions)))

                    layer.append(shift)
                    shift.gate.update_mapping(mapping, parts_qubits)

                    # after
                    if ((multiplicities[0] == self.acquaintance_size - 1) and
                        (multiplicities[1] > 1)):
                        # right part
                        post_qubits = parts_qubits[-self.acquaintance_size:]
                        post_acquaintance_gate = ACQUAINT(*post_qubits)

                        new_left_part = parts_qubits[multiplicities[1] - 1::-1]
                        max_reach = max((multiplicities[1] // 2) - 1, 0)
                        post_layer.append(
                                self.acquaint_insides(post_acquaintance_gate,
                                    new_left_part, max_reach, True))
                        reached_qubits = new_left_part[:max_reach + 1]
                        positions = list(mapping[q] for q in reached_qubits)
                        mapping.update(zip(reached_qubits, reversed(positions)))

                        posterior_interstitial_layer.append(
                                post_acquaintance_gate)

                    if ((multiplicities[1] == self.acquaintance_size - 1) and
                        (multiplicities[0] > 1)):
                        # left part
                        post_qubits = parts_qubits[:self.acquaintance_size]
                        post_acquaintance_gate = ACQUAINT(*post_qubits)

                        max_reach = (multiplicities[1] // 2) - 1
                        post_layer.append(
                                self.acquaint_insides(post_acquaintance_gate,
                                    parts_qubits[multiplicities[1]:],
                                    max_reach, True))

                        reached_qubits = (
                                parts_qubits[multiplicities[1]:
                                    ][:max_reach + 1])
                        positions = list(mapping[q] for q in reached_qubits)
                        mapping.update(zip(reached_qubits, reversed(positions)))

                        posterior_interstitial_layer.append(
                                post_acquaintance_gate)

                parts[i] = parts_qubits[:multiplicities[1]]
                parts[i + 1] = parts_qubits[multiplicities[1]:]
            prior_interstitial_layer.sort(key=op_sort_key)
            yield prior_interstitial_layer
            yield pre_layer
            yield layer
            yield post_layer
        posterior_interstitial_layer.sort(key=op_sort_key)
        for op in posterior_interstitial_layer:
            yield op

        # finish reversal
        for part in reversed(parts):
            part_len = len(part)
            if part_len > 1:
                positions = [mapping[q] for q in part]
                offset = min(positions)
                permutation = {
                        i: (mapping[q] - offset) for i, q in enumerate(part)}
                reverse_permutation = {
                        i: part_len - p - 1 for i, p in permutation.items()}
                yield LinearPermutationGate(reverse_permutation,
                        self.swap_gate)(*part)

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


    def permutation(self, qubit_count: int) -> Dict[int, int]:
        if qubit_count < sum(self.part_lens):
            raise ValueError('qubit_count must be as large as the sum of the'
                             'part lens.')
        return {i: j for i, j in
                enumerate(reversed(range(sum(self.part_lens))))}
def op_acquaintance_size(op: Operation):
    if not isinstance(op, GateOperation):
        return 0
    if isinstance(op.gate, AcquaintanceOpportunityGate):
        return len(op.qubits)
    if isinstance(op.gate, SwapNetworkGate):
        if (op.gate.acquaintance_size - 1) in op.gate.part_lens:
            return op.gate.acquaintance_size
    return 0
