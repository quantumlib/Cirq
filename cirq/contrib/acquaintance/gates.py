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
from operator import indexOf
from typing import Iterable, Sequence, TYPE_CHECKING

from cirq import CompositeGate, TextDiagrammable
from cirq.ops import (
        Operation, Gate, gate_features, QubitId, OP_TREE, SWAP)
from cirq.contrib.acquaintance.shift import CircularShiftGate

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
                 acquaintance_size: int=0,
                 swap_gate: Gate=SWAP
                 ) -> None:
        self.part_lens = tuple(part_lens)
        self.acquaintance_size = acquaintance_size
        self.swap_gate = swap_gate

    def default_decompose(self, qubits: Sequence[QubitId]) -> OP_TREE:
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
        posterior_interstitial_layer = [] # type: List[Operation]
        for layer_num in range(n_parts):
            prior_interstitial_layer = posterior_interstitial_layer
            layer = []
            posterior_interstitial_layer = []
            for i in range(layer_num % 2, n_parts - 1, 2):
                left_part, right_part = parts[i:i+2]
                parts_qubits = left_part + right_part
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
                    if self.acquaintance_size > 3:
                        raise NotImplementedError()
                    layer.append(shift)
                    if multiplicities[0] == self.acquaintance_size - 1:
                        pre_qubits = parts_qubits[:self.acquaintance_size]
                        prior_interstitial_layer.append(ACQUAINT(*pre_qubits))
                        post_qubits = parts_qubits[-self.acquaintance_size:]
                        posterior_interstitial_layer.append(
                                ACQUAINT(*post_qubits))
                    if multiplicities[1] == self.acquaintance_size - 1:
                        pre_qubits = parts_qubits[-self.acquaintance_size:]
                        prior_interstitial_layer.append(ACQUAINT(*pre_qubits))
                        post_qubits = parts_qubits[:self.acquaintance_size]
                        posterior_interstitial_layer.append(
                                ACQUAINT(*post_qubits))

                parts[i] = parts_qubits[:multiplicities[1]]
                parts[i + 1] = parts_qubits[multiplicities[1]:]
            prior_interstitial_layer.sort(key=op_sort_key)
            yield prior_interstitial_layer
            yield layer
        posterior_interstitial_layer.sort(key=op_sort_key)
        for op in posterior_interstitial_layer:
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
