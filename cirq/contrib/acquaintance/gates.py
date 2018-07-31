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
from typing import Sequence, Dict, Tuple, List

from cirq import CompositeGate, TextDiagrammable
from cirq.ops import (
        Operation, Gate, gate_features, QubitId, OP_TREE, SWAP, GateOperation)
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
        PermutationGate, SwapPermutationGate, LinearPermutationGate)

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

def acquaint_insides(swap_gate: Gate,
                     acquaintance_gate: Operation,
                     qubits: Sequence[QubitId],
                     before: bool,
                     layers: Dict[str, List[Operation]],
                     mapping: Dict[QubitId, int]
                     ) -> None:
    """Acquaints half a list of qubits.

    Args:
        qubits: The list of qubits of which half are individually acquainted
            with another list of qubits.
        layers: The layers to put gates into.
        acquaintance_gate: The acquaintance gate that acquaints the end qubit
            with another list of qubits.
        before: Whether the acquainting is done before the shift.
        swap_gate: The gate used to swap logical indices.
        mapping: The mapping from qubits to logical indices. Used to keep track
            of the effect of inside-acquainting swaps.
    """

    max_reach = _get_max_reach(len(qubits), round_up=before)
    reaches = chain(range(1, max_reach + 1),
                    range(max_reach, -1, -1))
    offsets = (0, 1) * max_reach
    swap_gate = SwapPermutationGate(swap_gate)
    ops = []
    for offset, reach in zip(offsets, reaches):
        if offset == before:
            ops.append(acquaintance_gate)
        for dr in range(offset, reach, 2):
            ops.append(swap_gate(*qubits[dr:dr + 2]))
    layers['pre' if before else 'post'] += ops

    # add interstitial gate
    interstitial = ('prior' if before else 'posterior') + '_interstitial'
    layers[interstitial].append(acquaintance_gate)

    # update mapping
    reached_qubits = qubits[:max_reach + 1]
    positions = list(mapping[q] for q in reached_qubits)
    mapping.update(zip(reached_qubits, reversed(positions)))

def _get_max_reach(size: int, round_up: bool=True) -> int:
    if round_up:
        return int(ceil(size / 2)) - 1
    return max((size // 2) - 1, 0)


def acquaint_and_shift(parts: Tuple[List[QubitId], List[QubitId]],
                       layers: Dict[str, List[Operation]],
                       acquaintance_size: int,
                       swap_gate: Gate,
                       mapping: Dict[QubitId, int]):
    """Acquaints and shifts a pair of lists of qubits. The first part is
    acquainted with every qubit individually in the second part, and vice
    versa. Operations are grouped into several layers:
        * prior_interstitial: The first layer of acquaintance gates.
        * prior: The combination of acquaintance gates and swaps that acquaints
            the inner halves.
        * intra: The shift gate.
        * post: The combination of acquaintance gates and swaps that acquaints
            the outer halves.
        * posterior_interstitial: The last layer of acquaintance gates.

    Args:
        parts: The two lists of qubits to acquaint.
        layers: The layers to put gates into.
        acquaintance_size: The number of qubits to acquaint at a time.
        swap_gate: The gate used to swap logical indices.
        mapping: The mapping from qubits to logical indices. Used to keep track
            of the effect of inside-acquainting swaps.
    """
    left_part, right_part = parts
    left_size, right_size = len(left_part), len(right_part)
    assert not (set(left_part) & set(right_part))
    qubits = left_part + right_part
    shift = CircularShiftGate(left_size,
                              swap_gate=swap_gate)(
                                      *qubits)
    if max(left_size, right_size) != acquaintance_size - 1:
        layers['intra'].append(shift)
    elif acquaintance_size == 2:
        layers['prior_interstitial'].append(ACQUAINT(*qubits))
        layers['intra'].append(shift)
    else:
        # before
        if left_size == acquaintance_size - 1:
            # right part
            pre_acquaintance_gate = ACQUAINT(*qubits[:acquaintance_size])
            acquaint_insides(
                    swap_gate=swap_gate,
                    acquaintance_gate=pre_acquaintance_gate,
                    qubits=right_part,
                    before=True,
                    layers=layers,
                    mapping=mapping)

        if right_size == acquaintance_size - 1:
            # left part
            pre_acquaintance_gate = ACQUAINT(*qubits[-acquaintance_size:])
            acquaint_insides(
                    swap_gate=swap_gate,
                    acquaintance_gate=pre_acquaintance_gate,
                    qubits=left_part[::-1],
                    before=True,
                    layers=layers,
                    mapping=mapping)

        layers['intra'].append(shift)
        shift.gate.update_mapping(mapping, qubits)

        # after
        if ((left_size == acquaintance_size - 1) and
            (right_size > 1)):
            # right part
            post_acquaintance_gate = ACQUAINT(*qubits[-acquaintance_size:])

            new_left_part = qubits[right_size - 1::-1]
            acquaint_insides(
                    swap_gate=swap_gate,
                    acquaintance_gate=post_acquaintance_gate,
                    qubits=new_left_part,
                    before=False,
                    layers=layers,
                    mapping=mapping)

        if ((right_size == acquaintance_size - 1) and
            (left_size > 1)):
            # left part

            post_acquaintance_gate = ACQUAINT(*qubits[:acquaintance_size])
            acquaint_insides(
                    swap_gate=swap_gate,
                    acquaintance_gate=post_acquaintance_gate,
                    qubits=qubits[right_size:],
                    before=False,
                    layers=layers,
                    mapping=mapping)


class SwapNetworkGate(CompositeGate, PermutationGate):
    """A single gate representing a generalized swap network.

    Args:
        part_lens: An sequence indicating the sizes of the parts in the
            partition defining the swap network.
        acquaintance_size: An int indicating the locality of the logical gates
            desired; used to keep track of this while nesting. If 0, no
            acquaintance gates are inserted.

    Attributes:
        part_lens: See above
        acquaintance_size: See above.
        swap_gate: The gate used to swap logical indices.
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
        layers = {'posterior_interstitial': []
                  } # type: Dict[str, List[Operation]]
        for layer_num in range(n_parts):
            layers['prior_interstitial'] = layers['posterior_interstitial']
            for l in ('pre', 'intra', 'post', 'posterior_interstitial'):
                layers[l] = []
            for i in range(layer_num % 2, n_parts - 1, 2):
                left_part, right_part = parts[i:i+2]
                acquaint_and_shift(parts=(left_part, right_part),
                                   layers=layers,
                                   acquaintance_size=self.acquaintance_size,
                                   swap_gate=self.swap_gate,
                                   mapping=mapping)

                parts_qubits = list(left_part + right_part)
                parts[i] = parts_qubits[:len(right_part)]
                parts[i + 1] = parts_qubits[len(right_part):]
            layers['prior_interstitial'].sort(key=op_sort_key)
            for l in ('prior_interstitial', 'pre', 'intra', 'post'):
                yield layers[l]
        layers['posterior_interstitial'].sort(key=op_sort_key)
        yield layers['posterior_interstitial']

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
        assert set(layers.keys()).issubset((
            'prior_interstitial', 'pre', 'intra',
            'post', 'posterior_interstitial'))

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
