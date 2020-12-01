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

import enum
import itertools
from typing import Dict, Sequence, Tuple, Union, TYPE_CHECKING

from cirq import ops

from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate, SwapPermutationGate

if TYPE_CHECKING:
    import cirq


@enum.unique
class BipartiteGraphType(enum.Enum):
    MATCHING = 1
    COMPLETE = 2

    def __repr__(self) -> str:
        return 'cirq.contrib.acquaintance.bipartite.BipartiteGraphType.' + self.name


class BipartiteSwapNetworkGate(PermutationGate):
    """A swap network that acquaints qubits in one half with qubits in the
    other.


    Acts on 2k qubits, acquainting some of the first k qubits with some of the
    latter k. May have the effect permuting the qubits within each half.

    Possible subgraphs include:
        MATCHING: acquaints qubit 1 with qubit (2k - 1), qubit 2 with qubit
            (2k- 2), and so on through qubit k with qubit k + 1.
        COMPLETE: acquaints each of qubits 1 through k with each of qubits k +
            1 through 2k.

    Args:
        part_size: The number of qubits in each half.
        subgraph: The bipartite subgraph of pairs of qubits to acquaint.
        swap_gate: The gate used to swap logical indices.

    Attributes:
        part_size: See above.
        subgraph: See above.
        swap_gate: See above.
    """

    def __init__(
        self,
        subgraph: Union[str, BipartiteGraphType],
        part_size: int,
        swap_gate: 'cirq.Gate' = ops.SWAP,
    ) -> None:
        super().__init__(2 * part_size, swap_gate)
        self.part_size = part_size
        self.subgraph = (
            subgraph if isinstance(subgraph, BipartiteGraphType) else BipartiteGraphType[subgraph]
        )
        self.swap_gate = swap_gate

    def decompose_complete(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        swap_gate = SwapPermutationGate(self.swap_gate)
        if self.part_size == 1:
            yield acquaint(*qubits)
            return
        for k in range(-self.part_size + 1, self.part_size - 1):
            for x in range(abs(k), 2 * self.part_size - abs(k), 2):
                yield acquaint(*qubits[x : x + 2])
                yield swap_gate(*qubits[x : x + 2])
        yield acquaint(qubits[self.part_size - 1], qubits[self.part_size])
        for k in reversed(range(-self.part_size + 1, self.part_size - 1)):
            for x in range(abs(k), 2 * self.part_size - abs(k), 2):
                yield acquaint(*qubits[x : x + 2])
                yield swap_gate(*qubits[x : x + 2])

    def decompose_matching(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        swap_gate = SwapPermutationGate(self.swap_gate)
        for k in range(-self.part_size + 1, self.part_size):
            for x in range(abs(k), 2 * self.part_size - abs(k), 2):
                if (x + 1) % self.part_size:
                    yield swap_gate(*qubits[x : x + 2])
                else:
                    yield acquaint(*qubits[x : x + 2])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        if len(qubits) != 2 * self.part_size:
            raise ValueError('len(qubits) != 2 * self.part_size')
        if self.subgraph == BipartiteGraphType.COMPLETE:
            return self.decompose_complete(qubits)
        if self.subgraph == BipartiteGraphType.MATCHING:
            return self.decompose_matching(qubits)
        raise NotImplementedError('No decomposition implemented for ' + str(self.subgraph))

    def permutation(self) -> Dict[int, int]:
        if self.num_qubits() != 2 * self.part_size:
            raise ValueError('qubit_count != 2 * self.part_size')
        if self.subgraph == BipartiteGraphType.MATCHING:
            return dict(
                enumerate(
                    itertools.chain(
                        *(
                            range(self.part_size + offset - 1, offset - 1, -1)
                            for offset in (0, self.part_size)
                        )
                    )
                )
            )
        if self.subgraph == BipartiteGraphType.COMPLETE:
            return dict(enumerate(range(2 * self.part_size)))
        raise NotImplementedError(str(self.subgraph) + 'not implemented')

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        qubit_count = 2 * self.part_size
        if args.known_qubit_count not in (None, qubit_count):
            raise ValueError('args.known_qubit_count not in (None, 2 * self.part_size)')
        partial_permutation = self.permutation()
        permutation = {i: partial_permutation.get(i, i) for i in range(qubit_count)}

        if self.subgraph == BipartiteGraphType.MATCHING:
            name = 'Matching'
        elif self.subgraph == BipartiteGraphType.COMPLETE:
            name = 'K_{{{0}, {0}}}'.format(self.part_size)
        # NB: self.subgraph not in BipartiteGraphType caught by self.permutation
        arrow = '↦' if args.use_unicode_characters else '->'

        wire_symbols = tuple(
            name
            + ':'
            + str((i // self.part_size, i % self.part_size))
            + arrow
            + str((j // self.part_size, j % self.part_size))
            for i, j in permutation.items()
        )
        return wire_symbols

    def __repr__(self) -> str:
        args: Tuple[str, ...] = (repr(self.subgraph), repr(self.part_size))
        if self.swap_gate != ops.SWAP:
            args += (repr(self.swap_gate),)
        args_str = ', '.join(args)
        return f'cirq.contrib.acquaintance.bipartite.BipartiteSwapNetworkGate({args_str})'

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.subgraph == other.subgraph
            and self.part_size == other.part_size
            and self.swap_gate == other.swap_gate
        )
