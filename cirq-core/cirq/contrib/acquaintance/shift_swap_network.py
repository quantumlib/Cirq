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

from __future__ import annotations

import functools
import itertools
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate
from cirq.contrib.acquaintance.shift import CircularShiftGate

if TYPE_CHECKING:
    import cirq


class ShiftSwapNetworkGate(PermutationGate):
    """A swap network that generalizes the circular shift gate.

    Given a specification of two partitions, implements a swap network that has
    the overall effect of:
        * For every pair of parts, one from each partition, acquainting the
            union of the corresponding qubits.
        * Circularly shifting the two sets of qubits.

    Args:
        left_part_lens: The sizes of the parts in the partition of the first
            set of qubits.
        right_part_lens: The sizes of the parts in the partition of the second
            set of qubits.
        swap_gate: The gate to use when decomposing.

    Attributes:
        part_lens: A mapping from the side (as a str, 'left' or 'right') to the
            part sizes of the corresponding partition.
        swap_gate: The gate to use when decomposing.
    """

    def __init__(
        self,
        left_part_lens: Iterable[int],
        right_part_lens: Iterable[int],
        swap_gate: cirq.Gate = ops.SWAP,
    ) -> None:

        self.part_lens = {'left': tuple(left_part_lens), 'right': tuple(right_part_lens)}

        for part_lens in self.part_lens.values():
            if min(part_lens) < 1:
                raise ValueError('not min(part_lens)')

        self.swap_gate = swap_gate

    def acquaintance_size(self) -> int:
        return sum(max(self.part_lens[side]) for side in ('left', 'right'))

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> Iterator[cirq.OP_TREE]:
        part_lens = list(itertools.chain(*(self.part_lens[side] for side in ('left', 'right'))))

        n_qubits = 0
        parts = []
        for part_len in part_lens:
            parts.append(list(qubits[n_qubits : n_qubits + part_len]))
            n_qubits += part_len

        n_parts = len(part_lens)
        n_left_parts = len(self.part_lens['left'])
        n_right_parts = n_parts - n_left_parts

        mins = itertools.chain(range(n_left_parts - 1, 0, -1), range(n_right_parts))
        maxs = itertools.chain(range(n_left_parts, n_parts), range(n_parts - 1, n_right_parts, -1))
        SHIFT = functools.partial(CircularShiftGate, swap_gate=self.swap_gate)

        for i, j in zip(mins, maxs):
            for k in range(i, j, 2):
                left_part, right_part = parts[k : k + 2]
                parts_qubits = left_part + right_part
                yield acquaint(*parts_qubits)
                yield SHIFT(len(parts_qubits), len(left_part))(*parts_qubits)
                parts[k] = parts_qubits[: len(right_part)]
                parts[k + 1] = parts_qubits[len(right_part) :]

    def qubit_count(self, side: Optional[str] = None) -> int:
        if side is None:
            return sum(self.qubit_count(side) for side in self.part_lens)
        return sum(self.part_lens[side])

    def num_qubits(self) -> int:
        return self.qubit_count()

    def permutation(self) -> Dict[int, int]:
        return dict(
            zip(
                range(self.num_qubits()),
                itertools.chain(
                    range(self.qubit_count('right'), self.num_qubits()),
                    range(self.qubit_count('right')),
                ),
            )
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        qubit_count = self.qubit_count()
        assert args.known_qubit_count in (None, qubit_count)

        arrow = '↦' if args.use_unicode_characters else '->'

        wire_symbols = []
        for i, side in enumerate(('left', 'right')):
            for j, part_len in enumerate(self.part_lens[side]):
                for k in range(part_len):
                    wire_symbols.append(str((i, j, k)) + arrow + str((int(not (i)), j, k)))
        return tuple(wire_symbols)

    def __repr__(self) -> str:
        args = tuple(repr(self.part_lens[side]) for side in ('left', 'right'))
        if self.swap_gate != ops.SWAP:
            args += (repr(self.swap_gate),)
        args_str = ', '.join(args)
        return f'cirq.contrib.acquaintance.shift_swap_network.ShiftSwapNetworkGate({args_str})'

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.part_lens == other.part_lens
            and self.swap_gate == other.swap_gate
        )
