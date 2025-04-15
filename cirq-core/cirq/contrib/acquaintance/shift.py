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

import itertools
from typing import Any, Dict, Iterator, Sequence, Tuple, TYPE_CHECKING

from cirq import ops, value
from cirq.contrib.acquaintance.permutation import PermutationGate, SwapPermutationGate

if TYPE_CHECKING:
    import cirq


@value.value_equality
class CircularShiftGate(PermutationGate):
    """Performs a cyclical permutation of the qubits to the left by a specified amount."""

    def __init__(self, num_qubits: int, shift: int, swap_gate: cirq.Gate = ops.SWAP) -> None:
        """Construct a circular shift gate.

        Args:
            num_qubits: The number of qubits to shift.
            shift: The number of positions to circularly left shift the qubits.
            swap_gate: The gate to use when decomposing.
        """
        super(CircularShiftGate, self).__init__(num_qubits, swap_gate)
        self.shift = shift

    def __repr__(self) -> str:
        return (
            'cirq.contrib.acquaintance.CircularShiftGate('
            f'num_qubits={self.num_qubits()!r},'
            f'shift={self.shift!r}, swap_gate={self.swap_gate!r})'
        )

    def _value_equality_values_(self) -> Any:
        return self.shift, self.swap_gate, self.num_qubits()

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> Iterator[cirq.OP_TREE]:
        n = len(qubits)
        left_shift = self.shift % n
        right_shift = n - left_shift
        mins = itertools.chain(range(left_shift - 1, 0, -1), range(right_shift))
        maxs = itertools.chain(range(left_shift, n), range(n - 1, right_shift, -1))
        swap_gate = SwapPermutationGate(self.swap_gate)
        for i, j in zip(mins, maxs):
            for k in range(i, j, 2):
                yield swap_gate(*qubits[k : k + 2])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        if args.known_qubit_count is None:
            return NotImplemented  # pragma: no cover
        direction_symbols = ('╲', '╱') if args.use_unicode_characters else ('\\', '/')
        wire_symbols = tuple(
            direction_symbols[int(i >= self.shift)]
            + str(i)
            + direction_symbols[int(i < self.shift)]
            for i in range(self.num_qubits())
        )
        return wire_symbols

    def permutation(self) -> Dict[int, int]:
        shift = self.shift % self.num_qubits()
        permuted_indices = itertools.chain(range(shift, self.num_qubits()), range(shift))
        return {s: i for i, s in enumerate(permuted_indices)}
