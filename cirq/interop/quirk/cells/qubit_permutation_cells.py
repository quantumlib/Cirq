# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING

from cirq import ops, value
from cirq.interop.quirk.cells.cell import (
    CELL_SIZES,
    CellMaker,
)

if TYPE_CHECKING:
    import cirq


@value.value_equality
class QuirkQubitPermutationGate(ops.QubitPermutationGate):
    """A qubit permutation gate specified by a permutation list."""

    def __init__(self, identifier: str, name: str, permutation: Sequence[int]):
        """
        Args:
            identifier: Quirk identifier string.
            name: Label to include in circuit diagram info.
            permutation: A shuffled sequence of integers from 0 to
                len(permutation) - 1. The entry at offset `i` is the result
                of permuting `i`.
        """
        self.identifier = identifier
        self.name = name
        super().__init__(permutation)

    def _value_equality_values_(self):
        return self.identifier, self.name, self.permutation

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> Tuple[str, ...]:
        return tuple(f'{self.name}[{i}>{self.permutation[i]}]'
                     for i in range(len(self.permutation)))

    def __repr__(self) -> str:
        return ('cirq.interop.quirk.QuirkQubitPermutationGate('
                f'identifier={repr(self.identifier)},'
                f'name={repr(self.name)},'
                f'permutation={repr(self.permutation)})')


def generate_all_qubit_permutation_cell_makers() -> Iterator[CellMaker]:
    yield from _permutation_family("<<", 'left_rotate', lambda _, x: x + 1)
    yield from _permutation_family(">>", 'right_rotate', lambda _, x: x - 1)
    yield from _permutation_family("rev", 'reverse', lambda _, x: ~x)
    yield from _permutation_family("weave", 'interleave', _interleave_bit)
    yield from _permutation_family("split", 'deinterleave', _deinterleave_bit)


def _permutation_family(identifier_prefix: str, name: str,
                        permute: Callable[[int, int], int]
                       ) -> Iterator[CellMaker]:
    for n in CELL_SIZES:
        permutation = tuple(permute(n, i) % n for i in range(n))
        yield _permutation(identifier_prefix + str(n), name, permutation)


def _permutation(
        identifier: str,
        name: str,
        permutation: Tuple[int, ...],
) -> CellMaker:
    return CellMaker(
        identifier,
        size=len(permutation),
        maker=lambda args: QuirkQubitPermutationGate(
            identifier=identifier, name=name, permutation=permutation).on(
                *args.qubits))


def _interleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    group = x // h
    stride = x % h
    return stride * 2 + group


def _deinterleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    stride = x // 2
    group = x % 2
    return stride + group * h
