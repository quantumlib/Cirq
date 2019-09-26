# Copyright 2018 The Cirq Developers
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

from typing import Callable, Iterable, Iterator

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import CELL_SIZES, CellMaker
from cirq.contrib.quirk.cells.explicit_operations_cell import ExplicitOperationsCell


class QuirkQubitPermutationOperation(ops.Operation):
    """A qubit permutation operation specified by a permute function."""

    def __init__(self, name: str, qubits: Iterable['cirq.Qid'],
                 permute: Callable[[int], int]):
        self.name = name
        self._qubits = tuple(qubits)
        self.permute = permute

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return QuirkQubitPermutationOperation(self.name, new_qubits, self.permute)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        # Compute the permutation index list.
        permuted_axes = list(range(len(args.target_tensor.shape)))
        for i in range(len(args.axes)):
            j = self.permute(i)
            ai = args.axes[i]
            aj = args.axes[j]
            assert args.target_tensor.shape[ai] == args.target_tensor.shape[aj]
            permuted_axes[aj] = ai

        # Delegate to numpy to do the permuted copy.
        args.available_buffer[...] = args.target_tensor.transpose(permuted_axes)
        return args.available_buffer

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        return tuple(f'{self.name}[{i}>{self.permute(i)}]'
                     for i in range(len(self._qubits)))

    def __repr__(self):
        return 'cirq.quirk.QubitPermutation({!r}, {!r})'.format(
            self._qubits, self.permute)


def all_qubit_permutation_cells():
    yield from qubit_permutation_cell_family(
        "<<", 'left_rotate', lambda n, x: (x + 1) % n)
    yield from qubit_permutation_cell_family(
        ">>", 'right_rotate', lambda n, x: (x - 1) % n)
    yield from qubit_permutation_cell_family("rev",
                                          'reverse', lambda n, x: n - x - 1)
    yield from qubit_permutation_cell_family("weave", 'interleave', interleave_bit)
    yield from qubit_permutation_cell_family("split", 'deinterleave',
                                          deinterleave_bit)


def qubit_permutation_cell_family(identifier_prefix: str, name: str,
                                  permutation: Callable[[int, int], int]
                                  ) -> Iterator[CellMaker]:
    f = lambda args: ExplicitOperationsCell([
        QuirkQubitPermutationOperation(name, args.qubits, lambda e: permutation(
            len(args.qubits), e))
    ])
    for i in CELL_SIZES:
        yield CellMaker(identifier_prefix + str(i), i, f)


def interleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    group = x // h
    stride = x % h
    return stride * 2 + group


def deinterleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    stride = x // 2
    group = x % 2
    return stride + group * h
