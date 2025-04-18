# Copyright 2019 The Cirq Developers
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

from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING

from cirq.interop.quirk.cells.cell import Cell, CELL_SIZES, CellMaker

if TYPE_CHECKING:
    import cirq


class InputCell(Cell):
    """A modifier that provides a quantum input to gates in the same column."""

    def __init__(self, qubits: Iterable[cirq.Qid], letter: str):
        self.qubits = tuple(qubits)
        self.letter = letter

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List[cirq.Qid]) -> Cell:
        return InputCell(qubits=Cell._replace_qubits(self.qubits, qubits), letter=self.letter)

    def modify_column(self, column: List[Optional[Cell]]):
        for i in range(len(column)):
            cell = column[i]
            if cell is not None:
                column[i] = cell.with_input(self.letter, self.qubits)


class SetDefaultInputCell(Cell):
    """A persistent modifier that provides a fallback classical input."""

    def __init__(self, letter: str, value: int):
        self.letter = letter
        self.value = value

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List[cirq.Qid]) -> Cell:
        return self

    def persistent_modifiers(self):
        return {f'set_default_{self.letter}': lambda cell: cell.with_input(self.letter, self.value)}


def generate_all_input_cell_makers() -> Iterator[CellMaker]:
    # Quantum inputs.
    yield from _input_family("inputA", "a")
    yield from _input_family("inputB", "b")
    yield from _input_family("inputR", "r")
    yield from _input_family("revinputA", "a", rev=True)
    yield from _input_family("revinputB", "b", rev=True)

    # Classical inputs.
    yield CellMaker("setA", 2, lambda args: SetDefaultInputCell('a', args.value))
    yield CellMaker("setB", 2, lambda args: SetDefaultInputCell('b', args.value))
    yield CellMaker("setR", 2, lambda args: SetDefaultInputCell('r', args.value))


def _input_family(identifier_prefix: str, letter: str, rev: bool = False) -> Iterator[CellMaker]:
    for n in CELL_SIZES:
        yield CellMaker(
            identifier=identifier_prefix + str(n),
            size=n,
            maker=lambda args: InputCell(
                qubits=args.qubits[::-1] if rev else args.qubits, letter=letter
            ),
        )
