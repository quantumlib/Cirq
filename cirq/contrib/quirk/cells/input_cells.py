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

from typing import Optional, List, Iterator

import cirq
from cirq.contrib.quirk.cells.cell import Cell, CELL_SIZES, CellMaker


class InputCell(Cell):
    """A modifier that provides a quantum input to gates in the same column."""

    def __init__(self, qubits: List[cirq.Qid], letter: str):
        self.qubits = qubits
        self.letter = letter

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not None:
                column[i] = gate.with_input(self.letter, self.qubits)


class SetDefaultInputCell(Cell):
    """A persistent modifier that provides a fallback classical input."""

    def __init__(self, letter: str, value: int):
        self.letter = letter
        self.value = value

    def persistent_modifiers(self):
        return {
            f'set_default_{self.letter}':
            lambda cell: cell.with_input(self.letter, self.value)
        }


def generate_all_input_cells():
    # Quantum inputs.
    yield from reg_input_family("inputA", "a")
    yield from reg_input_family("inputB", "b")
    yield from reg_input_family("inputR", "r")
    yield from reg_input_family("revinputA", "a", rev=True)
    yield from reg_input_family("revinputB", "b", rev=True)

    # Classical inputs.
    yield CellMaker("setA",
                    2, lambda args: SetDefaultInputCell('a', args.value))
    yield CellMaker("setB",
                    2, lambda args: SetDefaultInputCell('b', args.value))
    yield CellMaker("setR",
                    2, lambda args: SetDefaultInputCell('r', args.value))


def reg_input_family(identifier_prefix: str, letter: str,
                     rev: bool = False) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield CellMaker(
            identifier_prefix + str(i), i, lambda args: InputCell(
                args.qubits[::-1] if rev else args.qubits, letter))
