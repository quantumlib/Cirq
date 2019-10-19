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

from typing import List, TYPE_CHECKING, Callable

from cirq import circuits
from cirq.contrib.quirk.cells.cell import Cell

if TYPE_CHECKING:
    import cirq


class CompositeCell(Cell):
    """A cell made up of a grid of sub-cells.

    This is used for custom circuit gates.
    """

    def __init__(self, height: int, sub_cell_cols: List[List[Cell]]):
        self.height = height
        self._sub_cell_cols = sub_cell_cols

    def _transform_cells(self, func: Callable[[Cell], Cell]) -> 'CompositeCell':
        return CompositeCell(
            self.height,
            [[None if cell is None else func(cell)
              for cell in col]
             for col in self._sub_cell_cols])

    def with_qubits(self, qubits: List['cirq.Qid']) -> 'Cell':
        return self._transform_cells(lambda cell: cell.with_qubits(qubits))

    def with_input(self, letter, register):
        return self._transform_cells(lambda cell: cell.with_input(
            letter, register))

    def controlled_by(self, qubit):
        return self._transform_cells(lambda cell: cell.controlled_by(qubit))

    def circuit(self) -> 'cirq.Circuit':
        result = circuits.Circuit()
        for col in self._sub_cell_cols:
            basis_change = circuits.Circuit(
                cell.basis_change() for cell in col if cell is not None)
            body = circuits.Circuit(
                cell.operations() for cell in col if cell is not None)
            result += basis_change
            result += body
            result += basis_change**-1
        return result

    def operations(self) -> 'cirq.OP_TREE':
        return self.circuit()
