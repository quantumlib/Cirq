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

from typing import Optional, List, Iterable, Iterator

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import Cell, CellMaker


class ControlCell(Cell):
    """A modifier that adds controls to other cells in the column."""

    def __init__(self, qubit: cirq.Qid, basis_change: List[cirq.Operation]):
        self.qubit = qubit
        self._basis_change = basis_change

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not None:
                column[i] = gate.controlled_by(self.qubit)

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change


class ParityControlCell(Cell):
    """A modifier that adds a group parity control to other cells in the column.

    The parity controls in a column are satisfied *as a group* if an odd number
    of them are individually satisfied.
    """

    def __init__(self, qubits: Iterable['cirq.Qid'],
                 basis_change: Iterable['cirq.Operation']):
        self.qubits = list(qubits)
        self._basis_change = list(basis_change)

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is self:
                continue
            elif isinstance(gate, ParityControlCell):
                # The first parity control to modify the column must merge all
                # of the other parity controls into itself.
                column[i] = None
                self._basis_change += gate._basis_change
                self.qubits += gate.qubits
            elif gate is not None:
                column[i] = gate.controlled_by(self.qubits[0])

    def basis_change(self) -> 'cirq.OP_TREE':
        yield from self._basis_change

        # Temporarily move the ZZZ..Z parity observable onto a single qubit.
        for q in self.qubits[1:]:
            yield ops.CNOT(q, self.qubits[0])


def generate_all_control_cell_makers():
    # Controls.
    yield _reg_control("•", basis_change=None)
    yield _reg_control("◦", basis_change=ops.X)
    yield _reg_control("⊕", basis_change=ops.Y**0.5)
    yield _reg_control("⊖", basis_change=ops.Y**-0.5)
    yield _reg_control("⊗", basis_change=ops.X**-0.5)
    yield _reg_control("(/)", basis_change=ops.X**0.5)

    # Parity controls.
    yield _reg_parity_control("xpar", basis_change=ops.Y**0.5)
    yield _reg_parity_control("ypar", basis_change=ops.X**-0.5)
    yield _reg_parity_control("zpar", basis_change=None)


def _reg_control(identifier: str, *,
                 basis_change: Optional['cirq.Gate']) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        func=lambda args: ControlCell(
            args.qubits[0],
            basis_change.on(args.qubits[0]) if basis_change else []))


def _reg_parity_control(identifier: str,
                        *,
                        basis_change: Optional['cirq.Gate'] = None
                       ) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        func=lambda args: ParityControlCell(args.qubits, (
        ) if basis_change is None else basis_change.on_each(args.qubits)))
