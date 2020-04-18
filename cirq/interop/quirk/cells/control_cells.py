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

from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union

from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker

if TYPE_CHECKING:
    import cirq


@value.value_equality
class ControlCell(Cell):
    """A modifier that adds controls to other cells in the column."""

    def __init__(self, qubit: 'cirq.Qid',
                 basis_change: Iterable['cirq.Operation']):
        self.qubit = qubit
        self._basis_change = tuple(basis_change)

    def _value_equality_values_(self) -> Any:
        return self.qubit, self._basis_change

    def __repr__(self) -> str:
        return (f'cirq.interop.quirk.cells.control_cells.ControlCell('
                f'\n    {self.qubit!r},'
                f'\n    {self._basis_change!r})')

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ControlCell(
            qubit=Cell._replace_qubit(self.qubit, qubits),
            basis_change=tuple(
                op.with_qubits(*Cell._replace_qubits(op.qubits, qubits))
                for op in self._basis_change))

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not None:
                column[i] = gate.controlled_by(self.qubit)

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change


@value.value_equality(unhashable=True)
class ParityControlCell(Cell):
    """A modifier that adds a group parity control to other cells in the column.

    The parity controls in a column are satisfied *as a group* if an odd number
    of them are individually satisfied.
    """

    def __init__(self, qubits: Iterable['cirq.Qid'],
                 basis_change: Iterable['cirq.Operation']):
        self.qubits = list(qubits)
        self._basis_change = list(basis_change)

    def _value_equality_values_(self) -> Any:
        return self.qubits, self._basis_change

    def __repr__(self) -> str:
        return (f'cirq.interop.quirk.cells.control_cells.ParityControlCell('
                f'\n    {self.qubits!r},'
                f'\n    {self._basis_change!r})')

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ParityControlCell(
            qubits=Cell._replace_qubits(self.qubits, qubits),
            basis_change=tuple(
                op.with_qubits(*Cell._replace_qubits(op.qubits, qubits))
                for op in self._basis_change))

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


def generate_all_control_cell_makers() -> Iterator[CellMaker]:
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
                 basis_change: Optional['cirq.SingleQubitGate']) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        maker=lambda args: ControlCell(qubit=args.qubits[0],
                                       basis_change=_basis_else_empty(
                                           basis_change, args.qubits[0])))


def _reg_parity_control(identifier: str,
                        *,
                        basis_change: Optional['cirq.SingleQubitGate'] = None
                       ) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        maker=lambda args: ParityControlCell(qubits=args.qubits,
                                             basis_change=_basis_else_empty(
                                                 basis_change, args.qubits)))


def _basis_else_empty(basis_change: Optional['cirq.SingleQubitGate'],
                      qureg: Union['cirq.Qid', Iterable['cirq.Qid']]
                     ) -> Iterable['cirq.Operation']:
    if basis_change is None:
        return ()
    return basis_change.on_each(qureg)
