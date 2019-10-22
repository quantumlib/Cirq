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

from typing import (
    Callable,
    Optional,
    List,
    NamedTuple,
    Any,
    Iterable,
    Sequence,
    TYPE_CHECKING,
    Union,
    Dict,
)

from cirq import ops, value

if TYPE_CHECKING:
    import cirq


class Cell:
    """A gate, operation, display, operation modifier, etc from Quirk.

    Represents something that can go into a column in Quirk, and supports the
    operations ultimately necessary to transform a grid of these cells into a
    `cirq.Circuit`.
    """

    def with_input(self, letter: str,
                   register: Union[Sequence['cirq.Qid'], int]) -> 'Cell':
        """The same cell, but linked to an explicit input register or constant.

        If the cell doesn't need the input, it is returned unchanged.

        Args:
            letter: The input variable name ('a', 'b', or 'r').
            register: The list of qubits to use as the input, or else a
                classical constant to use as the input.

        Returns:
            The same cell, but with the specified input made explicit.
        """
        return self

    def controlled_by(self, qubit: 'cirq.Qid') -> 'Cell':
        """The same cell, but with an explicit control on its main operations.

        Cells with effects that do not need to be controlled are permitted to
        return themselves unmodified.

        Args:
            qubit: The control qubit.

        Returns:
            A modified cell with an additional control.
        """
        return self

    def operations(self) -> 'cirq.OP_TREE':
        """Returns operations that implement the cell's main action.

        Returns:
            A `cirq.OP_TREE` of operations implementing the cell.

        Raises:
            ValueError:
                The cell is not ready for conversion into operations, e.g. it
                may still have unspecified inputs.
        """
        return ()

    def basis_change(self) -> 'cirq.OP_TREE':
        """Operations to conjugate a column with.

        The main distinctions between operations performed during the body of a
        column and operations performed during the basis change are:

        1. Basis change operations are not affected by operation modifiers in
            the column. For example, adding a control into the same column will
            not affect the basis change.
        2. Basis change operations happen twice, once when starting a column and
            a second time (but inverted) when ending a column.

        Returns:
            A `cirq.OP_TREE` of basis change operations.
        """
        return ()

    def modify_column(self, column: List[Optional['Cell']]) -> None:
        """Applies this cell's modification to its column.

        For example, a control cell will add a control qubit to other operations
        in the column.

        Args:
            column: A mutable list of cells in the column, including empty
                cells (with value `None`). This method is permitted to change
                the items in the list, but must not change the length of the
                list.

        Returns:
            Nothing. The `column` argument is mutated in place.
        """

    def persistent_modifiers(self) -> Dict[str, Callable[['Cell'], 'Cell']]:
        """Overridable modifications to apply to the rest of the circuit.

        Persistent modifiers apply to all cells in the same column and also to
        all cells in future columns (until a column overrides the modifier with
        another one using the same key).

        Returns:
            A dictionary of keyed modifications. Each modifier lasts until a
            later cell specifies a new modifier with the same key.
        """
        return {}


@value.value_equality
class ExplicitOperationsCell(Cell):
    """A quirk cell with known body operations and basis change operations."""

    def __init__(self,
                 operations: Iterable[ops.Operation],
                 basis_change: Iterable[ops.Operation] = ()):
        self._operations = tuple(operations)
        self._basis_change = tuple(basis_change)

    def _value_equality_values_(self):
        return self._operations, self._basis_change

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change

    def operations(self) -> 'cirq.OP_TREE':
        return self._operations

    def controlled_by(self, qubit: 'cirq.Qid') -> 'ExplicitOperationsCell':
        return ExplicitOperationsCell(
            [op.controlled_by(qubit) for op in self._operations],
            self._basis_change)


CELL_SIZES = range(1, 17)

CellMakerArgs = NamedTuple('CellMakerArgs', [
    ('qubits', Sequence['cirq.Qid']),
    ('value', Any),
    ('row', int),
    ('col', int),
])

CellMaker = NamedTuple('CellMaker', [
    ('identifier', str),
    ('size', int),
    ('maker', Callable[[CellMakerArgs], Union[None, 'Cell', 'cirq.Operation']]),
])
CellMaker.__doc__ = """Turns Quirk identifiers into Cirq operations.

Attributes:
    identifier: A string that identifies the cell type, such as "X" or "QFT3".
    size: The height of the operation. The number of qubits it covers.
    maker: A function that takes a `cirq.contrib.quirk.cells.CellMakerArgs` and
        returns either a `cirq.Operation` or a `cirq.contrib.quirk.cells.Cell`.
        Returning a cell is more flexible, because cells can modify other cells
        in the same column before producing operations, whereas returning an
        operation is simple.
"""
