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

from typing import Callable, Optional, Union, Iterable, List, Sequence

import numpy as np

import cirq
from cirq import ops


class Cell:
    """A gate, operation, display, operation modifier, etc from Quirk.

    Represents something that can go into a column in Quirk, and supports the
    operations ultimately necessary to transform a grid of these cells into a
    `cirq.Circuit`.
    """

    def with_input(self, letter: str, register: Union[List[cirq.Qid], int]):
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

    def controlled_by(self, qubit: 'cirq.Qid'):
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
        return []

    def basis_change(self) -> 'cirq.OP_TREE':
        """Operations to perform before a column is executed.

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
        return []

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


class ArithmeticCell(ops.ArithmeticOperation, Cell):

    def __init__(self, identifier: str,
                 registers: Sequence[Union[Sequence['cirq.Qid'], int]],
                 operation: Callable):
        self.identifier = identifier
        self._registers = registers
        self._operation = operation

    def with_input(self, letter, register):
        return self.with_registers(
            *[r if r != letter else register for r in self._registers])

    def operations(self) -> 'cirq.OP_TREE':
        return self

    def registers(self):
        return self._registers

    def with_registers(self, *new_registers: Union[int, Sequence['cirq.Qid']]
                      ) -> 'cirq.ArithmeticOperation':
        return ArithmeticCell(self.identifier, new_registers, self._operation)

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        return self._operation(*registers)

    def __repr__(self):
        return 'cirq.quirk.ArithmeticCell({!r}, {!r} ,{!r})'.format(
            self.identifier, self.registers(), self._operation)


class OpsCell(Cell):
    """A quirk cell with known operations."""

    def __init__(self,
                 operations: Iterable[ops.Operation],
                 basis_change: Iterable[ops.Operation] = ()):
        self._operations = tuple(operations)
        self._basis_change = tuple(basis_change)

    def operations(self) -> 'cirq.OP_TREE':
        return self._operations

    def controlled_by(self, qubit: 'cirq.Qid'):
        return OpsCell([op.controlled_by(qubit) for op in self._operations],
                       self._basis_change)

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change


class QuirkPseudoSwapOperation(Cell):

    def __init__(self, qubits: Iterable['cirq.Qid'],
                 controls: Iterable['cirq.Qid']):
        self._qubits = list(qubits)
        self._controls = list(controls)

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not self and isinstance(gate, QuirkPseudoSwapOperation):
                assert self._controls == gate._controls
                self._qubits += gate._qubits
                column[i] = None

    def operations(self) -> 'cirq.OP_TREE':
        if len(self._qubits) != 2:
            raise ValueError('Wrong number of swap gates in a column.')
        return ops.SWAP(*self._qubits).controlled_by(*self._controls)

    def controlled_by(self, qubit: 'cirq.Qid'):
        return QuirkPseudoSwapOperation(self._qubits, self._controls + [qubit])


class DependentCell(Cell):
    """Applies an operation that depends on an input gate."""

    def __init__(self, register: Union[str, List['cirq.Qid']],
                 target: 'cirq.Qid',
                 op_maker: Callable[[int, int, Sequence['cirq.Qid']],
                                    'cirq.Operation']):
        self.register = register
        self.target = target
        self.op_maker = op_maker

    def with_input(self, letter: str, register: Union[List[cirq.Qid], int]):
        if self.register == letter:
            if isinstance(register, int):
                raise TypeError('Dependent operation requires known length '
                                'input; classical constant not allowed.')
            return DependentCell(register, self.target, self.op_maker)
        return self

    def controlled_by(self, qubit: 'cirq.Qid'):
        return DependentCell(
            self.register, self.target, lambda a, b, c: self.op_maker(a, b, c).
            controlled_by(qubit))

    def operations(self) -> 'cirq.OP_TREE':
        if isinstance(self.register, str):
            raise ValueError(f'Missing input {self.register}')
        return DependentOperation(self.register, self.op_maker, [self.target])


class DependentOperation(ops.Operation):
    """Operates on target qubits in a way that varies based on an input qureg.
    """

    def __init__(self, register: Iterable['cirq.Qid'],
                 op_maker: Callable[[int, int, Sequence['cirq.Qid']],
                                    'cirq.Operation'],
                 op_qubits: Iterable['cirq.Qid']):
        self.register = tuple(register)
        self.op_maker = op_maker
        self.op_qubits = tuple(op_qubits)

    def qubits(self):
        return self.register + self.op_qubits

    def with_qubits(self, *new_qubits):
        new_op_qubits = new_qubits[:len(self.op_qubits)]
        new_register = new_qubits[len(self.op_qubits):]
        return DependentOperation(new_register, self.op_maker, new_op_qubits)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        # Get input register value.
        transposed_args = args.with_axes_transposed_to_start()
        size = np.product(q.dimension for q in self.register)
        value = transposed_args.target_tensor.reshape(
            size, transposed_args.target_tensor // size)[0]

        # Apply dependent operation.
        operation = self.op_maker(value, size, self.op_qubits)
        return cirq.apply_unitary(
            operation,
            cirq.ApplyUnitaryArgs(args.target_tensor, args.available_buffer,
                                  args.axes[len(self.register):]))


class QubitPermutation(ops.Operation):
    """A qubit permutation operation specified by a permute function."""

    def __init__(self, qubits: Iterable['cirq.Qid'],
                 permute: Callable[[int], int]):
        self._qubits = tuple(qubits)
        self.permute = permute

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return QubitPermutation(new_qubits, self.permute)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        # Compute the permutation index list.
        permuted_axes = list(range(len(args.target_tensor.shape)))
        for i in range(len(args.axes)):
            j = self.permute(i)
            ai = args.axes[i]
            aj = args.axes[j]
            assert args.target_tensor.shape[ai] == args.target_tensor.shape[aj]
            permuted_axes[ai] = aj

        # Delegate to numpy to do the permuted copy.
        args.available_buffer[permuted_axes] = args.target_tensor
        return args.available_buffer

    def __repr__(self):
        return 'cirq.quirk.QubitPermutation({!r}, {!r})'.format(
            self._qubits, self.permute)
