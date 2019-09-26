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

from typing import Callable, Optional, Union, Iterable, List, Sequence, Dict

import numpy as np

import cirq
from cirq import ops, linalg


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

    def persistent_modifiers(self) -> Dict[str, Callable[['Cell'], 'Cell']]:
        """Overridable modifications to apply to the rest of the circuit.

        Persistent modifiers apply to all cells in the same column, not just to
        future columns.

        Returns:
            A dictionary of keyed modifications. Each modifier lasts until a
            later cell specifies a new modifier with the same key.
        """
        return {}


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


class ArithmeticCell(ops.ArithmeticOperation, Cell):

    def __init__(self,
                 identifier: str,
                 registers: Sequence[
                     Optional[Union[Sequence['cirq.Qid'], int]]],
                 register_letters: Sequence[Optional[str]],
                 operation: Callable,
                 is_modular: bool):
        if is_modular:
            f = operation
            operation = (lambda *args:
                f(*args) % args[-1] if args[0] < args[-1] else args[0])

        self.identifier = identifier
        self._registers = registers
        self._register_letters = register_letters
        self._operation = operation
        self._is_modular = is_modular

    def with_input(self, letter, register):
        return self.with_registers(*[
            reg if letter != reg_letter else register
            for reg, reg_letter in zip(self._registers,
                                       self._register_letters)
        ])

    def operations(self) -> 'cirq.OP_TREE':
        if self._is_modular:
            assert self._register_letters.index(None) == 0
            assert self._register_letters.index('r') == len(
                self._register_letters) - 1
            x = self._registers[0]
            r = self._registers[-1]
            if r > 1 << len(x) if isinstance(r, int) else len(r) > len(x):
                raise ValueError('Target too small for modulus.\n'
                                 f'Target: {x}\n'
                                 f'Modulus: {r}')

        missing_inputs = [
            letter
            for reg, letter in zip(self._registers, self._register_letters)
            if reg is None
        ]
        if missing_inputs:
            raise ValueError(f'Missing input: {sorted(missing_inputs)}')
        return self

    def registers(self):
        return self._registers

    def with_registers(self, *new_registers: Union[int, Sequence['cirq.Qid']]
                      ) -> 'cirq.ArithmeticOperation':
        return ArithmeticCell(self.identifier,
                              new_registers,
                              self._register_letters,
                              self._operation,
                              is_modular=self._is_modular)

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        return self._operation(*registers)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        consts = ''.join(
            f',{letter}={reg}'
            for reg, letter in zip(self._registers, self._register_letters) \
            if isinstance(reg, int))
        result = []
        for reg, letter in zip(self._registers, self._register_letters):
            if not isinstance(reg, int):
                for i, q in enumerate(reg):
                    if letter is None:
                        if i:
                            label = f'#{i+1}'
                        else:
                            label = f'Quirk({self.identifier}{consts})'
                    else:
                        label = letter.upper() + str(i)
                    result.append(label)
        return tuple(result)

    def __str__(self):
        return f'QuirkArithmetic({self.identifier})'

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

    def __init__(self,
                 identifier: str,
                 register: Optional[List['cirq.Qid']],
                 register_letter: str,
                 target: 'cirq.Qid',
                 op_maker: Callable[[int, int, Sequence['cirq.Qid']],
                                    'cirq.Operation']):
        self.identifier = identifier
        self.register = register
        self.register_letter = register_letter
        self.target = target
        self.op_maker = op_maker

    def with_input(self, letter: str, register: Union[List[cirq.Qid], int]):
        if self.register is None and self.register_letter == letter:
            if isinstance(register, int):
                raise ValueError('Dependent operation requires known length '
                                 'input; classical constant not allowed.')
            return DependentCell(self.identifier, register, self.register_letter, self.target, self.op_maker)
        return self

    def controlled_by(self, qubit: 'cirq.Qid'):
        return DependentCell(
            self.identifier,
            self.register, self.register_letter, self.target, lambda a, b, c: self.op_maker(a, b, c).
            controlled_by(qubit))

    def operations(self) -> 'cirq.OP_TREE':
        if self.register is None:
            raise ValueError(f'Missing input {repr(self.register_letter)}')
        return DependentOperation(self.identifier,
                                  self.register,
                                  self.register_letter,
                                  self.op_maker,
                                  [self.target])


class DependentOperation(ops.Operation):
    """Operates on target qubits in a way that varies based on an input qureg.
    """

    def __init__(self,
                 identifier: str,
                 register: Iterable['cirq.Qid'],
                 register_letter: str,
                 op_maker: Callable[[int, int, Sequence['cirq.Qid']],
                                    'cirq.Operation'],
                 op_qubits: Iterable['cirq.Qid']):
        self.identifier = identifier
        self.register = tuple(register)
        self.register_letter = register_letter
        self.op_maker = op_maker
        self.op_qubits = tuple(op_qubits)

    @property
    def qubits(self):
        return self.op_qubits + self.register

    def with_qubits(self, *new_qubits):
        new_op_qubits = new_qubits[:len(self.op_qubits)]
        new_register = new_qubits[len(self.op_qubits):]
        return DependentOperation(self.identifier, new_register, self.register_letter, self.op_maker, new_op_qubits)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        result = [self.identifier.replace('n', str(len(self.register)))]
        result.extend(f'#{i+1}' for i in range(1, len(self.op_qubits)))
        result.extend(self.register_letter.upper() + str(i)
                      for i in range(len(self.register)))
        return tuple(result)

    def _has_unitary_(self):
        return True

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        transposed_args = args.with_axes_transposed_to_start()

        target_axes = transposed_args.axes[:len(self.op_qubits)]
        control_axes = transposed_args.axes[len(self.op_qubits):]
        control_max = np.product([q.dimension for q in self.register]).item()

        for i in range(control_max):
            operation = self.op_maker(i,
                                      control_max,
                                      self.op_qubits)
            control_index = linalg.slice_for_qubits_equal_to(
                control_axes, big_endian_qureg_value=i)
            sub_args = cirq.ApplyUnitaryArgs(
                transposed_args.target_tensor[control_index],
                transposed_args.available_buffer[control_index],
                target_axes)
            sub_result = cirq.apply_unitary(operation, sub_args)

            if sub_result is not sub_args.target_tensor:
                sub_args.target_tensor[...] = sub_result

        return args.target_tensor


class QubitPermutation(ops.Operation):
    """A qubit permutation operation specified by a permute function."""

    def __init__(self,
                 name: str,
                 qubits: Iterable['cirq.Qid'],
                 permute: Callable[[int], int]):
        self.name = name
        self._qubits = tuple(qubits)
        self.permute = permute

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return QubitPermutation(self.name, new_qubits, self.permute)

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
