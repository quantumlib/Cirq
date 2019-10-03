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

from typing import Callable, Optional, Union, Iterable, List, Sequence, Iterator

import numpy as np

import cirq
from cirq import ops, linalg
from cirq.contrib.quirk.cells.cell import Cell, CellMaker


class InputRotationCell(Cell):
    """Applies an operation that depends on an input gate."""

    def __init__(self, identifier: str,
                 register: Optional[Sequence['cirq.Qid']], register_letter: str,
                 target: 'cirq.Qid',
                 op_maker: Callable[[int, int, Sequence['cirq.Qid']],
                                    'cirq.Operation']):
        self.identifier = identifier
        self.register = None if register is None else tuple(register)
        self.register_letter = register_letter
        self.target = target
        self.op_maker = op_maker

    def with_input(self, letter, register):
        if self.register is None and self.register_letter == letter:
            if isinstance(register, int):
                raise ValueError('Dependent operation requires known length '
                                 'input; classical constant not allowed.')
            return InputRotationCell(self.identifier, register,
                                     self.register_letter, self.target,
                                     self.op_maker)
        return self

    def controlled_by(self, qubit: 'cirq.Qid'):
        return InputRotationCell(
            self.identifier, self.register,
            self.register_letter, self.target, lambda a, b, c: self.op_maker(
                a, b, c).controlled_by(qubit))

    def operations(self) -> 'cirq.OP_TREE':
        if self.register is None:
            raise ValueError(f'Missing input {repr(self.register_letter)}')
        return QuirkInputRotationOperation(self.identifier, self.register,
                                           self.register_letter, self.op_maker,
                                           [self.target])


class QuirkInputRotationOperation(ops.Operation):
    """Operates on target qubits in a way that varies based on an input qureg.
    """

    def __init__(self, identifier: str, register: Iterable['cirq.Qid'],
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
        return QuirkInputRotationOperation(self.identifier, new_register,
                                           self.register_letter, self.op_maker,
                                           new_op_qubits)

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
            operation = self.op_maker(i, control_max, self.op_qubits)
            control_index = linalg.slice_for_qubits_equal_to(
                control_axes, big_endian_qureg_value=i)
            sub_args = cirq.ApplyUnitaryArgs(
                transposed_args.target_tensor[control_index],
                transposed_args.available_buffer[control_index], target_axes)
            sub_result = cirq.apply_unitary(operation, sub_args)

            if sub_result is not sub_args.target_tensor:
                sub_args.target_tensor[...] = sub_result

        return args.target_tensor


def generate_all_input_rotation_cell_makers() -> Iterator[CellMaker]:
    yield reg_input_rotation_gate("X^(A/2^n)", ops.X, +1)
    yield reg_input_rotation_gate("Y^(A/2^n)", ops.Y, +1)
    yield reg_input_rotation_gate("Z^(A/2^n)", ops.Z, +1)
    yield reg_input_rotation_gate("X^(-A/2^n)", ops.X, -1)
    yield reg_input_rotation_gate("Y^(-A/2^n)", ops.Y, -1)
    yield reg_input_rotation_gate("Z^(-A/2^n)", ops.Z, -1)


def reg_input_rotation_gate(identifier: str, gate: 'cirq.Gate',
                            factor: float) -> CellMaker:
    return CellMaker(
        identifier, gate.num_qubits(), lambda args: InputRotationCell(
            identifier=identifier,
            register=None,
            register_letter='a',
            target=args.qubits[0],
            op_maker=lambda v, n, qs: gate**(factor * v / n)))
