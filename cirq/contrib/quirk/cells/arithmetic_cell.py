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

from typing import Callable, Optional, Union, Iterable, Sequence

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import Cell


class ArithmeticCell(Cell):

    def __init__(
            self, identifier: str,
            registers: Sequence[Optional[Union[Sequence['cirq.Qid'], int]]],
            register_letters: Sequence[Optional[str]], operation: Callable,
            is_modular: bool):
        self.identifier = identifier
        self._registers = registers
        self._register_letters = register_letters
        self._operation = operation
        self._is_modular = is_modular

    def with_input(self, letter, register):
        return ArithmeticCell(
            self.identifier,
            [
                reg if letter != reg_letter else register
                for reg, reg_letter in
                zip(self._registers, self._register_letters)
            ],
            self._register_letters,
            self._operation,
            self._is_modular
        )

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

        return QuirkArithmeticOperation(
            self.identifier,
            self._registers,
            self._register_letters,
            self._operation,
            self._is_modular)


class QuirkArithmeticOperation(ops.ArithmeticOperation):

    def __init__(
            self, identifier: str,
            registers: Sequence[Optional[Union[Sequence['cirq.Qid'], int]]],
            register_letters: Sequence[Optional[str]], operation: Callable,
            is_modular: bool):
        self.identifier = identifier
        self._registers = registers
        self._register_letters = register_letters
        self._operation = operation
        self._is_modular = is_modular

    def registers(self):
        return self._registers

    def with_registers(self, *new_registers: Union[int, Sequence['cirq.Qid']]
                      ) -> 'cirq.ArithmeticOperation':
        return QuirkArithmeticOperation(
            self.identifier,
            new_registers,
            self._register_letters,
            self._operation,
            self._is_modular)

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        if self._is_modular:
            if registers[0] >= registers[-1]:
                return registers[0]
        result = self._operation(*registers)
        if self._is_modular:
            result %= registers[-1]
        return result

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        consts = ''.join(
            f',{letter}={reg}'
            for reg, letter in zip(self._registers, self._register_letters)
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
        return f'Quirk({self.identifier})'

    def __repr__(self):
        return 'cirq.quirk.QuirkArithmeticOperation({!r}, {!r}, {!r}, {!r})'.format(
            self.identifier,
            self._register_letters,
            self.registers(),
            self._operation)
