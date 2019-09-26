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
import inspect
from typing import (Callable, Optional, Union, Iterable, Sequence, Iterator,
                    Tuple, Any, cast)

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import Cell, CellMaker, CELL_SIZES


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
        return ArithmeticCell(self.identifier, [
            reg if letter != reg_letter else register
            for reg, reg_letter in zip(self._registers, self._register_letters)
        ], self._register_letters, self._operation, self._is_modular)

    def operations(self) -> 'cirq.OP_TREE':
        missing_inputs = [
            letter
            for reg, letter in zip(self._registers, self._register_letters)
            if reg is None
        ]
        if missing_inputs:
            raise ValueError(f'Missing input: {sorted(missing_inputs)}')

        if self._is_modular:
            assert self._register_letters.index(None) == 0
            assert self._register_letters.index('r') == len(
                self._register_letters) - 1
            x = cast(Union[Sequence['cirq.Qid'], int], self._registers[0])
            r = cast(Union[Sequence['cirq.Qid'], int], self._registers[-1])
            assert x is not None
            assert r is not None
            if r > 1 << len(x) if isinstance(r, int) else len(r) > len(x):
                raise ValueError('Target too small for modulus.\n'
                                 f'Target: {x}\n'
                                 f'Modulus: {r}')

        return QuirkArithmeticOperation(self.identifier, self._registers,
                                        self._register_letters, self._operation,
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
        return QuirkArithmeticOperation(self.identifier, new_registers,
                                        self._register_letters, self._operation,
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
            self.identifier, self._register_letters, self.registers(),
            self._operation)


def generate_all_arithmetic_cell_makers() -> Iterator[CellMaker]:
    # Comparisons.
    yield reg_arithmetic_gate("^A<B", 1, lambda x, a, b: x ^ int(a < b))
    yield reg_arithmetic_gate("^A>B", 1, lambda x, a, b: x ^ int(a > b))
    yield reg_arithmetic_gate("^A<=B", 1, lambda x, a, b: x ^ int(a <= b))
    yield reg_arithmetic_gate("^A>=B", 1, lambda x, a, b: x ^ int(a >= b))
    yield reg_arithmetic_gate("^A=B", 1, lambda x, a, b: x ^ int(a == b))
    yield reg_arithmetic_gate("^A!=B", 1, lambda x, a, b: x ^ int(a != b))

    # Addition.
    yield from reg_arithmetic_family("inc", lambda x: x + 1)
    yield from reg_arithmetic_family("dec", lambda x: x - 1)
    yield from reg_arithmetic_family("+=A", lambda x, a: x + a)
    yield from reg_arithmetic_family("-=A", lambda x, a: x - a)

    # Multiply-accumulate.
    yield from reg_arithmetic_family("+=AA", lambda x, a: x + a * a)
    yield from reg_arithmetic_family("-=AA", lambda x, a: x - a * a)
    yield from reg_arithmetic_family("+=AB", lambda x, a, b: x + a * b)
    yield from reg_arithmetic_family("-=AB", lambda x, a, b: x - a * b)

    # Misc.
    yield from reg_arithmetic_family("+cntA", lambda x, a: x + popcnt(a))
    yield from reg_arithmetic_family("-cntA", lambda x, a: x - popcnt(a))
    yield from reg_arithmetic_family("^=A", lambda x, a: x ^ a)
    yield from reg_arithmetic_family(
        "Flip<A", lambda x, a: a - x - 1 if x < a else x)

    # Multiplication.
    yield from reg_arithmetic_family("*A", lambda x, a: x * a if a & 1 else x)
    yield from reg_size_dependent_arithmetic_family(
        "/A", lambda n: lambda x, a: x * mod_inv_else_1(a, 1 << n))

    # Modular addition.
    yield from reg_modular_arithmetic_family("incmodR", lambda x, r: x + 1)
    yield from reg_modular_arithmetic_family("decmodR", lambda x, r: x - 1)
    yield from reg_modular_arithmetic_family("+AmodR", lambda x, a, r: x + a)
    yield from reg_modular_arithmetic_family("-AmodR", lambda x, a, r: x - a)

    # Modular multiply-accumulate.
    yield from reg_modular_arithmetic_family(
        "+ABmodR", lambda x, a, b, r: x + a * b)
    yield from reg_modular_arithmetic_family(
        "-ABmodR", lambda x, a, b, r: x - a * b)

    # Modular multiply.
    yield from reg_modular_arithmetic_family(
        "*AmodR", lambda x, a, r: x * invertible_else_1(a, r))
    yield from reg_modular_arithmetic_family(
        "/AmodR", lambda x, a, r: x * mod_inv_else_1(a, r))
    yield from reg_modular_arithmetic_family(
        "*BToAmodR", lambda x, a, b, r: x * pow(invertible_else_1(b, r), a, r))
    yield from reg_modular_arithmetic_family(
        "/BToAmodR", lambda x, a, b, r: x * pow(mod_inv_else_1(b, r), a, r))


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, y, x = _extended_gcd(b % a, a)
    return gcd, x - (b // a) * y, y


def invertible_else_1(a: int, m: int) -> Optional[int]:
    """Returns `a` if `a` has a multiplicative inverse, else 1."""
    i = mod_inv_else_1(a, m)
    return a if i != 1 else i


def mod_inv_else_1(a: int, m: int) -> Optional[int]:
    """Returns `a**-1` if `a` has a multiplicative inverse, else 1."""
    if m == 0:
        return 1
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        return 1
    return x % m


def popcnt(a: int) -> int:
    """Returns the Hamming weight of the given non-negative integer."""
    t = 0
    while a > 0:
        a &= a - 1
        t += 1
    return t


IntsToIntCallable = Union[Callable[[int], int], Callable[[int, int], int],
                          Callable[[int, int, int], int],
                          Callable[[int, int, int, int], int],]


def reg_arithmetic_family(identifier_prefix: str,
                          func: IntsToIntCallable) -> Iterator[CellMaker]:
    yield from reg_size_dependent_arithmetic_family(identifier_prefix,
                                                    func=lambda _: func,
                                                    is_modular=False)


def reg_modular_arithmetic_family(identifier_prefix: str,
                                  func: IntsToIntCallable
                                 ) -> Iterator[CellMaker]:
    yield from reg_size_dependent_arithmetic_family(identifier_prefix,
                                                    func=lambda _: func,
                                                    is_modular=True)


def reg_size_dependent_arithmetic_family(
        identifier_prefix: str,
        func: Callable[[int], IntsToIntCallable],
        is_modular: bool = False) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield reg_arithmetic_gate(identifier_prefix + str(i),
                                  size=i,
                                  func=func(i),
                                  is_modular=is_modular)


def reg_arithmetic_gate(identifier: str,
                        size: int,
                        func: IntsToIntCallable,
                        is_modular: bool = False) -> CellMaker:
    param_names = list(inspect.signature(func).parameters)
    assert param_names[0] == 'x'
    if is_modular:
        assert param_names[-1] == 'r'
    return CellMaker(
        identifier, size, lambda args: ArithmeticCell(
            identifier=identifier,
            registers=[args.qubits] + [None] * len(param_names[1:]),
            register_letters=[None] + param_names[1:],
            operation=func,
            is_modular=is_modular))
