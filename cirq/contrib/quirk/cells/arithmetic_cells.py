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
                    Tuple, Any, cast, List)

import cirq
from cirq import ops, value
from cirq.contrib.quirk.cells.cell import Cell, CellMaker, CELL_SIZES


@value.value_equality
class QuirkArithmeticOperation(ops.ArithmeticOperation):

    def __init__(self, identifier: str,
                 operation: 'cirq.contrib.quirk.QuirkArithmeticLambda',
                 target: Sequence['cirq.Qid'],
                 inputs: Sequence[Optional[Union[Sequence['cirq.Qid'], int]]]):
        self.identifier = identifier
        self.target = target
        self.inputs = inputs
        self.operation = operation

    def _value_equality_values_(self):
        return self.identifier, self.operation, self.target, self.inputs

    def registers(self):
        return [self.target, *self.inputs]

    def with_registers(self, *new_registers: Union[int, Sequence['cirq.Qid']]
                      ) -> 'cirq.ArithmeticOperation':
        if isinstance(new_registers[0], int):
            raise ValueError('The first register is the mutable target. '
                             'It must be a list of qubits, not the constant '
                             f'{new_registers[0]}.')

        return QuirkArithmeticOperation(self.identifier, self.operation,
                                        new_registers[0], new_registers[1:])

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        return self.operation(*registers)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        lettered_args = list(zip(self.operation.letters, self.inputs))

        result = []

        # Target register labels.
        consts = ''.join(f',{letter}={reg}' for letter, reg in lettered_args
                         if isinstance(reg, int))
        result.append(f'Quirk({self.identifier}{consts})')
        result.extend(f'#{i}' for i in range(2, len(self.target) + 1))

        # Input register labels.
        for letter, reg in lettered_args:
            if not isinstance(reg, int):
                result.extend(f'{letter.upper()}{i}'
                              for i in range(len(cast(Sequence, reg))))

        return result

    def __str__(self):
        return f'Quirk({self.identifier})'

    def __repr__(self):
        return ('cirq.contrib.quirk.QuirkArithmeticOperation(\n'
                f'    {repr(self.identifier)},\n'
                f'    operation={repr(self.operation)},\n'
                f'    target={repr(self.target)},\n'
                f'    inputs={_indented_list_lines_repr(self.inputs)},\n'
                ')')


@value.value_equality
class QuirkArithmeticLambda:
    """A callable with parameter name dependent behavior."""

    def __init__(self, code: str):
        """
        Args:
            code: A string containing a python lambda expression using only
                values built into python or cirq. The lambda expression is
                passed as a string to guarantee it has a `repr`.
        """
        self.code = code
        self._func: Callable = eval(code, {'cirq': cirq}, {})

        # The lambda parameter names indicate the input letter to match.
        letters: List[str] = list(inspect.signature(self._func).parameters)
        # The target is always first, and should be ignored.
        assert letters and letters[0] == 'x'
        self.letters = tuple(letters[1:])

        # The last argument is the modulus r for modular arithmetic.
        self.is_modular = letters[-1] == 'r'

    def _value_equality_values_(self):
        return self.code

    def __call__(self, *args, **kwargs):
        assert not kwargs
        if self.is_modular:
            if args[0] >= args[-1]:
                return args[0]

        result = self._func(*args)
        if self.is_modular:
            result %= args[-1]
        return result

    def __repr__(self):
        return 'cirq.contrib.quirk.QuirkArithmeticLambda({!r})'.format(
            self.code)


class ArithmeticCell(Cell):

    def __init__(self, identifier: str, operation: QuirkArithmeticLambda,
                 target: Sequence['cirq.Qid'],
                 inputs: Sequence[Union[None, Sequence['cirq.Qid'], int]]):
        self.identifier = identifier
        self.operation = operation
        self.target = target
        self.inputs = inputs

    def with_input(self, letter, register):
        new_inputs = [
            reg if letter != reg_letter else register
            for reg, reg_letter in zip(self.inputs, self.operation.letters)
        ]
        return ArithmeticCell(self.identifier, self.operation, self.target,
                              new_inputs)

    def operations(self) -> 'cirq.OP_TREE':
        missing_inputs = [
            letter for reg, letter in zip(self.inputs, self.operation.letters)
            if reg is None
        ]
        if missing_inputs:
            raise ValueError(f'Missing input: {sorted(missing_inputs)}')

        if self.operation.is_modular:
            r = cast(Union[Sequence['cirq.Qid'], int], self.inputs[-1])
            assert r is not None
            if isinstance(r, int):
                over = r > 1 << len(self.target)
            else:
                over = len(cast(Sequence, r)) > len(self.target)
            if over:
                raise ValueError('Target too small for modulus.\n'
                                 f'Target: {self.target}\n'
                                 f'Modulus: {r}')

        return QuirkArithmeticOperation(self.identifier, self.operation,
                                        self.target, self.inputs)


def _indented_list_lines_repr(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '        ' + '\n        '.join(block.split('\n'))
    return '[\n{}\n    ]'.format(indented)


def generate_all_arithmetic_cell_makers() -> Iterator[CellMaker]:
    # Comparisons.
    yield _reg_arithmetic_gate("^A<B", 1, "lambda x, a, b: x ^ int(a < b)")
    yield _reg_arithmetic_gate("^A>B", 1, "lambda x, a, b: x ^ int(a > b)")
    yield _reg_arithmetic_gate("^A<=B", 1, "lambda x, a, b: x ^ int(a <= b)")
    yield _reg_arithmetic_gate("^A>=B", 1, "lambda x, a, b: x ^ int(a >= b)")
    yield _reg_arithmetic_gate("^A=B", 1, "lambda x, a, b: x ^ int(a == b)")
    yield _reg_arithmetic_gate("^A!=B", 1, "lambda x, a, b: x ^ int(a != b)")

    # Addition.
    yield from _reg_arithmetic_family("inc", "lambda x: x + 1")
    yield from _reg_arithmetic_family("dec", "lambda x: x - 1")
    yield from _reg_arithmetic_family("+=A", "lambda x, a: x + a")
    yield from _reg_arithmetic_family("-=A", "lambda x, a: x - a")

    # Multiply-accumulate.
    yield from _reg_arithmetic_family("+=AA", "lambda x, a: x + a * a")
    yield from _reg_arithmetic_family("-=AA", "lambda x, a: x - a * a")
    yield from _reg_arithmetic_family("+=AB", "lambda x, a, b: x + a * b")
    yield from _reg_arithmetic_family("-=AB", "lambda x, a, b: x - a * b")

    # Misc.
    yield from _reg_arithmetic_family(
        "+cntA", "lambda x, a: x + cirq.contrib.quirk.popcnt(a)")
    yield from _reg_arithmetic_family(
        "-cntA", "lambda x, a: x - cirq.contrib.quirk.popcnt(a)")
    yield from _reg_arithmetic_family("^=A", "lambda x, a: x ^ a")
    yield from _reg_arithmetic_family("Flip<A",
                                      "lambda x, a: a - x - 1 if x < a else x")

    # Multiplication.
    yield from _reg_arithmetic_family("*A",
                                      "lambda x, a: x * a if a & 1 else x")
    yield from _reg_size_dependent_arithmetic_family(
        "/A", lambda n:
        f"lambda x, a: x * cirq.contrib.quirk.mod_inv_else_1(a, 1 << {n})")

    # Modular addition.
    yield from _reg_arithmetic_family("incmodR", "lambda x, r: x + 1")
    yield from _reg_arithmetic_family("decmodR", "lambda x, r: x - 1")
    yield from _reg_arithmetic_family("+AmodR", "lambda x, a, r: x + a")
    yield from _reg_arithmetic_family("-AmodR", "lambda x, a, r: x - a")

    # Modular multiply-accumulate.
    yield from _reg_arithmetic_family("+ABmodR", "lambda x, a, b, r: x + a * b")
    yield from _reg_arithmetic_family("-ABmodR", "lambda x, a, b, r: x - a * b")

    # Modular multiply.
    yield from _reg_arithmetic_family(
        "*AmodR",
        "lambda x, a, r: x * cirq.contrib.quirk.invertible_else_1(a, r)")
    yield from _reg_arithmetic_family(
        "/AmodR", "lambda x, a, r: x * cirq.contrib.quirk.mod_inv_else_1(a, r)")
    yield from _reg_arithmetic_family(
        "*BToAmodR", "lambda x, a, b, r: "
        "x * pow(cirq.contrib.quirk.invertible_else_1(b, r), a, r)")
    yield from _reg_arithmetic_family(
        "/BToAmodR", "lambda x, a, b, r: "
        "x * pow(cirq.contrib.quirk.mod_inv_else_1(b, r), a, r)")


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, y, x = _extended_gcd(b % a, a)
    return gcd, x - (b // a) * y, y


def invertible_else_1(a: int, m: int) -> int:
    """Returns `a` if `a` has a multiplicative inverse, else 1."""
    i = mod_inv_else_1(a, m)
    return a if i != 1 else i


def mod_inv_else_1(a: int, m: int) -> int:
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


def _reg_arithmetic_family(identifier_prefix: str,
                           func: str) -> Iterator[CellMaker]:
    yield from _reg_size_dependent_arithmetic_family(
        identifier_prefix, size_to_func=lambda _: func)


def _reg_size_dependent_arithmetic_family(
        identifier_prefix: str,
        size_to_func: Callable[[int], str],
) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield _reg_arithmetic_gate(identifier_prefix + str(i),
                                   size=i,
                                   func=size_to_func(i))


def _reg_arithmetic_gate(identifier: str, size: int, func: str) -> CellMaker:
    operation = QuirkArithmeticLambda(func)
    return CellMaker(identifier=identifier,
                     size=size,
                     maker=lambda args: ArithmeticCell(identifier=identifier,
                                                       target=args.qubits,
                                                       inputs=[None] * len(
                                                           operation.letters),
                                                       operation=operation))
