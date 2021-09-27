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
from typing import (
    Callable,
    Optional,
    Union,
    Iterable,
    Sequence,
    Iterator,
    Tuple,
    Any,
    cast,
    List,
    Dict,
    TYPE_CHECKING,
)

from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES

if TYPE_CHECKING:
    import cirq


@value.value_equality
class QuirkArithmeticOperation(ops.ArithmeticOperation):
    """Applies arithmetic to a target and some inputs.

    Implements Quirk-specific implicit effects like assuming that the presence
    of an 'r' input implies modular arithmetic.

    In Quirk, modular operations have no effect on values larger than the
    modulus. This convention is used because unitarity forces *some* convention
    on out-of-range values (they cannot simply disappear or raise exceptions),
    and the simplest is to do nothing. This call handles ensuring that happens,
    and ensuring the new target register value is normalized modulo the modulus.
    """

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        identifier: str,
        target: Sequence['cirq.Qid'],
        inputs: Sequence[Union[Sequence['cirq.Qid'], int]],
    ):
        """Inits QuirkArithmeticOperation.

        Args:
            identifier: The quirk identifier string for this operation.
            target: The target qubit register.
            inputs: Qubit registers (or classical constants) that
                determine what happens to the target.
        """
        self.identifier = identifier
        self.target: Tuple['cirq.Qid', ...] = tuple(target)
        self.inputs: Tuple[Union[Sequence['cirq.Qid'], int], ...] = tuple(
            e if isinstance(e, int) else tuple(e) for e in inputs
        )

        for input_register in self.inputs:
            if isinstance(input_register, int):
                continue
            if set(self.target) & set(input_register):
                raise ValueError(f'Overlapping registers: {self.target} {self.inputs}')

        if self.operation.is_modular:
            r = inputs[-1]
            if isinstance(r, int):
                over = r > 1 << len(target)
            else:
                over = len(cast(Sequence, r)) > len(target)
            if over:
                raise ValueError(f'Target too small for modulus.\nTarget: {target}\nModulus: {r}')

    # pylint: enable=missing-raises-doc
    @property
    def operation(self) -> '_QuirkArithmeticCallable':
        return ARITHMETIC_OP_TABLE[self.identifier]

    def _value_equality_values_(self) -> Any:
        return self.identifier, self.target, self.inputs

    def registers(self) -> Sequence[Union[int, Sequence['cirq.Qid']]]:
        return [self.target, *self.inputs]

    def with_registers(
        self, *new_registers: Union[int, Sequence['cirq.Qid']]
    ) -> 'QuirkArithmeticOperation':
        if len(new_registers) != len(self.inputs) + 1:
            raise ValueError(
                'Wrong number of registers.\n'
                f'New registers: {repr(new_registers)}\n'
                f'Operation: {repr(self)}'
            )

        if isinstance(new_registers[0], int):
            raise ValueError(
                'The first register is the mutable target. '
                'It must be a list of qubits, not the constant '
                f'{new_registers[0]}.'
            )

        return QuirkArithmeticOperation(self.identifier, new_registers[0], new_registers[1:])

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        return self.operation(*registers)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        lettered_args = list(zip(self.operation.letters, self.inputs))

        result: List[str] = []

        # Target register labels.
        consts = ''.join(
            f',{letter}={reg}' for letter, reg in lettered_args if isinstance(reg, int)
        )
        result.append(f'Quirk({self.identifier}{consts})')
        result.extend(f'#{i}' for i in range(2, len(self.target) + 1))

        # Input register labels.
        for letter, reg in lettered_args:
            if not isinstance(reg, int):
                result.extend(f'{letter.upper()}{i}' for i in range(len(cast(Sequence, reg))))

        return result

    def __repr__(self) -> str:
        return (
            'cirq.interop.quirk.QuirkArithmeticOperation(\n'
            f'    {repr(self.identifier)},\n'
            f'    target={repr(self.target)},\n'
            f'    inputs={_indented_list_lines_repr(self.inputs)},\n'
            ')'
        )


_IntsToIntCallable = Union[
    Callable[[int], int],
    Callable[[int, int], int],
    Callable[[int, int, int], int],
    Callable[[int, int, int, int], int],
]


class _QuirkArithmeticCallable:
    """A callable with parameter-name-dependent behavior."""

    def __init__(self, func: _IntsToIntCallable):
        """Inits _QuirkArithmeticCallable.

        Args:
            func: Maps target int to its output value based on other input ints.
        """
        self.func = func

        # The lambda parameter names indicate the input letter to match.
        letters: List[str] = list(inspect.signature(self.func).parameters)
        # The target is always first, and should be ignored.
        assert letters and letters[0] == 'x'
        self.letters = tuple(letters[1:])

        # The last argument is the modulus r for modular arithmetic.
        self.is_modular = letters[-1] == 'r'

    def __call__(self, *args, **kwargs):
        assert not kwargs
        if self.is_modular:
            if args[0] >= args[-1]:
                return args[0]

        result = self.func(*args)
        if self.is_modular:
            result %= args[-1]
        return result


@value.value_equality
class ArithmeticCell(Cell):
    def __init__(
        self,
        identifier: str,
        target: Sequence['cirq.Qid'],
        inputs: Sequence[Union[None, Sequence['cirq.Qid'], int]],
    ):
        self.identifier = identifier
        self.target = tuple(target)
        self.inputs = tuple(inputs)

    def gate_count(self) -> int:
        return 1

    def _value_equality_values_(self) -> Any:
        return self.identifier, self.target, self.inputs

    def __repr__(self) -> str:
        return (
            f'cirq.interop.quirk.cells.arithmetic_cells.ArithmeticCell('
            f'\n    {self.identifier!r},'
            f'\n    {self.target!r},'
            f'\n    {self.inputs!r})'
        )

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ArithmeticCell(
            identifier=self.identifier,
            target=Cell._replace_qubits(self.target, qubits),
            inputs=[
                e if e is None or isinstance(e, int) else Cell._replace_qubits(e, qubits)
                for e in self.inputs
            ],
        )

    @property
    def operation(self):
        return ARITHMETIC_OP_TABLE[self.identifier]

    def with_input(
        self, letter: str, register: Union[Sequence['cirq.Qid'], int]
    ) -> 'ArithmeticCell':
        new_inputs = [
            reg if letter != reg_letter else register
            for reg, reg_letter in zip(self.inputs, self.operation.letters)
        ]
        return ArithmeticCell(self.identifier, self.target, new_inputs)

    def operations(self) -> 'cirq.OP_TREE':
        missing_inputs = [
            letter for reg, letter in zip(self.inputs, self.operation.letters) if reg is None
        ]
        if missing_inputs:
            raise ValueError(f'Missing input: {sorted(missing_inputs)}')

        return QuirkArithmeticOperation(
            self.identifier,
            self.target,
            cast(Sequence[Union[Sequence['cirq.Qid'], int]], self.inputs),
        )


def _indented_list_lines_repr(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '        ' + '\n        '.join(block.split('\n'))
    return f'[\n{indented}\n    ]'


def _generate_helper() -> Iterator[CellMaker]:
    # Comparisons.
    yield _arithmetic_gate("^A<B", 1, lambda x, a, b: x ^ int(a < b))
    yield _arithmetic_gate("^A>B", 1, lambda x, a, b: x ^ int(a > b))
    yield _arithmetic_gate("^A<=B", 1, lambda x, a, b: x ^ int(a <= b))
    yield _arithmetic_gate("^A>=B", 1, lambda x, a, b: x ^ int(a >= b))
    yield _arithmetic_gate("^A=B", 1, lambda x, a, b: x ^ int(a == b))
    yield _arithmetic_gate("^A!=B", 1, lambda x, a, b: x ^ int(a != b))

    # Addition.
    yield from _arithmetic_family("inc", lambda x: x + 1)
    yield from _arithmetic_family("dec", lambda x: x - 1)
    yield from _arithmetic_family("+=A", lambda x, a: x + a)
    yield from _arithmetic_family("-=A", lambda x, a: x - a)

    # Multiply-accumulate.
    yield from _arithmetic_family("+=AA", lambda x, a: x + a * a)
    yield from _arithmetic_family("-=AA", lambda x, a: x - a * a)
    yield from _arithmetic_family("+=AB", lambda x, a, b: x + a * b)
    yield from _arithmetic_family("-=AB", lambda x, a, b: x - a * b)

    # Misc.
    yield from _arithmetic_family("+cntA", lambda x, a: x + _popcnt(a))
    yield from _arithmetic_family("-cntA", lambda x, a: x - _popcnt(a))
    yield from _arithmetic_family("^=A", lambda x, a: x ^ a)
    yield from _arithmetic_family("Flip<A", lambda x, a: a - x - 1 if x < a else x)

    # Multiplication.
    yield from _arithmetic_family("*A", lambda x, a: x * a if a & 1 else x)
    yield from _size_dependent_arithmetic_family(
        "/A", lambda n: lambda x, a: x * _mod_inv_else_1(a, 1 << n)
    )

    # Modular addition.
    yield from _arithmetic_family("incmodR", lambda x, r: x + 1)
    yield from _arithmetic_family("decmodR", lambda x, r: x - 1)
    yield from _arithmetic_family("+AmodR", lambda x, a, r: x + a)
    yield from _arithmetic_family("-AmodR", lambda x, a, r: x - a)

    # Modular multiply-accumulate.
    yield from _arithmetic_family("+ABmodR", lambda x, a, b, r: x + a * b)
    yield from _arithmetic_family("-ABmodR", lambda x, a, b, r: x - a * b)

    # Modular multiply.
    yield from _arithmetic_family("*AmodR", lambda x, a, r: x * _invertible_else_1(a, r))
    yield from _arithmetic_family("/AmodR", lambda x, a, r: x * _mod_inv_else_1(a, r))
    yield from _arithmetic_family(
        "*BToAmodR", lambda x, a, b, r: x * pow(_invertible_else_1(b, r), a, r)
    )
    yield from _arithmetic_family(
        "/BToAmodR", lambda x, a, b, r: x * pow(_mod_inv_else_1(b, r), a, r)
    )


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, y, x = _extended_gcd(b % a, a)
    return gcd, x - (b // a) * y, y


def _invertible_else_1(a: int, m: int) -> int:
    """Returns `a` if `a` has a multiplicative inverse, else 1."""
    i = _mod_inv_else_1(a, m)
    return a if i != 1 else i


def _mod_inv_else_1(a: int, m: int) -> int:
    """Returns `a**-1` if `a` has a multiplicative inverse, else 1."""
    if m == 0:
        return 1
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        return 1
    return x % m


def _popcnt(a: int) -> int:
    """Returns the Hamming weight of the given non-negative integer."""
    t = 0
    while a > 0:
        a &= a - 1
        t += 1
    return t


def _arithmetic_family(identifier_prefix: str, func: _IntsToIntCallable) -> Iterator[CellMaker]:
    yield from _size_dependent_arithmetic_family(identifier_prefix, size_to_func=lambda _: func)


def _size_dependent_arithmetic_family(
    identifier_prefix: str,
    size_to_func: Callable[[int], _IntsToIntCallable],
) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield _arithmetic_gate(identifier_prefix + str(i), size=i, func=size_to_func(i))


def _arithmetic_gate(identifier: str, size: int, func: _IntsToIntCallable) -> CellMaker:
    operation = _QuirkArithmeticCallable(func)
    assert identifier not in ARITHMETIC_OP_TABLE
    ARITHMETIC_OP_TABLE[identifier] = operation
    return CellMaker(
        identifier=identifier,
        size=size,
        maker=lambda args: ArithmeticCell(
            identifier=identifier, target=args.qubits, inputs=[None] * len(operation.letters)
        ),
    )


ARITHMETIC_OP_TABLE: Dict[str, _QuirkArithmeticCallable] = {}
# Caching is necessary in order to avoid overwriting entries in the table.
_cached_cells: Optional[Tuple[CellMaker, ...]] = None


def generate_all_arithmetic_cell_makers() -> Iterable[CellMaker]:
    global _cached_cells
    if _cached_cells is None:
        _cached_cells = tuple(_generate_helper())
    return _cached_cells
