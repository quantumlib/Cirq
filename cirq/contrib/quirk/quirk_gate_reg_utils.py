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

import inspect
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    List,
    Tuple,
    NamedTuple,
    Iterator,
)

import numpy as np
import sympy.parsing.sympy_parser

import cirq
from cirq import ops
from cirq.contrib.quirk.quirk_parse_gates import (OpsCell, QubitPermutation,
                                                  InputCell, ControlCell,
                                                  DependentCell, ArithmeticCell,
                                                  Cell)

GATE_SIZES = range(1, 17)

CellArgs = NamedTuple('CellArgs', [
    ('qubits', List['cirq.Qid']),
    ('value', Any),
    ('row', int),
    ('col', int),
])


CellType = NamedTuple('CellType', [
    ('identifier', str),
    ('size', int),
    ('func', Callable[[CellArgs], Optional[Cell]]),
])


def reg_gate(identifier: str, gate: cirq.Gate,
             basis_change: cirq.Gate = None) -> Iterator[CellType]:
    yield CellType(
        identifier, gate.num_qubits(), lambda args: OpsCell(
            [gate.on(*args.qubits)],
            basis_change=[basis_change.on(*args.qubits)] if basis_change else ()))


def reg_measurement(identifier: str, basis_change: cirq.Gate = None):
    yield CellType(
        identifier,
        1,
        lambda args: OpsCell(
            [ops.measure(*args.qubits, key=f'row={args.row},col={args.col}')],
            basis_change=[basis_change.on(*args.qubits)] if basis_change else ()))


def reg_family(identifier_prefix: str, gate_maker: Callable[[int], cirq.Gate]
              ) -> Iterator[CellType]:
    f = lambda args: OpsCell([gate_maker(len(args.qubits)).on(*args.qubits)])
    yield CellType(identifier_prefix, 1, f)
    for i in GATE_SIZES:
        yield CellType(identifier_prefix + str(i), i, f)


def reg_formula_gate(
        identifier: str, default_formula: str,
        gate_func: Callable[[Union[sympy.Symbol, float]], cirq.Gate]
) -> Iterator[CellType]:
    yield CellType(
        identifier,
        gate_func(0).num_qubits(), lambda args: OpsCell(
            [gate_func(parse_formula(args.value, default_formula)).on(*args.qubits)]))


def reg_ignored_family(identifier_prefix: str) -> Iterator[CellType]:
    yield from reg_ignored_gate(identifier_prefix)
    for i in GATE_SIZES:
        yield from reg_ignored_gate(identifier_prefix + str(i))


def reg_ignored_gate(identifier: str):
    yield CellType(identifier, 0, lambda _: None)


def reg_unsupported_gate(identifier: str,
                         reason: str) -> Iterator[CellType]:

    def fail(_):
        raise NotImplementedError(
            f'Converting the Quirk gate {identifier} is not implemented yet. '
            f'Reason: {reason}')

    yield CellType(identifier, 0, fail)


def reg_unsupported_gates(*identifiers: str,
                          reason: str) -> Iterator[CellType]:
    for identifier in identifiers:
        yield from reg_unsupported_gate(identifier, reason)


def reg_unsupported_family(identifier_prefix: str,
                           reason: str) -> Iterator[CellType]:
    for i in GATE_SIZES:
        yield from reg_unsupported_gate(identifier_prefix + str(i), reason)


def reg_arithmetic_family(identifier_prefix: str, func: Callable[[Any], int]
                         ) -> Iterator[CellType]:
    yield from reg_size_dependent_arithmetic_family(
        identifier_prefix,
        func=lambda _: func,
        is_modular=False)


def reg_modular_arithmetic_family(
        identifier_prefix: str, func: Callable[[Any], int]
                         ) -> Iterator[CellType]:
    yield from reg_size_dependent_arithmetic_family(
        identifier_prefix,
        func=lambda _: func,
        is_modular=True)


def reg_size_dependent_arithmetic_family(
        identifier_prefix: str,
        func: Callable[[int], Callable[[Any], int]],
        is_modular: bool = False) -> Iterator[CellType]:
    for i in GATE_SIZES:
        yield from reg_arithmetic_gate(identifier_prefix + str(i),
                                       size=i,
                                       func=func(i),
                                       is_modular=is_modular)


def reg_arithmetic_gate(identifier: str,
                        size: int,
                        func: Callable[[Any], int],
                        is_modular: bool = False) -> Iterator[CellType]:
    param_names = list(inspect.signature(func).parameters)
    assert param_names[0] == 'x'
    yield CellType(
        identifier, size, lambda args: ArithmeticCell(
            identifier=identifier,
            registers=[args.qubits] + [None] * len(param_names[1:]),
            register_letters=[None] + param_names[1:],
            operation=func,
            is_modular=is_modular))


def reg_control(identifier: str,
                basis_change: Optional['cirq.Gate']) -> Iterator[CellType]:
    yield CellType(
        identifier, 1, lambda args: ControlCell(
            args.qubits[0],
            basis_change.on(args.qubits[0]) if basis_change else []))


def reg_input_family(identifier_prefix: str, letter: str,
                     rev: bool = False) -> Iterator[CellType]:
    for i in GATE_SIZES:
        yield CellType(
            identifier_prefix + str(i), i, lambda args: InputCell(
                args.qubits[::-1] if rev else args.qubits, letter))


def reg_parameterized_gate(identifier: str, gate: cirq.Gate,
                           factor: float) -> Iterator[CellType]:
    yield CellType(
        identifier, gate.num_qubits(), lambda args: DependentCell(
            identifier=identifier,
            register=None,
            register_letter='a',
            target=args.qubits[0],
            op_maker=lambda v, n, qs: gate**(factor * v / n)))


def reg_const(identifier: str,
              operation: 'cirq.Operation') -> Iterator[CellType]:
    yield CellType(identifier,
                   1, lambda _: OpsCell([operation]))


def reg_bit_permutation_family(identifier_prefix: str,
                               name: str,
                               permutation: Callable[[int, int], int]
                              ) -> Iterator[CellType]:
    f = lambda args: OpsCell(
        [QubitPermutation(name, args.qubits, lambda e: permutation(len(args.qubits), e))])
    for i in GATE_SIZES:
        yield CellType(identifier_prefix + str(i), i, f)


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, y, x = _extended_gcd(b % a, a)
    return gcd, x - (b // a) * y, y


def invertible_else_1(a: int, m: int) -> Optional[int]:
    """Returns `a` if it has a multiplicative inverse, else 1."""
    i = mod_inv_else_1(a, m)
    return a if i != 1 else i


def mod_inv_else_1(a: int, m: int) -> Optional[int]:
    if m == 0:
        return 1
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        return 1
    return x % m


def popcnt(a: int) -> int:
    t = 0
    while a > 0:
        a &= a - 1
        t += 1
    return t


def interleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    group = x // h
    stride = x % h
    return stride * 2 + group


def deinterleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    stride = x // 2
    group = x % 2
    return stride + group * h


def parse_formula(formula: Any,
                  default_formula: Any = None) -> Union[float, sympy.Basic]:
    if formula is None:
        formula = default_formula
    if not isinstance(formula, str):
        raise TypeError('Formula must be a string: {!r}'.format(formula))

    formula = expand_unicode_fractions(formula)
    try:
        result = sympy.parsing.sympy_parser.parse_expr(formula)
    except SyntaxError as ex:
        raise SyntaxError(
            'Failed to parse the gate formula {!r}.\n'
            'This is likely due to differences in how sympy and Quirk parse.\n'
            'For example, Quirk allows "2 pi" whereas sympy requires "2*pi"\n'
            'Parsing of sympy-incompatible formulas is not supported yet.'
            ''.format(formula)) from ex
    if not result.free_symbols <= {sympy.Symbol('t')}:
        raise SyntaxError('Formula has variables besides time "t": {!r}'
                          ''.format(formula))
    if not result.free_symbols:
        result = float(result)
    return result


UNICODE_FRACTIONS = {
    "½": "(1/2)",
    "¼": "(1/4)",
    "¾": "(3/4)",
    "⅓": "(1/3)",
    "⅔": "(2/3)",
    "⅕": "(1/5)",
    "⅖": "(2/5)",
    "⅗": "(3/5)",
    "⅘": "(4/5)",
    "⅙": "(1/6)",
    "⅚": "(5/6)",
    "⅐": "(1/7)",
    "⅛": "(1/8)",
    "⅜": "(3/8)",
    "⅝": "(5/8)",
    "⅞": "(7/8)",
    "⅑": "(1/9)",
    "⅒": "(1/10)",
}


def expand_unicode_fractions(text: str) -> str:
    for k, v in UNICODE_FRACTIONS.items():
        text = text.replace('√' + k, f'(sqrt{v})')
        text = text.replace(k, v)
    return text


def parse_matrix(text: str) -> np.ndarray:
    if not text.startswith('{{') or not text.endswith('}}'):
        raise ValueError('No opening/closing braces.\ntext: {!r}'.format(text))
    text = expand_unicode_fractions(text[2:-2])
    rows = text.split('},{')
    return np.array([[_parse_complex(c) for c in row.split(',')]
                     for row in rows])


def _parse_complex(text: str) -> complex:
    try:
        if (text.endswith('i') and
                len(text) > 1 and
                not text.endswith('+i') and
                not text.endswith('-i')):
            text = text[:-1] + '*i'
        expr = sympy.parsing.sympy_parser.parse_expr(text)
        return complex(expr.subs({'i': 1j}))
    except Exception as ex:
        raise ValueError('Failed to parse complex from {!r}'.format(text)
                         ) from ex
