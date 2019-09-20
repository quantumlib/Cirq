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

import sympy

import cirq
from cirq.contrib.quirk.quirk_parse_gates import (OpsCell, QubitPermutation,
                                                  InputCell, ControlCell,
                                                  DependentCell, ArithmeticCell,
                                                  Cell)

GATE_SIZES = range(1, 17)

RegistryEntry = NamedTuple('RegistryEntry', [
    ('identifier', str),
    ('size', int),
    ('func', Callable[[List['cirq.Qid'], Any], Optional[Cell]]),
])


def reg_gate(identifier: str, gate: cirq.Gate,
             basis_change: cirq.Gate = None) -> Iterator[RegistryEntry]:
    yield RegistryEntry(
        identifier, gate.num_qubits(), lambda qubits, _: OpsCell(
            [gate.on(*qubits)], basis_change=[basis_change.on(*qubits)] if basis_change else ()))


def reg_family(identifier_prefix: str, gate_maker: Callable[[int], cirq.Gate]
              ) -> Iterator[RegistryEntry]:
    f = lambda qubits, _: OpsCell([gate_maker(len(qubits)).on(*qubits)])
    yield RegistryEntry(identifier_prefix, 1, f)
    for i in GATE_SIZES:
        yield RegistryEntry(identifier_prefix + str(i), i, f)


def reg_formula_gate(
        identifier: str, default_formula: str,
        gate_func: Callable[[Union[sympy.Symbol, float]], cirq.Gate]
) -> Iterator[RegistryEntry]:
    yield RegistryEntry(
        identifier,
        gate_func(0).num_qubits(), lambda qubits, value: OpsCell(
            [gate_func(parse_formula(value, default_formula)).on(*qubits)]))


def reg_ignored_family(identifier_prefix: str) -> Iterator[RegistryEntry]:
    yield from reg_ignored_gate(identifier_prefix)
    for i in GATE_SIZES:
        yield from reg_ignored_gate(identifier_prefix + str(i))


def reg_ignored_gate(identifier: str):
    yield RegistryEntry(identifier, 0, lambda a, b: None)


def reg_unsupported_gate(identifier: str,
                         reason: str) -> Iterator[RegistryEntry]:

    def fail(qubits, value):
        raise NotImplementedError(
            f'Converting the Quirk gate {identifier} is not implemented yet. '
            f'Reason: {reason}')

    yield RegistryEntry(identifier, 0, fail)


def reg_unsupported_gates(*identifiers: str,
                          reason: str) -> Iterator[RegistryEntry]:
    for identifier in identifiers:
        yield from reg_unsupported_gate(identifier, reason)


def reg_unsupported_family(identifier_prefix: str,
                           reason: str) -> Iterator[RegistryEntry]:
    for i in GATE_SIZES:
        yield from reg_unsupported_gate(identifier_prefix + str(i), reason)


def reg_arithmetic_family(identifier_prefix: str, func: Callable[[Any], int]
                         ) -> Iterator[RegistryEntry]:
    for i in GATE_SIZES:
        yield from reg_arithmetic_gate(identifier_prefix + str(i), i, func)


def reg_size_dependent_arithmetic_family(
        identifier_prefix: str,
        func: Callable[[int], Callable[[Any], int]]) -> Iterator[RegistryEntry]:
    for i in GATE_SIZES:
        yield from reg_arithmetic_gate(identifier_prefix + str(i), i, func(i))


def reg_arithmetic_gate(identifier: str, size: int,
                        func: Callable[[Any], int]) -> Iterator[RegistryEntry]:
    param_names = list(inspect.signature(func).parameters)
    assert param_names[0] == 'x'
    yield RegistryEntry(
        identifier, size, lambda qubits, _: ArithmeticCell(
            identifier=identifier,
            registers=[qubits] + param_names[1:],
            operation=func))


def reg_control(identifier: str,
                basis_change: Optional['cirq.Gate']) -> Iterator[RegistryEntry]:
    yield RegistryEntry(
        identifier, 1, lambda qubits, _: ControlCell(
            qubits[0],
            basis_change.on(qubits[0]) if basis_change else []))


def reg_input_family(identifier_prefix: str, letter: str,
                     rev: bool = False) -> Iterator[RegistryEntry]:
    for i in GATE_SIZES:
        yield RegistryEntry(
            identifier_prefix + str(i), i, lambda qubits, _: InputCell(
                qubits[::-1] if rev else qubits, 'a'))


def reg_parameterized_gate(identifier: str, gate: cirq.Gate,
                           factor: float) -> Iterator[RegistryEntry]:
    yield RegistryEntry(
        identifier, gate.num_qubits(), lambda qubits, _: DependentCell(
            register='a',
            target=qubits[0],
            op_maker=lambda v, n, qs: gate**(factor * v / n)))


def reg_const(identifier: str,
              operation: 'cirq.Operation') -> Iterator[RegistryEntry]:
    yield RegistryEntry(identifier,
                        1, lambda qubits, value: OpsCell([operation]))


def reg_bit_permutation_family(identifier_prefix,
                               permutation: Callable[[int, int], int]
                              ) -> Iterator[RegistryEntry]:
    f = lambda qubits, _: OpsCell(
        [QubitPermutation(qubits, lambda e: permutation(len(qubits), e))])
    for i in GATE_SIZES:
        yield RegistryEntry(identifier_prefix + str(i),
                                 i,
                                 f)


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, y, x = _extended_gcd(b % a, a)
    return gcd, x - (b // a) * y, y


def modular_multiplicative_inverse(a: int, m: int) -> Optional[int]:
    if m == 0:
        return None
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        return None
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
                  default_formula: str) -> Union[float, sympy.Basic]:
    if formula is None:
        formula = default_formula
    if not isinstance(formula, str):
        raise TypeError('Formula must be a string: {!r}'.format(formula))

    from sympy.parsing.sympy_parser import parse_expr
    try:
        result = parse_expr(formula)
    except SyntaxError as ex:
        raise SyntaxError(
            'Failed to parse the gate formula {!r}.\n'
            'This is likely due to differences in how sympy and Quirk parse.\n'
            'For example, Quirk allows "2 pi" whereas sympy requires "2*pi"\n'
            'Parsing of sympy-incompatible formulas is not supported yet.'
            ''.format(formula)) from ex
    if not result.free_symbols <= {'t'}:
        raise SyntaxError('Formula has variables besides time "t": {!r}'
                          ''.format(formula))
    if not result.free_symbols:
        result = float(result)
    return result
