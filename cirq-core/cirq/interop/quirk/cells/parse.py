# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
    SupportsFloat,
    TypeVar,
)

import cmath
import re

import numpy as np
import sympy


def _merge_scientific_float_tokens(tokens: Iterable[str]) -> List[str]:
    tokens = list(tokens)
    i = 0
    while 'e' in tokens[i + 1 :]:
        i = tokens.index('e', i + 1)
        s = i - 1
        e = i + 1
        if not re.match('[0-9]', str(tokens[s])):
            continue
        if re.match('[+-]', str(tokens[e])):
            e += 1
        if re.match('[0-9]', str(tokens[e])):
            e += 1
            tokens[s:e] = [''.join(tokens[s:e])]
            i -= 1
    return tokens


T = TypeVar('T')


def _segment_by(seq: Iterable[T], *, key: Callable[[T], Any]) -> Iterator[List[T]]:
    group: List[T] = []
    last_key = None
    for item in seq:
        item_key = key(item)
        if len(group) and item_key != last_key:
            yield group
            group = []
        group.append(item)
        last_key = item_key
    if len(group) > 0:
        yield group


def _tokenize(text: str) -> List[str]:
    def classify(e: str) -> Union[str, float]:
        assert e.strip() != ''  # Because _segment_by drops empty entries.
        if re.match(r'[.0-9]', e):
            return "#"
        if re.match(r'[_a-z]', e):
            return "a"
        return np.nan  # Always split.

    result = []
    for part in re.split(r'\s', text.lower()):
        for group in _segment_by(part, key=classify):
            result.append(''.join(group))

    return _merge_scientific_float_tokens(g for g in result if g.strip())


_ResolvedToken = Union[sympy.Expr, complex]


class _CustomQuirkOperationToken:
    def __init__(
        self,
        unary_action: Optional[Callable[[_ResolvedToken], _ResolvedToken]],
        binary_action: Optional[Callable[[_ResolvedToken, _ResolvedToken], _ResolvedToken]],
        priority: float,
    ):
        self.unary_action = unary_action
        self.binary_action = binary_action
        self.priority = priority


class _HangingNode:
    def __init__(
        self, func: Callable[[_ResolvedToken, _ResolvedToken], _ResolvedToken], weight: float
    ):
        self.func = func
        self.weight = weight


_HangingToken = Union[_ResolvedToken, str, _CustomQuirkOperationToken]


def _translate_token(token_id: str, token_map: Mapping[str, _HangingToken]) -> _HangingToken:
    if re.match(r'[0-9]+(\.[0-9]+)?', token_id):
        return float(token_id)

    if token_id in token_map:
        return token_map[token_id]

    raise ValueError(f"Unrecognized token: {token_id}")


def _parse_formula_using_token_map(
    text: str, token_map: Dict[str, _HangingToken]
) -> _ResolvedToken:
    """Parses a value from an infix arithmetic expression."""
    tokens: List[_HangingToken] = [_translate_token(e, token_map) for e in _tokenize(text)]

    # Cut off trailing operation, so parse fails less often as users are typing.
    if (
        len(tokens)
        and isinstance(tokens[-1], _CustomQuirkOperationToken)
        and tokens[-1].priority is not None
    ):
        tokens = tokens[:-1]

    ops: List[Union[str, _HangingNode]] = []
    vals: List[Optional[_HangingToken]] = []

    # Hack: use the 'priority' field as a signal of 'is an operation'
    def is_valid_end_token(tok: _HangingToken) -> bool:
        return tok != "(" and not isinstance(tok, _CustomQuirkOperationToken)

    def is_valid_end_state() -> bool:
        return len(vals) == 1 and len(ops) == 0

    def apply(op: Union[str, _HangingNode]) -> None:
        assert isinstance(op, _HangingNode)
        if len(vals) < 2:
            raise ValueError("Bad expression: operated on nothing.\ntext={text!r}")
        b = vals.pop()
        a = vals.pop()
        # Note: vals seems to be _HangingToken
        # func operates on _ResolvedTokens. Ignoring type issues for now.
        vals.append(op.func(a, b))

    def close_paren() -> None:
        while True:
            if len(ops) == 0:
                raise ValueError("Bad expression: unmatched ')'.\ntext={text!r}")
            op = ops.pop()
            if op == "(":
                break
            apply(op)

    def burn_ops(w: float) -> None:
        while len(ops) and len(vals) >= 2 and vals[-1] is not None:
            top = ops[-1]
            if not isinstance(top, _HangingNode) or top.weight is None or top.weight < w:
                break
            apply(ops.pop())

    def feed_op(could_be_binary: bool, token: Any) -> None:
        # Implied multiplication?
        mul = cast(_CustomQuirkOperationToken, token_map["*"])
        if could_be_binary and token != ")":
            if not isinstance(token, _CustomQuirkOperationToken) or token.binary_action is None:
                burn_ops(mul.priority)
                ops.append(
                    _HangingNode(
                        func=cast(Callable[[T, T], T], mul.binary_action), weight=mul.priority
                    )
                )

        if isinstance(token, _CustomQuirkOperationToken):
            if could_be_binary and token.binary_action is not None:
                burn_ops(token.priority)
                ops.append(_HangingNode(func=token.binary_action, weight=token.priority))
            elif token.unary_action is not None:
                burn_ops(token.priority)
                vals.append(None)
                # this avoids mypy complaint about None not being callable
                token_unary_action = token.unary_action
                ops.append(_HangingNode(func=lambda _, b: token_unary_action(b), weight=np.inf))
            elif token.binary_action is not None:
                raise ValueError("Bad expression: binary op in bad spot.\ntext={text!r}")

    was_valid_end_token = False
    for token in tokens:
        feed_op(was_valid_end_token, token)
        was_valid_end_token = is_valid_end_token(token)

        if token == "(":
            ops.append("(")
        elif token == ")":
            close_paren()
        elif was_valid_end_token:
            vals.append(token)

    burn_ops(-np.inf)

    if not is_valid_end_state():
        raise ValueError(f"Incomplete expression.\ntext={text!r}")

    return cast(_ResolvedToken, vals[0])


UNICODE_FRACTIONS = {
    "½": 1 / 2,
    "¼": 1 / 4,
    "¾": 3 / 4,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
    "⅕": 1 / 5,
    "⅖": 2 / 5,
    "⅗": 3 / 5,
    "⅘": 4 / 5,
    "⅙": 1 / 6,
    "⅚": 5 / 6,
    "⅐": 1 / 7,
    "⅛": 1 / 8,
    "⅜": 3 / 8,
    "⅝": 5 / 8,
    "⅞": 7 / 8,
    "⅑": 1 / 9,
    "⅒": 1 / 10,
}

PARSE_COMPLEX_TOKEN_MAP_ALL: Dict[str, _HangingToken] = {
    **UNICODE_FRACTIONS,
    'i': 1j,
    'e': sympy.E,
    'pi': sympy.pi,
    '(': '(',
    ')': ')',
    'sqrt': _CustomQuirkOperationToken(unary_action=sympy.sqrt, binary_action=None, priority=4),
    'exp': _CustomQuirkOperationToken(unary_action=sympy.exp, binary_action=None, priority=4),
    'ln': _CustomQuirkOperationToken(unary_action=sympy.log, binary_action=None, priority=4),
    '^': _CustomQuirkOperationToken(
        unary_action=None, binary_action=lambda a, b: a ** complex(b), priority=3
    ),
    '*': _CustomQuirkOperationToken(
        unary_action=None, binary_action=lambda a, b: a * b, priority=2
    ),
    '/': _CustomQuirkOperationToken(
        unary_action=None, binary_action=lambda a, b: a / b, priority=2
    ),
    '+': _CustomQuirkOperationToken(
        unary_action=lambda e: e, binary_action=lambda a, b: a + b, priority=1
    ),
    '-': _CustomQuirkOperationToken(
        unary_action=lambda a: -a, binary_action=lambda a, b: a - b, priority=1
    ),
}
PARSE_COMPLEX_TOKEN_MAP_ALL["√"] = PARSE_COMPLEX_TOKEN_MAP_ALL["sqrt"]

PARSE_COMPLEX_TOKEN_MAP_DEG: Dict[str, _HangingToken] = {
    **PARSE_COMPLEX_TOKEN_MAP_ALL,
    "cos": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.cos(e * sympy.pi / 180), binary_action=None, priority=4
    ),
    "sin": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.sin(e * sympy.pi / 180), binary_action=None, priority=4
    ),
    "asin": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.asin(e) * 180 / sympy.pi, binary_action=None, priority=4
    ),
    "acos": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.acos(e) * 180 / sympy.pi, binary_action=None, priority=4
    ),
}
PARSE_COMPLEX_TOKEN_MAP_DEG["arccos"] = PARSE_COMPLEX_TOKEN_MAP_DEG["acos"]
PARSE_COMPLEX_TOKEN_MAP_DEG["arcsin"] = PARSE_COMPLEX_TOKEN_MAP_DEG["asin"]

PARSE_COMPLEX_TOKEN_MAP_RAD: Dict[str, _HangingToken] = {
    **PARSE_COMPLEX_TOKEN_MAP_ALL,
    "cos": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.cos(e) if isinstance(e, sympy.Basic) else cmath.cos(e),
        binary_action=None,
        priority=4,
    ),
    "sin": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.sin(e) if isinstance(e, sympy.Basic) else cmath.sin(e),
        binary_action=None,
        priority=4,
    ),
    "asin": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.asin(e) if isinstance(e, sympy.Basic) else np.arcsin(e),
        binary_action=None,
        priority=4,
    ),
    "acos": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.acos(e) if isinstance(e, sympy.Basic) else np.arccos(e),
        binary_action=None,
        priority=4,
    ),
    "tan": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.tan(e) if isinstance(e, sympy.Basic) else np.tan(e),
        binary_action=None,
        priority=4,
    ),
    "atan": _CustomQuirkOperationToken(
        unary_action=lambda e: sympy.atan(e) if isinstance(e, sympy.Basic) else np.arctan(e),
        binary_action=None,
        priority=4,
    ),
}


def parse_matrix(text: str) -> np.ndarray:
    """Attempts to parse a complex matrix in exactly the same way as Quirk."""
    text = re.sub(r'\s', '', text)
    if len(text) < 4 or text[:2] != "{{" or text[-2:] != "}}":
        raise ValueError("Not surrounded by {{}}.")
    return np.array(
        [[parse_complex(cell) for cell in row.split(',')] for row in text[2:-2].split('},{')],
        dtype=np.complex128,
    )


def parse_complex(text: str) -> complex:
    """Attempts to parse a complex number in exactly the same way as Quirk."""
    try:
        return complex(_parse_formula_using_token_map(text, PARSE_COMPLEX_TOKEN_MAP_DEG))
    except Exception as ex:
        raise ValueError(f'Failed to parse complex from {text!r}') from ex


def parse_formula(formula: str) -> Union[float, sympy.Expr]:
    """Attempts to parse formula text in exactly the same way as Quirk."""
    if not isinstance(formula, str):
        raise TypeError('formula must be a string')

    token_map: Dict[str, _HangingToken] = {**PARSE_COMPLEX_TOKEN_MAP_RAD, 't': sympy.Symbol('t')}
    result = _parse_formula_using_token_map(formula, token_map)

    if isinstance(result, sympy.Basic):
        if result.free_symbols:
            return result
        result = complex(result)

    if isinstance(result, complex):
        if abs(np.imag(result)) > 1e-8:
            raise ValueError('Not a real result.')
        result = np.real(result)

    return float(cast(SupportsFloat, result))
