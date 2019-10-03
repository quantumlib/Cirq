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

import re
from typing import Any, Union

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr


def parse_formula(formula: Any) -> Union[float, sympy.Basic]:
    """Attempts to parse formula text in exactly the same way as Quirk."""
    if not isinstance(formula, str):
        raise TypeError('formula must be a string: {!r}'.format(formula))

    try:
        result = _parse_scalar(formula, t=sympy.Symbol('t'))
    except Exception as ex:
        raise ValueError(
            f"Failed to parse the gate formula {repr(formula)}.\n"
            "This is likely due to a bug where cirq fails to exactly emulate "
            "Quirk's parsing. Please report it.") from ex
    assert result.free_symbols <= {sympy.Symbol('t')}
    if not result.free_symbols:
        result = complex(result)
        if abs(np.imag(result)) > 1e-8:
            raise ValueError(
                'Expected a real value formula but got {!r}'.format(formula))
        result = float(np.real(result))
    return result


def parse_matrix(text: str) -> np.ndarray:
    """Attempts to parse a complex matrix in exactly the same way as Quirk."""
    if not text.startswith('{{') or not text.endswith('}}'):
        raise ValueError('No opening/closing braces.\ntext: {!r}'.format(text))
    text = text[2:-2]
    rows = text.split('},{')
    return np.array([[parse_complex(c) for c in row.split(',')] for row in rows
                    ])


def parse_complex(text: str) -> complex:
    """Attempts to parse a complex number in exactly the same way as Quirk."""
    try:
        return complex(_parse_scalar(text))
    except Exception as ex:
        raise ValueError(
            'Failed to parse complex from {!r}'.format(text)) from ex


def _parse_scalar(text: str, **extras) -> sympy.Basic:
    text = _expand_unicode_fractions(text)

    # Convert number suffixing into implicit action.
    text = re.sub(r'([0123456789)])([a-df-zA-DF-Z])', r'\1 \2', text)

    # Canonicalize redundant function names.
    text = text.replace('√', ' sqrt')
    text = text.replace('arcsin', 'asin')
    text = text.replace('arccos', 'acos')
    text = text.replace('arctan', 'atan')

    # Convert implicit function application into explicit application.
    # TODO(craiggidney): support nested implicit function application.
    text = re.sub(r'(a?(cos|sin|tan)|exp|ln|sqrt)\s*([^-\s(+*/^)]+)', r'\1(\3)',
                  text)

    # Convert implicit multiplication into explicit multiplication.
    text = re.sub(r'([^-^+*/])\s([^-^+*/])', r'\1*\2', text)

    # Translate exponentiation notation.
    text = text.replace('^', '**')

    return parse_expr(text,
                      transformations=[],
                      local_dict={},
                      global_dict={
                          'sqrt': sympy.sqrt,
                          'exp': sympy.exp,
                          'ln': sympy.ln,
                          'cos': sympy.cos,
                          'sin': sympy.sin,
                          'tan': sympy.tan,
                          'acos': sympy.acos,
                          'asin': sympy.asin,
                          'atan': sympy.atan,
                          'i': sympy.I,
                          'e': sympy.E,
                          'pi': sympy.pi,
                          **extras,
                      })


def _expand_unicode_fractions(text: str) -> str:
    for k, v in UNICODE_FRACTIONS.items():
        text = text.replace(k, v)
    return text


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
