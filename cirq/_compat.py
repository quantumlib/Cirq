# Copyright 2018 The Cirq Developers
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

"""Workarounds for differences between version 2 and 3 of python."""

import sys


# Python 3 deprecates `fractions.gcd` in factor of `math.gcd`. Also, `math.gcd`
# never returns a negative result whereas `fractions.gcd` seems to match the
# sign of the second argument.

# coverage: ignore
from typing import Any

import sympy
import numpy as np


if sys.version_info < (3,):

    import fractions   # pylint: disable=unused-import
    import re

    def gcd(a, b):
        return abs(fractions.gcd(a, b))


    def proper_repr(value: Any) -> str:
        """Overrides sympy and numpy returning repr strings that don't parse."""

        if isinstance(value, sympy.Basic):
            result = sympy.srepr(value)

            # HACK: work around https://github.com/sympy/sympy/issues/16074
            # (only handles a few cases)
            fixed_tokens = ['Symbol',
                            'pi',
                            'Mul',
                            'Add',
                            'Integer',
                            'Float',
                            'Rational']
            for token in fixed_tokens:
                result = result.replace(token, 'sympy.' + token)

            # HACK: work around https://github.com/sympy/sympy/issues/16086
            result = re.sub(
                r'sympy.Symbol\(([^u][^)]*)\)', r'sympy.Symbol(u"\1")', result)
            return result

        if isinstance(value, np.ndarray):
            return 'np.array({!r})'.format(value.tolist())

        return repr(value)

else:

    from math import gcd  # pylint: disable=unused-import

    def proper_repr(value: Any) -> str:
        """Overrides sympy and numpy returning repr strings that don't parse."""

        if isinstance(value, sympy.Basic):
            result = sympy.srepr(value)

            # HACK: work around https://github.com/sympy/sympy/issues/16074
            # (only handles a few cases)
            fixed_tokens = ['Symbol',
                            'pi',
                            'Mul',
                            'Add',
                            'Integer',
                            'Float',
                            'Rational']
            for token in fixed_tokens:
                result = result.replace(token, 'sympy.' + token)

            return result

        if isinstance(value, np.ndarray):
            return 'np.array({!r})'.format(value.tolist())

        return repr(value)
