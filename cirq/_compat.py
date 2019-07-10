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

"""Workarounds for compatibility issues between versions and libraries."""
import functools
import logging
from typing import Any, Callable, Optional

import numpy as np
import sympy


def proper_repr(value: Any) -> str:
    """Overrides sympy and numpy returning repr strings that don't parse."""

    if isinstance(value, sympy.Basic):
        result = sympy.srepr(value)

        # HACK: work around https://github.com/sympy/sympy/issues/16074
        # (only handles a few cases)
        fixed_tokens = [
            'Symbol', 'pi', 'Mul', 'Add', 'Mod', 'Integer', 'Float', 'Rational'
        ]
        for token in fixed_tokens:
            result = result.replace(token, 'sympy.' + token)

        return result

    if isinstance(value, np.ndarray):
        return 'np.array({!r})'.format(value.tolist())
    return repr(value)


def deprecated(*, deadline: str, fix: str, func_name: Optional[str] = None
              ) -> Callable[[Callable], Callable]:
    """Marks a function as deprecated.

    Args:
        deadline: The version where the function will be deleted (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")

    Returns:
        A decorator that decorates functions with a deprecation warning.
    """

    def decorator(func: Callable) -> Callable:
        used = False

        @functools.wraps(func)
        def decorated_func(*args, **kwargs) -> Any:
            nonlocal used
            if not used:
                used = True
                qualname = (func.__qualname__
                            if func_name is None else func_name)
                logging.warning(
                    'DEPRECATION\n'
                    'The function %s was used but is deprecated.\n'
                    'It will be removed in cirq %s.\n'
                    '%s\n', qualname, deadline, fix)

            return func(*args, **kwargs)

        decorated_func.__doc__ = (
            f'THIS FUNCTION IS DEPRECATED.\n\n'
            f'IT WILL BE REMOVED IN `cirq {deadline}`.\n\n'
            f'{fix}\n\n'
            f'------\n\n'
            f'{decorated_func.__doc__ or ""}')

        return decorated_func

    return decorator
