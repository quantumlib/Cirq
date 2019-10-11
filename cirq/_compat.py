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
from typing import Any, Callable, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import sympy


def proper_repr(value: Any) -> str:
    """Overrides sympy and numpy returning repr strings that don't parse."""

    if isinstance(value, sympy.Basic):
        result = sympy.srepr(value)

        # HACK: work around https://github.com/sympy/sympy/issues/16074
        # (only handles a few cases)
        fixed_tokens = [
            'Symbol', 'pi', 'Mul', 'Pow', 'Add', 'Mod', 'Integer', 'Float',
            'Rational'
        ]
        for token in fixed_tokens:
            result = result.replace(token, 'sympy.' + token)

        return result

    if isinstance(value, np.ndarray):
        return 'np.array({!r}, dtype=np.{})'.format(value.tolist(), value.dtype)

    if isinstance(value, pd.MultiIndex):
        return (f'pd.MultiIndex.from_tuples({repr(list(value))}, '
                f'names={repr(list(value.names))})')

    if isinstance(value, pd.Index):
        return (f'pd.Index({repr(list(value))}, '
                f'name={repr(value.name)}, '
                f'dtype={repr(str(value.dtype))})')

    if isinstance(value, pd.DataFrame):
        cols = [value[col].tolist() for col in value.columns]
        rows = list(zip(*cols))
        return (f'pd.DataFrame('
                f'\n    columns={proper_repr(value.columns)}, '
                f'\n    index={proper_repr(value.index)}, '
                f'\n    data={repr(rows)}'
                f'\n)')

    return repr(value)


def deprecated(*, deadline: str, fix: str, func_name: Optional[str] = None
              ) -> Callable[[Callable], Callable]:
    """Marks a function as deprecated.

    Args:
        deadline: The version where the function will be deleted (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        func_name: How to refer to the function.
            Defaults to `func.__qualname__`.

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
            f'{decorated_func.__doc__ or ""}')

        return decorated_func

    return decorator


def deprecated_parameter(
        *,
        deadline: str,
        fix: str,
        func_name: Optional[str] = None,
        parameter_desc: str,
        match: Callable[[Tuple[Any, ...], Dict[str, Any]], bool],
        rewrite: Optional[
            Callable[[Tuple[Any, ...], Dict[str, Any]],
                     Tuple[Tuple[Any, ...], Dict[str, Any]]]] = None,
) -> Callable[[Callable], Callable]:
    """Marks a function parameter as deprecated.

    Also handles rewriting the deprecated parameter into the new signature.

    Args:
        deadline: The version where the parameter will be deleted (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        func_name: How to refer to the function.
            Defaults to `func.__qualname__`.
        parameter_desc: The name and type of the parameter being deprecated,
            e.g. "janky_count" or "janky_count keyword" or
            "positional janky_count".
        match: A lambda that takes args, kwargs and determines if the
            deprecated parameter is present or not. This determines whether or
            not the deprecation warning is printed, and also whether or not
            rewrite is called.
        rewrite: Returns new args/kwargs that don't use the deprecated
            parameter. Defaults to making no changes.

    Returns:
        A decorator that decorates functions with a parameter deprecation
            warning.
    """

    def decorator(func: Callable) -> Callable:
        used = False

        @functools.wraps(func)
        def decorated_func(*args, **kwargs) -> Any:
            nonlocal used
            if match(args, kwargs):
                if rewrite is not None:
                    args, kwargs = rewrite(args, kwargs)

                if not used:
                    used = True
                    qualname = (func.__qualname__
                                if func_name is None else func_name)
                    logging.warning(
                        'DEPRECATION\n'
                        f'The %s parameter of %s was used but is deprecated.\n'
                        'It will be removed in cirq %s.\n'
                        '%s\n', parameter_desc, qualname, deadline, fix)

            return func(*args, **kwargs)

        return decorated_func

    return decorator
