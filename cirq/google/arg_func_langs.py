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
import math
from typing import (
    List,
    Union,
    Optional,
    Iterator,
    Iterable,
    cast,
    Set,
    Dict,
    FrozenSet,
)

import numpy as np
import sympy
from cirq.google.api import v2

SUPPORTED_FUNCTIONS_FOR_LANGUAGE: Dict[Optional[str], FrozenSet[str]] = {
    '': frozenset(),
    'linear': frozenset({'add', 'mul'}),
    # None means any. Is used when inferring the language during serialization.
    None: frozenset({'add', 'mul'}),
}

SUPPORTED_SYMPY_OPS = (sympy.Symbol, sympy.Add, sympy.Mul)

# Argument types for gates.
ARG_LIKE = Union[int, float, List[bool], str, sympy.Symbol, sympy.Add, sympy.
                 Mul]

# Supported function languages in order from least to most flexible.
# Clients should use the least flexible language they can, to make it easier
# to gradually roll out new capabilities to clients and servers.
LANGUAGE_ORDER = [
    '',
    'linear',
]


def _max_lang(langs: Iterable[str]) -> str:
    i = max((LANGUAGE_ORDER.index(e) for e in langs), default=0)
    return LANGUAGE_ORDER[i]


def _infer_function_language_from_circuit(value: v2.program_pb2.Circuit) -> str:
    return _max_lang({
        e for moment in value.moments for op in moment.operations
        for e in _function_languages_from_operation(op)
    })


def _infer_function_language_from_schedule(value: v2.program_pb2.Schedule
                                          ) -> str:
    return _max_lang({
        e for op in value.scheduled_operations
        for e in _function_languages_from_operation(op.operation)
    })


def _function_languages_from_operation(value: v2.program_pb2.Operation
                                      ) -> Iterator[str]:
    for arg in value.args.values():
        yield from _function_languages_from_arg(arg)


def _function_languages_from_arg(arg_proto: v2.program_pb2.Arg
                                ) -> Iterator[str]:

    which = arg_proto.WhichOneof('arg')
    if which == 'func':
        if arg_proto.func.type in ['add', 'mul']:
            yield 'linear'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)


def _arg_to_proto(value: ARG_LIKE,
                  *,
                  arg_function_language: Optional[str],
                  out: Optional[v2.program_pb2.Arg] = None
                 ) -> v2.program_pb2.Arg:
    """Writes an argument value into an Arg proto.

    Args:
        value: The value to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into as well as the `arg_function_language`
        that was used.
    """

    if arg_function_language not in SUPPORTED_FUNCTIONS_FOR_LANGUAGE:
        raise ValueError(f'Unrecognized arg_function_language: '
                         f'{arg_function_language!r}')
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE[arg_function_language]

    msg = v2.program_pb2.Arg() if out is None else out

    def check_support(func_type: str) -> str:
        if func_type not in supported:
            lang = (repr(arg_function_language)
                    if arg_function_language is not None else '[any]')
            raise ValueError(f'Function type {func_type!r} not supported by '
                             f'arg_function_language {lang}')
        return func_type

    if isinstance(value, (float, int, sympy.Integer, sympy.Float,
                          sympy.Rational, sympy.NumberSymbol)):
        msg.arg_value.float_value = float(value)
    elif isinstance(value, str):
        msg.arg_value.string_value = value
    elif (isinstance(value, (list, tuple, np.ndarray)) and
          all(isinstance(x, (bool, np.bool_)) for x in value)):
        # Some protobuf / numpy combinations do not support np.bool_, so cast.
        msg.arg_value.bool_values.values.extend([bool(x) for x in value])
    elif isinstance(value, sympy.Symbol):
        msg.symbol = str(value.free_symbols.pop())
    elif isinstance(value, sympy.Add):
        msg.func.type = check_support('add')
        for arg in value.args:
            _arg_to_proto(arg,
                          arg_function_language=arg_function_language,
                          out=msg.func.args.add())
    elif isinstance(value, sympy.Mul):
        msg.func.type = check_support('mul')
        for arg in value.args:
            _arg_to_proto(arg,
                          arg_function_language=arg_function_language,
                          out=msg.func.args.add())
    else:
        raise ValueError(f'Unrecognized arg type: {type(value)}')

    return msg


def _arg_from_proto(
        arg_proto: v2.program_pb2.Arg,
        *,
        arg_function_language: str,
        required_arg_name: Optional[str] = None,
) -> Optional[ARG_LIKE]:
    """Extracts a python value from an argument value proto.

    Args:
        arg_proto: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.
        required_arg_name: If set to `None`, the method will return `None` when
            given an unset proto value. If set to a string, the method will
            instead raise an error complaining that the value is missing in that
            situation.

    Returns:
        The deserialized value, or else None if there was no set value and
        `required_arg_name` was set to `None`.
    """
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE.get(arg_function_language)
    if supported is None:
        raise ValueError(f'Unrecognized arg_function_language: '
                         f'{arg_function_language!r}')

    which = arg_proto.WhichOneof('arg')
    if which == 'arg_value':
        arg_value = arg_proto.arg_value
        which_val = arg_value.WhichOneof('arg_value')
        if which_val == 'float_value' or which_val == 'double_value':
            if which_val == 'double_value':
                result = float(arg_value.double_value)
            else:
                result = float(arg_value.float_value)
            if math.ceil(result) == math.floor(result):
                result = int(result)
            return result
        if which_val == 'bool_values':
            return list(arg_value.bool_values.values)
        if which_val == 'string_value':
            return str(arg_value.string_value)
        raise ValueError(f'Unrecognized value type: {which_val!r}')

    if which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)

    if which == 'func':
        func = arg_proto.func

        if func.type not in cast(Set[str], supported):
            raise ValueError(
                f'Unrecognized function type {func.type!r} '
                f'for arg_function_language={arg_function_language!r}')

        if func.type == 'add':
            return sympy.Add(*[
                _arg_from_proto(a,
                                arg_function_language=arg_function_language,
                                required_arg_name='An addition argument')
                for a in func.args
            ])

        if func.type == 'mul':
            return sympy.Mul(*[
                _arg_from_proto(a,
                                arg_function_language=arg_function_language,
                                required_arg_name='A multiplication argument')
                for a in func.args
            ])

    if required_arg_name is not None:
        raise ValueError(
            f'{required_arg_name} is missing or has an unrecognized '
            f'argument type (WhichOneof("arg")={which!r}).')

    return None
