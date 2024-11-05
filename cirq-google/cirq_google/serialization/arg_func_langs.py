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
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
from cirq.qis import CliffordTableau

SUPPORTED_FUNCTIONS_FOR_LANGUAGE: Dict[Optional[str], FrozenSet[str]] = {
    '': frozenset(),
    'linear': frozenset({'add', 'mul'}),
    'exp': frozenset({'add', 'mul', 'pow'}),
    # None means any. Is used when inferring the language during serialization.
    None: frozenset({'add', 'mul', 'pow'}),
}
MOST_PERMISSIVE_LANGUAGE = 'exp'

SUPPORTED_SYMPY_OPS = (sympy.Symbol, sympy.Add, sympy.Mul, sympy.Pow)

# Argument types for gates.
ARG_LIKE = Union[int, float, numbers.Real, Sequence[bool], str, sympy.Expr]
ARG_RETURN_LIKE = Union[float, int, str, List[bool], List[int], List[float], List[str], sympy.Expr]
FLOAT_ARG_LIKE = Union[float, sympy.Expr]

# Types for comparing floats
# Includes sympy types.  Needed for arg parsing.
FLOAT_TYPES = (
    float,
    int,
    np.integer,
    np.floating,
    sympy.Integer,
    sympy.Float,
    sympy.Rational,
    sympy.NumberSymbol,
)

# Supported function languages in order from least to most flexible.
# Clients should use the least flexible language they can, to make it easier
# to gradually roll out new capabilities to clients and servers.
LANGUAGE_ORDER = ['', 'linear', 'exp']


def _max_lang(langs: Iterable[str]) -> str:
    i = max((LANGUAGE_ORDER.index(e) for e in langs), default=0)
    return LANGUAGE_ORDER[i]


def _infer_function_language_from_circuit(value: v2.program_pb2.Circuit) -> str:
    return _max_lang(
        {
            e
            for moment in value.moments
            for op in moment.operations
            for e in _function_languages_from_operation(op)
        }
    )


def _function_languages_from_operation(value: v2.program_pb2.Operation) -> Iterator[str]:
    for arg in value.args.values():
        yield from _function_languages_from_arg(arg)


def _function_languages_from_arg(arg_proto: v2.program_pb2.Arg) -> Iterator[str]:
    which = arg_proto.WhichOneof('arg')  # pragma: no cover
    if which == 'func':
        if arg_proto.func.type in ['add', 'mul']:
            yield 'linear'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)
        if arg_proto.func.type in ['pow']:
            yield 'exp'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)


def float_arg_to_proto(
    value: ARG_LIKE,
    *,
    arg_function_language: Optional[str] = None,
    out: Optional[v2.program_pb2.FloatArg] = None,
) -> v2.program_pb2.FloatArg:
    """Writes an argument value into an FloatArg proto.

    Note that the FloatArg proto is a slimmed down form of the
    Arg proto, so this proto should only be used when the argument
    is known to be a float or expression that resolves to a float.

    Args:
        value: The value to encode.  This must be a float or compatible
            sympy expression. Strings and repeated booleans are not allowed.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.FloatArg() if out is None else out

    if isinstance(value, FLOAT_TYPES):
        msg.float_value = float(value)
    else:
        _arg_func_to_proto(value, arg_function_language, msg)

    return msg


def arg_to_proto(
    value: ARG_LIKE,
    *,
    arg_function_language: Optional[str] = None,
    out: Optional[v2.program_pb2.Arg] = None,
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

    Raises:
        ValueError: if the object holds unsupported values.
    """
    msg = v2.program_pb2.Arg() if out is None else out

    if isinstance(value, FLOAT_TYPES):
        msg.arg_value.float_value = float(value)
    elif isinstance(value, str):
        msg.arg_value.string_value = value
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value):
            if isinstance(value[0], str):
                if not all(isinstance(x, str) for x in value):
                    raise ValueError('Sequences of mixed object types are not supported')
                msg.arg_value.string_values.values.extend(str(x) for x in value)
            else:
                # This is a numerical field.
                numerical_fields = [
                    [msg.arg_value.bool_values.values, (bool, np.bool_)],
                    [msg.arg_value.int64_values.values, (int, np.integer, bool)],
                    [msg.arg_value.double_values.values, (float, np.floating, int, bool)],
                ]
                cur_index = 0
                non_numerical = None
                for v in value:
                    while cur_index < len(numerical_fields) and not isinstance(
                        v, numerical_fields[cur_index][1]
                    ):
                        cur_index += 1
                    if cur_index == len(numerical_fields):
                        non_numerical = v
                        break

                if non_numerical is not None:
                    raise ValueError(
                        'Mixed Sequences with objects of type '
                        f'{type(non_numerical)} are not supported'
                    )
                field, types_tuple = numerical_fields[cur_index]
                field.extend(types_tuple[0](x) for x in value)
    else:
        _arg_func_to_proto(value, arg_function_language, msg)

    return msg


def _arg_func_to_proto(
    value: ARG_LIKE,
    arg_function_language: Optional[str],
    msg: Union[v2.program_pb2.Arg, v2.program_pb2.FloatArg],
) -> None:
    def check_support(func_type: str) -> str:
        if func_type not in supported:
            lang = repr(arg_function_language) if arg_function_language is not None else '[any]'
            raise ValueError(
                f'Function type {func_type!r} not supported by arg_function_language {lang}'
            )
        return func_type

    if arg_function_language not in SUPPORTED_FUNCTIONS_FOR_LANGUAGE:
        raise ValueError(f'Unrecognized arg_function_language: {arg_function_language!r}')
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE[arg_function_language]

    if isinstance(value, sympy.Symbol):
        msg.symbol = str(value.free_symbols.pop())
    elif isinstance(value, sympy.Add):
        msg.func.type = check_support('add')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    elif isinstance(value, sympy.Mul):
        msg.func.type = check_support('mul')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    elif isinstance(value, sympy.Pow):
        msg.func.type = check_support('pow')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    else:
        raise ValueError(
            f"Unrecognized Sympy expression type: {type(value)}."
            " Only the following types are recognized: 'sympy.Symbol', 'sympy.Add', 'sympy.Mul',"
            " 'sympy.Pow'."
        )


def float_arg_from_proto(
    arg_proto: v2.program_pb2.FloatArg,
    *,
    arg_function_language: str,
    required_arg_name: Optional[str] = None,
) -> Optional[FLOAT_ARG_LIKE]:
    """Extracts a python value from an argument value proto.

    This function handles `FloatArg` protos, that are required
    to be floats or symbolic expressions.

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

    Raises:
        ValueError: If the float arg proto is invalid.
    """
    which = arg_proto.WhichOneof('arg')
    if which == 'float_value':
        result = float(arg_proto.float_value)
        if round(result) == result:
            result = int(result)
        return result
    elif which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)
    elif which == 'func':
        func = _arg_func_from_proto(
            arg_proto.func,
            arg_function_language=arg_function_language,
            required_arg_name=required_arg_name,
        )
        if func is None and required_arg_name is not None:
            raise ValueError(
                f'Arg {arg_proto.func} could not be processed for {required_arg_name}.'
            )
        return cast(FLOAT_ARG_LIKE, func)
    elif which is None:
        if required_arg_name is not None:
            raise ValueError(f'Arg {required_arg_name} is missing.')
        return None
    else:
        raise ValueError(f'unrecognized argument type ({which}).')


def arg_from_proto(
    arg_proto: v2.program_pb2.Arg,
    *,
    arg_function_language: str,
    required_arg_name: Optional[str] = None,
) -> Optional[ARG_RETURN_LIKE]:
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

    Raises:
        ValueError: If the arg protohas a value of an unrecognized type or is
            missing a required arg name.
    """

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
        if which_val == 'int64_values':
            return [int(v) for v in arg_value.int64_values.values]
        if which_val == 'double_values':
            return [float(v) for v in arg_value.double_values.values]
        if which_val == 'string_values':
            return [str(v) for v in arg_value.string_values.values]
        raise ValueError(f'Unrecognized value type: {which_val!r}')

    if which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)

    if which == 'func':
        func = _arg_func_from_proto(
            arg_proto.func,
            arg_function_language=arg_function_language,
            required_arg_name=required_arg_name,
        )
        if func is not None:
            return func

    if required_arg_name is not None:
        raise ValueError(
            f'{required_arg_name} is missing or has an unrecognized '
            f'argument type (WhichOneof("arg")={which!r}).'
        )

    return None


def _arg_func_from_proto(
    func: v2.program_pb2.ArgFunction,
    *,
    arg_function_language: str,
    required_arg_name: Optional[str] = None,
) -> Optional[ARG_RETURN_LIKE]:
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE.get(arg_function_language)
    if supported is None:
        raise ValueError(f'Unrecognized arg_function_language: {arg_function_language!r}')

    if func.type not in supported:
        raise ValueError(
            f'Unrecognized function type {func.type!r} '
            f'for arg_function_language={arg_function_language!r}'
        )

    if func.type == 'add':
        return sympy.Add(
            *[
                arg_from_proto(
                    a,
                    arg_function_language=arg_function_language,
                    required_arg_name='An addition argument',
                )
                for a in func.args
            ]
        )

    if func.type == 'mul':
        return sympy.Mul(
            *[
                arg_from_proto(
                    a,
                    arg_function_language=arg_function_language,
                    required_arg_name='A multiplication argument',
                )
                for a in func.args
            ]
        )

    if func.type == 'pow':
        return sympy.Pow(
            *[
                arg_from_proto(
                    a,
                    arg_function_language=arg_function_language,
                    required_arg_name='A power argument',
                )
                for a in func.args
            ]
        )
    return None


def internal_gate_arg_to_proto(
    value: InternalGate, *, out: Optional[v2.program_pb2.InternalGate] = None
):
    """Writes an InternalGate object into an InternalGate proto.

    Args:
        value: The gate to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.InternalGate() if out is None else out
    msg.name = value.gate_name
    msg.module = value.gate_module
    msg.num_qubits = value.num_qubits()
    for k, v in value.gate_args.items():
        arg_to_proto(value=v, out=msg.gate_args[k])

    return msg


def internal_gate_from_proto(
    msg: v2.program_pb2.InternalGate, arg_function_language: str
) -> InternalGate:
    """Extracts an InternalGate object from an InternalGate proto.

    Args:
        msg: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.

    Returns:
        The deserialized InternalGate object.

    Raises:
        ValueError: On failure to parse any of the gate arguments.
    """
    gate_args = {}
    for k, v in msg.gate_args.items():
        gate_args[k] = arg_from_proto(v, arg_function_language=arg_function_language)
    return InternalGate(
        gate_name=str(msg.name),
        gate_module=str(msg.module),
        num_qubits=int(msg.num_qubits),
        **gate_args,
    )


def clifford_tableau_arg_to_proto(
    value: CliffordTableau, *, out: Optional[v2.program_pb2.CliffordTableau] = None
):
    """Writes an CliffordTableau object into an CliffordTableau proto.
    Args:
        value: The gate to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.
    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.CliffordTableau() if out is None else out
    msg.num_qubits = value.n
    msg.initial_state = value.initial_state
    msg.xs.extend(map(bool, value.xs.flatten()))
    msg.rs.extend(map(bool, value.rs.flatten()))
    msg.zs.extend(map(bool, value.zs.flatten()))
    return msg


def clifford_tableau_from_proto(
    msg: v2.program_pb2.CliffordTableau, arg_function_language: str
) -> CliffordTableau:
    """Extracts a CliffordTableau object from a CliffordTableau proto.
    Args:
        msg: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.
    Returns:
        The deserialized InternalGate object.
    """
    return CliffordTableau(
        num_qubits=msg.num_qubits,
        initial_state=msg.initial_state,
        rs=np.array(msg.rs, dtype=bool) if msg.rs else None,
        xs=np.array(msg.xs, dtype=bool).reshape((2 * msg.num_qubits, -1)) if msg.xs else None,
        zs=np.array(msg.zs, dtype=bool).reshape((2 * msg.num_qubits, -1)) if msg.zs else None,
    )
