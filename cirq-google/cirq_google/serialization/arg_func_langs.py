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
from typing import cast, List, Optional, Sequence, Union

import numpy as np
import sympy
import tunits

from cirq.qis import CliffordTableau
from cirq.value import BitMaskKeyCondition, Condition, KeyCondition, MeasurementKey, SympyCondition
from cirq_google.api import v2
from cirq_google.api.v2 import ndarrays
from cirq_google.ops import InternalGate

SUPPORTED_SYMPY_OPS = (sympy.Symbol, sympy.Add, sympy.Mul, sympy.Pow)

# Argument types for gates.
ARG_LIKE = Union[int, float, numbers.Real, Sequence[bool], str, sympy.Expr, tunits.Value]
ARG_RETURN_LIKE = Union[
    float, int, str, List[bool], List[int], List[float], List[str], sympy.Expr, tunits.Value
]
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

# The sympy operations supported by the proto format
# and their corresponding operation types
_SUPPORTED_SYMPY_TYPE_MAPPING = {
    sympy.Add: "add",
    sympy.Mul: "mul",
    sympy.Pow: "pow",
    sympy.Equality: "==",
    sympy.Unequality: "!=",
    sympy.GreaterThan: ">=",
    sympy.StrictGreaterThan: ">",
    sympy.LessThan: "<=",
    sympy.StrictLessThan: "<",
    sympy.And: "&",
    sympy.Or: "|",
    sympy.Xor: "^",
    sympy.Not: "!",
    # Note that "sympy.Indexed": "[]" is handled as a special case.
}
_SYMPY_EXPR_PER_TYPE = {op_type: expr for expr, op_type in _SUPPORTED_SYMPY_TYPE_MAPPING.items()}
_SUPPORTED_SYMPY_TYPES = tuple(_SUPPORTED_SYMPY_TYPE_MAPPING.keys())


def float_arg_to_proto(
    value: ARG_LIKE, *, out: Optional[v2.program_pb2.FloatArg] = None
) -> v2.program_pb2.FloatArg:
    """Writes an argument value into an FloatArg proto.

    Note that the FloatArg proto is a slimmed down form of the
    Arg proto, so this proto should only be used when the argument
    is known to be a float or expression that resolves to a float.

    Args:
        value: The value to encode.  This must be a float or compatible
            sympy expression. Strings and repeated booleans are not allowed.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.
    """
    msg = v2.program_pb2.FloatArg() if out is None else out

    if isinstance(value, FLOAT_TYPES):
        msg.float_value = float(value)
    else:
        _arg_func_to_proto(value, msg)

    return msg


def arg_to_proto(
    value: ARG_LIKE, *, out: Optional[v2.program_pb2.Arg] = None
) -> v2.program_pb2.Arg:
    """Writes an argument value into an Arg proto.

    Args:
        value: The value to encode.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into.

    Raises:
        ValueError: if the object holds unsupported values.
    """
    msg = v2.program_pb2.Arg() if out is None else out

    if isinstance(value, (bool, np.bool_)):
        msg.arg_value.bool_value = bool(value)
    elif isinstance(value, FLOAT_TYPES):
        msg.arg_value.float_value = float(value)
    elif isinstance(value, complex):
        msg.arg_value.complex_value.real_value = value.real
        msg.arg_value.complex_value.imag_value = value.imag
    elif isinstance(value, bytes):
        msg.arg_value.bytes_value = value
    elif isinstance(value, str):
        msg.arg_value.string_value = value
    elif isinstance(value, (list, tuple, np.ndarray, set, frozenset)):
        if isinstance(value, np.ndarray):
            _ndarray_to_proto(value, out=msg)
        elif len(value) == 0:
            # Convert empty list or tuple
            _tuple_to_proto(value, out=msg.arg_value.tuple_value)
        elif isinstance(value, list) and isinstance(value[0], str):
            # Note that we should not convert a tuple to a list here
            # in order to preserve types
            if not all(isinstance(x, str) for x in value):
                # Not a uniform list, convert to tuple
                _tuple_to_proto(value, out=msg.arg_value.tuple_value)
                return msg
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
                # Not a uniform list, convert to tuple
                _tuple_to_proto(value, out=msg.arg_value.tuple_value)
                return msg
            field, types_tuple = numerical_fields[cur_index]
            field.extend(types_tuple[0](x) for x in value)
    elif isinstance(value, tunits.Value):
        msg.arg_value.value_with_unit.MergeFrom(value.to_proto())
    elif isinstance(value, MeasurementKey):
        msg.measurement_key.string_key = value.name
        msg.measurement_key.path.extend(value.path)
    else:
        _arg_func_to_proto(value, msg)

    return msg


def _ndarray_to_proto(value: np.ndarray, out: v2.program_pb2.Arg):
    ndarray_msg = out.arg_value.ndarray_value
    match value.dtype:
        case np.float64:
            ndarrays.to_float64_array(value, out=ndarray_msg.float64_array)
        case np.float32:
            ndarrays.to_float32_array(value, out=ndarray_msg.float32_array)
        case np.float16:
            ndarrays.to_float16_array(value, out=ndarray_msg.float16_array)
        case np.int64:
            ndarrays.to_int64_array(value, out=ndarray_msg.int64_array)
        case np.int32:
            ndarrays.to_int32_array(value, out=ndarray_msg.int32_array)
        case np.int16:
            ndarrays.to_int16_array(value, out=ndarray_msg.int16_array)
        case np.int8:
            ndarrays.to_int8_array(value, out=ndarray_msg.int8_array)
        case np.uint8:
            ndarrays.to_uint8_array(value, out=ndarray_msg.uint8_array)
        case np.complex128:
            ndarrays.to_complex128_array(value, out=ndarray_msg.complex128_array)
        case np.complex64:
            ndarrays.to_complex64_array(value, out=ndarray_msg.complex64_array)
        case np.bool_:
            ndarrays.to_bitarray(value, out=ndarray_msg.bit_array)


def _ndarray_from_proto(msg: v2.program_pb2.ArgValue):
    ndarray_msg = msg.ndarray_value
    match ndarray_msg.WhichOneof('arr'):
        case 'float64_array':
            return ndarrays.from_float64_array(ndarray_msg.float64_array)
        case 'float32_array':
            return ndarrays.from_float32_array(ndarray_msg.float32_array)
        case 'float16_array':
            return ndarrays.from_float16_array(ndarray_msg.float16_array)
        case 'int64_array':
            return ndarrays.from_int64_array(ndarray_msg.int64_array)
        case 'int32_array':
            return ndarrays.from_int32_array(ndarray_msg.int32_array)
        case 'int16_array':
            return ndarrays.from_int16_array(ndarray_msg.int16_array)
        case 'int8_array':
            return ndarrays.from_int8_array(ndarray_msg.int8_array)
        case 'uint8_array':
            return ndarrays.from_uint8_array(ndarray_msg.uint8_array)
        case 'complex128_array':
            return ndarrays.from_complex128_array(ndarray_msg.complex128_array)
        case 'complex64_array':
            return ndarrays.from_complex64_array(ndarray_msg.complex64_array)
        case 'bit_array':
            return ndarrays.from_bitarray(ndarray_msg.bit_array)


def _tuple_to_proto(value: Union[list, tuple, set, frozenset], out: v2.program_pb2.Tuple):
    """Converts a tuple of mixed values to Arg protos."""
    if isinstance(value, list):
        out.sequence_type = v2.program_pb2.Tuple.SequenceType.LIST
    elif isinstance(value, tuple):
        out.sequence_type = v2.program_pb2.Tuple.SequenceType.TUPLE
    elif isinstance(value, set):
        out.sequence_type = v2.program_pb2.Tuple.SequenceType.SET
    elif isinstance(value, frozenset):
        out.sequence_type = v2.program_pb2.Tuple.SequenceType.FROZENSET
    else:
        out.sequence_type = v2.program_pb2.Tuple.SequenceType.UNSPECIFIED  # pragma: nocover
    for arg in value:
        new_arg = out.values.add()
        arg_to_proto(arg, out=new_arg)


def _arg_func_to_proto(
    value: ARG_LIKE, msg: Union[v2.program_pb2.Arg, v2.program_pb2.FloatArg]
) -> None:
    if isinstance(value, sympy.Symbol):
        msg.symbol = str(value.free_symbols.pop())
    elif isinstance(value, _SUPPORTED_SYMPY_TYPES):
        msg.func.type = _SUPPORTED_SYMPY_TYPE_MAPPING[type(value)]
        for arg in value.args:
            arg_to_proto(arg, out=msg.func.args.add())
    elif isinstance(value, sympy.Indexed):
        # Sympy version of M[a, b]
        msg.func.type = "[]"
        arg_to_proto(value.base.label, out=msg.func.args.add())
        for arg in value.indices:
            arg_to_proto(arg, out=msg.func.args.add())
    else:
        raise ValueError(
            f"Unrecognized Sympy expression type: {type(value)}."
            " Only the following types are recognized: 'sympy.Symbol', 'sympy.Add', 'sympy.Mul',"
            " 'sympy.Pow', and sympy comparison types."
        )


def float_arg_from_proto(
    arg_proto: v2.program_pb2.FloatArg, *, required_arg_name: Optional[str] = None
) -> Optional[FLOAT_ARG_LIKE]:
    """Extracts a python value from an argument value proto.

    This function handles `FloatArg` protos, that are required
    to be floats or symbolic expressions.

    Args:
        arg_proto: The proto containing a serialized value.
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
    match which:
        case 'float_value':
            result = float(arg_proto.float_value)
            if round(result) == result:
                result = int(result)
            return result
        case 'symbol':
            return sympy.Symbol(arg_proto.symbol)
        case 'func':
            func = _arg_func_from_proto(arg_proto.func, required_arg_name=required_arg_name)
            if func is None and required_arg_name is not None:
                raise ValueError(  # pragma: no cover
                    f'Arg {arg_proto.func} could not be processed for {required_arg_name}.'
                )
            return cast(FLOAT_ARG_LIKE, func)
        case None:
            if required_arg_name is not None:
                raise ValueError(f'Arg {required_arg_name} is missing.')
            return None
    raise ValueError(f'unrecognized argument type ({which}).')


def arg_from_proto(
    arg_proto: v2.program_pb2.Arg, *, required_arg_name: Optional[str] = None
) -> Optional[ARG_RETURN_LIKE]:
    """Extracts a python value from an argument value proto.

    Args:
        arg_proto: The proto containing a serialized value.
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
    match which:
        case 'arg_value':
            arg_value = arg_proto.arg_value
            which_val = arg_value.WhichOneof('arg_value')
            match which_val:
                case 'float_value':
                    result = float(arg_value.float_value)
                    if math.ceil(result) == math.floor(result):
                        return int(result)
                    return result
                case 'double_value':
                    result = float(arg_value.double_value)
                    if math.ceil(result) == math.floor(result):
                        return int(result)
                    return result
                case 'bool_value':
                    return bool(arg_value.bool_value)
                case 'bool_values':
                    return list(arg_value.bool_values.values)
                case 'string_value':
                    return str(arg_value.string_value)
                case 'int64_values':
                    return [int(v) for v in arg_value.int64_values.values]
                case 'double_values':
                    return [float(v) for v in arg_value.double_values.values]
                case 'string_values':
                    return [str(v) for v in arg_value.string_values.values]
                case 'value_with_unit':
                    return tunits.Value.from_proto(arg_value.value_with_unit)
                case 'bytes_value':
                    return bytes(arg_value.bytes_value)
                case 'complex_value':
                    return complex(
                        arg_value.complex_value.real_value, arg_value.complex_value.imag_value
                    )
                case 'tuple_value':
                    values = (
                        arg_from_proto(tuple_proto) for tuple_proto in arg_value.tuple_value.values
                    )
                    sequence_type = arg_value.tuple_value.sequence_type
                    match sequence_type:
                        case v2.program_pb2.Tuple.SequenceType.LIST:
                            return list(values)
                        case v2.program_pb2.Tuple.SequenceType.TUPLE:
                            return tuple(values)
                        case v2.program_pb2.Tuple.SequenceType.SET:
                            return set(values)
                        case v2.program_pb2.Tuple.SequenceType.FROZENSET:
                            return frozenset(values)
                    raise ValueError('Unrecognized type: {sequence_type}')  # pragma: no cover

                case 'ndarray_value':
                    return _ndarray_from_proto(arg_value)
            raise ValueError(f'Unrecognized value type: {which_val!r}')  # pragma: no cover
        case 'symbol':
            return sympy.Symbol(arg_proto.symbol)
        case 'func':
            func = _arg_func_from_proto(arg_proto.func, required_arg_name=required_arg_name)
            if func is not None:
                return func
        case 'measurement_key':
            return MeasurementKey(
                name=arg_proto.measurement_key.string_key,
                path=tuple(arg_proto.measurement_key.path),
            )

    if required_arg_name is not None:
        raise ValueError(
            f'{required_arg_name} is missing or has an unrecognized '
            f'argument type (WhichOneof("arg")={which!r}).'
        )

    return None


def _arg_func_from_proto(
    func: v2.program_pb2.ArgFunction, *, required_arg_name: Optional[str] = None
) -> Optional[ARG_RETURN_LIKE]:

    if (op_expr := _SYMPY_EXPR_PER_TYPE.get(func.type, None)) is not None:
        return op_expr(
            *[arg_from_proto(a, required_arg_name=f'An {func.type} argument') for a in func.args]
        )
    if func.type == "[]":
        # Handle sympy.Indexed i.e. M[a, b] as a special case.
        base = sympy.IndexedBase(arg_from_proto(func.args[0]))
        args = [arg_from_proto(a) for a in func.args[1:]]
        return sympy.Indexed(base, *args)
    raise ValueError(f'Unrecognized sympy function {func}')


def condition_to_proto(control: Condition, *, out: v2.program_pb2.Arg) -> v2.program_pb2.Arg:
    if isinstance(control, KeyCondition):
        out.measurement_key.string_key = control.key.name
        out.measurement_key.path.extend(control.key.path)
        out.measurement_key.index = control.index
    elif isinstance(control, SympyCondition):
        arg_to_proto(control.expr, out=out)
    elif isinstance(control, BitMaskKeyCondition):
        if control.equal_target:
            # Special function that represents (a & c) == b
            out.func.type = "bitmask=="
        else:
            # Special function that represents (a & c) != b
            out.func.type = "bitmask!="
        key_proto = out.func.args.add().measurement_key
        key_proto.string_key = control.key.name
        key_proto.path.extend(control.key.path)
        key_proto.index = control.index
        target_proto = out.func.args.add()
        arg_to_proto(control.target_value, out=target_proto)

        if control.bitmask is not None:
            bitmask_proto = out.func.args.add()
            arg_to_proto(control.bitmask, out=bitmask_proto)
    return out


def condition_from_proto(condition: v2.program_pb2.Arg) -> Condition:
    which = condition.WhichOneof("arg")
    if which == 'measurement_key':
        key = condition.measurement_key
        return KeyCondition(
            key=MeasurementKey(key.string_key, path=tuple(key.path)), index=key.index
        )
    elif which == 'func':
        if condition.func.type == "bitmask==" or condition.func.type == "bitmask!=":
            key = condition.func.args[0].measurement_key
            if len(condition.func.args) > 2:
                bitmask = int(arg_from_proto(condition.func.args[2]))  # type: ignore
            else:
                bitmask = None
            return BitMaskKeyCondition(
                key=MeasurementKey(key.string_key, path=tuple(key.path)),
                index=key.index,
                target_value=int(arg_from_proto(condition.func.args[1])),  # type: ignore
                equal_target=(condition.func.type == "bitmask=="),
                bitmask=bitmask,
            )
        else:
            expr = arg_from_proto(condition)
            return SympyCondition(expr)
    else:
        raise ValueError(f'Unrecognized condition {condition}')  # pragma: nocover


def internal_gate_arg_to_proto(
    value: InternalGate, *, out: Optional[v2.program_pb2.InternalGate] = None
):
    """Writes an InternalGate object into an InternalGate proto.

    Args:
        value: The gate to encode.
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

    for ck, cv in value.custom_args.items():
        msg.custom_args[ck].MergeFrom(cv)

    return msg


def internal_gate_from_proto(msg: v2.program_pb2.InternalGate) -> InternalGate:
    """Extracts an InternalGate object from an InternalGate proto.

    Args:
        msg: The proto containing a serialized value.

    Returns:
        The deserialized InternalGate object.

    Raises:
        ValueError: On failure to parse any of the gate arguments.
    """
    gate_args = {}
    for k, v in msg.gate_args.items():
        gate_args[k] = arg_from_proto(v)
    return InternalGate(
        gate_name=str(msg.name),
        gate_module=str(msg.module),
        num_qubits=int(msg.num_qubits),
        custom_args=msg.custom_args,
        **gate_args,
    )


def clifford_tableau_arg_to_proto(
    value: CliffordTableau, *, out: Optional[v2.program_pb2.CliffordTableau] = None
):
    """Writes an CliffordTableau object into an CliffordTableau proto.
    Args:
        value: The gate to encode.
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


def clifford_tableau_from_proto(msg: v2.program_pb2.CliffordTableau) -> CliffordTableau:
    """Extracts a CliffordTableau object from a CliffordTableau proto.
    Args:
        msg: The proto containing a serialized value.
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
