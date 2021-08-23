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

import numpy as np
import pytest
import sympy

from google.protobuf import json_format

import cirq
import cirq_google
from cirq_google.serialization.arg_func_langs import (
    arg_from_proto,
    arg_to_proto,
    float_arg_from_proto,
    float_arg_to_proto,
    ARG_LIKE,
    LANGUAGE_ORDER,
)
from cirq_google.api import v2


@pytest.mark.parametrize(
    'min_lang,value,proto',
    [
        ('', 1.0, {'arg_value': {'float_value': 1.0}}),
        ('', 1, {'arg_value': {'float_value': 1.0}}),
        ('', 'abc', {'arg_value': {'string_value': 'abc'}}),
        ('', [True, False], {'arg_value': {'bool_values': {'values': [True, False]}}}),
        ('', sympy.Symbol('x'), {'symbol': 'x'}),
        (
            'linear',
            sympy.Symbol('x') - sympy.Symbol('y'),
            {
                'func': {
                    'type': 'add',
                    'args': [
                        {'symbol': 'x'},
                        {
                            'func': {
                                'type': 'mul',
                                'args': [{'arg_value': {'float_value': -1.0}}, {'symbol': 'y'}],
                            }
                        },
                    ],
                }
            },
        ),
        (
            'exp',
            sympy.Symbol('x') ** sympy.Symbol('y'),
            {'func': {'type': 'pow', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
    ],
)
def test_correspondence(min_lang: str, value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    min_i = LANGUAGE_ORDER.index(min_lang)
    for i, lang in enumerate(LANGUAGE_ORDER):
        if i < min_i:
            with pytest.raises(ValueError, match='not supported by arg_function_language'):
                _ = arg_to_proto(value, arg_function_language=lang)
            with pytest.raises(ValueError, match='Unrecognized function type'):
                _ = arg_from_proto(msg, arg_function_language=lang)
        else:
            parsed = arg_from_proto(msg, arg_function_language=lang)
            packed = json_format.MessageToDict(
                arg_to_proto(value, arg_function_language=lang),
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                use_integers_for_enums=True,
            )

            assert parsed == value
            assert packed == proto


def test_double_value():
    """Note: due to backwards compatibility, double_val conversion is one-way.
    double_val can be converted to python float,
    but a python float is converted into a float_val not a double_val.
    """
    msg = v2.program_pb2.Arg()
    msg.arg_value.double_value = 1.0
    parsed = arg_from_proto(msg, arg_function_language='')
    assert parsed == 1


def test_serialize_sympy_constants():
    proto = arg_to_proto(sympy.pi, arg_function_language='')
    packed = json_format.MessageToDict(
        proto,
        including_default_value_fields=True,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    assert len(packed) == 1
    assert len(packed['arg_value']) == 1
    # protobuf 3.12+ truncates floats to 4 bytes
    assert np.isclose(packed['arg_value']['float_value'], np.float32(sympy.pi), atol=1e-7)


def test_unsupported_function_language():
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_to_proto(
            sympy.Symbol('a') + sympy.Symbol('b'), arg_function_language='NEVER GONNAH APPEN'
        )
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_to_proto(3 * sympy.Symbol('b'), arg_function_language='NEVER GONNAH APPEN')
    with pytest.raises(ValueError, match='Unrecognized arg_function_language'):
        _ = arg_from_proto(
            v2.program_pb2.Arg(
                func=v2.program_pb2.ArgFunction(
                    type='add',
                    args=[v2.program_pb2.Arg(symbol='a'), v2.program_pb2.Arg(symbol='b')],
                )
            ),
            arg_function_language='NEVER GONNAH APPEN',
        )


@pytest.mark.parametrize(
    'value,proto',
    [
        ((True, False), {'arg_value': {'bool_values': {'values': [True, False]}}}),
        (
            np.array([True, False], dtype=bool),
            {'arg_value': {'bool_values': {'values': [True, False]}}},
        ),
    ],
)
def test_serialize_conversion(value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    packed = json_format.MessageToDict(
        arg_to_proto(value, arg_function_language=''),
        including_default_value_fields=True,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    assert packed == proto


def test_infer_language():
    q = cirq.GridQubit(0, 0)
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')

    c_linear = cirq.Circuit(cirq.X(q) ** (b - a))
    packed = cirq_google.XMON.serialize(c_linear)
    assert packed.language.arg_function_language == 'linear'

    c_empty = cirq.Circuit(cirq.X(q) ** b)
    packed = cirq_google.XMON.serialize(c_empty)
    assert packed.language.arg_function_language == ''

    c_exp = cirq.Circuit(cirq.X(q) ** (b ** a))
    packed = cirq_google.XMON.serialize(c_exp)
    assert packed.language.arg_function_language == 'exp'


@pytest.mark.parametrize(
    'value,proto',
    [
        (4, v2.program_pb2.FloatArg(float_value=4.0)),
        (1.0, v2.program_pb2.FloatArg(float_value=1.0)),
        (sympy.Symbol('a'), v2.program_pb2.FloatArg(symbol='a')),
        (
            sympy.Symbol('a') + sympy.Symbol('b'),
            v2.program_pb2.FloatArg(
                func=v2.program_pb2.ArgFunction(
                    type='add',
                    args=[v2.program_pb2.Arg(symbol='a'), v2.program_pb2.Arg(symbol='b')],
                )
            ),
        ),
    ],
)
def test_float_args(value, proto):
    assert float_arg_to_proto(value) == proto
    assert float_arg_from_proto(proto, arg_function_language='exp') == value


def test_missing_required_arg():
    with pytest.raises(ValueError, match='blah is missing'):
        _ = float_arg_from_proto(
            v2.program_pb2.FloatArg(), arg_function_language='exp', required_arg_name='blah'
        )
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = arg_from_proto(
            v2.program_pb2.Arg(), arg_function_language='exp', required_arg_name='blah'
        )
    with pytest.raises(ValueError, match='Unrecognized function type '):
        _ = arg_from_proto(
            v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='magic')),
            arg_function_language='exp',
            required_arg_name='blah',
        )
    assert arg_from_proto(v2.program_pb2.Arg(), arg_function_language='exp') is None


def test_unrecognized_arg():
    """Getting to some parts of the codes imply that the
    set of supported of languages has changed.  Modify the
    supported languages to simulate this future code change."""
    cirq_google.serialization.arg_func_langs.SUPPORTED_FUNCTIONS_FOR_LANGUAGE['test'] = frozenset(
        {'magic'}
    )

    with pytest.raises(ValueError, match='could not be processed'):
        _ = float_arg_from_proto(
            v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='magic')),
            arg_function_language='test',
            required_arg_name='blah',
        )
    # Clean up for hermetic testing
    del cirq_google.serialization.arg_func_langs.SUPPORTED_FUNCTIONS_FOR_LANGUAGE['test']


def test_invalid_float_arg():
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = float_arg_from_proto(
            v2.program_pb2.Arg(arg_value=v2.program_pb2.ArgValue(float_value=0.5)),
            arg_function_language='test',
            required_arg_name='blah',
        )
