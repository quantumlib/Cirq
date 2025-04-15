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

import base64
import inspect
from typing import cast, Dict

import numpy as np
import pytest
import sympy
import tunits.units
from google.protobuf import json_format

import cirq_google
from cirq.qis import CliffordTableau
from cirq.value import BitMaskKeyCondition, KeyCondition, MeasurementKey, SympyCondition
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import (
    arg_from_proto,
    ARG_LIKE,
    arg_to_proto,
    clifford_tableau_arg_to_proto,
    clifford_tableau_from_proto,
    condition_from_proto,
    condition_to_proto,
    float_arg_from_proto,
    float_arg_to_proto,
    internal_gate_arg_to_proto,
    internal_gate_from_proto,
)


def _json_format_kwargs() -> Dict[str, bool]:
    """Determine kwargs to pass to json_format.MessageToDict.

    Protobuf v5 has a different signature for MessageToDict. If we ever move to requiring
    protobuf >= 5 this can be removed.
    """
    sig = inspect.signature(json_format.MessageToDict)
    new_arg = "always_print_fields_with_no_presence"
    old_arg = "including_default_value_fields"
    arg = new_arg if new_arg in sig.parameters else old_arg
    return {arg: True}


@pytest.mark.parametrize(
    'value,proto',
    [
        (1.0, {'arg_value': {'float_value': 1.0}}),
        (1.5, {'arg_value': {'float_value': 1.5}}),
        (1, {'arg_value': {'float_value': 1.0}}),
        (1 + 2j, {'arg_value': {'complex_value': {'real_value': 1.0, 'imag_value': 2.0}}}),
        (b'abcdef', {'arg_value': {'bytes_value': base64.b64encode(b'abcdef').decode("ascii")}}),
        ('abc', {'arg_value': {'string_value': 'abc'}}),
        (True, {'arg_value': {'bool_value': True}}),
        ([True, False], {'arg_value': {'bool_values': {'values': [True, False]}}}),
        ([42.9, 3.14], {'arg_value': {'double_values': {'values': [42.9, 3.14]}}}),
        ([3, 8], {'arg_value': {'int64_values': {'values': ['3', '8']}}}),
        (
            ['a', 3],
            {
                'arg_value': {
                    'tuple_value': {
                        'sequence_type': 1,
                        'values': [
                            {'arg_value': {'string_value': 'a'}},
                            {'arg_value': {'float_value': 3.0}},
                        ],
                    }
                }
            },
        ),
        (
            ('settings', 3.5, (True, 3 + 4.5j)),
            {
                'arg_value': {
                    'tuple_value': {
                        'sequence_type': 2,
                        'values': [
                            {'arg_value': {'string_value': 'settings'}},
                            {'arg_value': {'float_value': 3.5}},
                            {
                                'arg_value': {
                                    'tuple_value': {
                                        'sequence_type': 2,
                                        'values': [
                                            {'arg_value': {'bool_value': True}},
                                            {
                                                'arg_value': {
                                                    'complex_value': {
                                                        'real_value': 3.0,
                                                        'imag_value': 4.5,
                                                    }
                                                }
                                            },
                                        ],
                                    }
                                }
                            },
                        ],
                    }
                }
            },
        ),
        (['t1', 't2'], {'arg_value': {'string_values': {'values': ['t1', 't2']}}}),
        (sympy.Symbol('x'), {'symbol': 'x'}),
        (
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
            sympy.Symbol('x') ** sympy.Symbol('y'),
            {'func': {'type': 'pow', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Symbol('x') > sympy.Symbol('y'),
            {'func': {'type': '>', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Symbol('x') >= sympy.Symbol('y'),
            {'func': {'type': '>=', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Symbol('x') < sympy.Symbol('y'),
            {'func': {'type': '<', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Symbol('x') <= sympy.Symbol('y'),
            {'func': {'type': '<=', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Eq(sympy.Symbol('x'), sympy.Symbol('y')),
            {'func': {'type': '==', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Or(sympy.Symbol('x'), sympy.Symbol('y')),
            {'func': {'type': '|', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.And(sympy.Symbol('x'), sympy.Symbol('y')),
            {'func': {'type': '&', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            sympy.Xor(sympy.Symbol('x'), sympy.Symbol('y')),
            {'func': {'type': '^', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (sympy.Not(sympy.Symbol('x')), {'func': {'type': '!', 'args': [{'symbol': 'x'}]}}),
        (
            sympy.IndexedBase('M')[sympy.Symbol('x'), sympy.Symbol('y')],
            {'func': {'type': '[]', 'args': [{'symbol': 'M'}, {'symbol': 'x'}, {'symbol': 'y'}]}},
        ),
        (
            MeasurementKey(path=('nested',), name='key'),
            {'measurement_key': {'string_key': 'key', 'path': ['nested']}},
        ),
    ],
)
def test_correspondence(value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    parsed = arg_from_proto(msg)
    packed = json_format.MessageToDict(
        arg_to_proto(value),
        **_json_format_kwargs(),
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
    parsed = arg_from_proto(msg)
    assert parsed == 1
    msg = v2.program_pb2.Arg()
    msg.arg_value.double_value = 1.5
    parsed = arg_from_proto(msg)
    assert parsed == 1.5


def test_serialize_sympy_constants():
    proto = arg_to_proto(sympy.pi)
    packed = json_format.MessageToDict(
        proto,
        **_json_format_kwargs(),
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    assert len(packed) == 1
    assert len(packed['arg_value']) == 1
    # protobuf 3.12+ truncates floats to 4 bytes
    assert np.isclose(packed['arg_value']['float_value'], np.float32(sympy.pi), atol=1e-7)


@pytest.mark.parametrize(
    'value,proto',
    [
        ((True, False), {'arg_value': {'bool_values': {'values': [True, False]}}}),
        (
            np.array([True, False], dtype=bool),
            {'arg_value': {'ndarray_value': {'bit_array': {'flat_bytes': 'gA==', 'shape': [2]}}}},
        ),
    ],
)
def test_serialize_conversion(value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    packed = json_format.MessageToDict(
        arg_to_proto(value),
        **_json_format_kwargs(),
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    assert packed == proto


@pytest.mark.parametrize(
    'value',
    [
        np.array([[True, False], [False, True]], dtype=bool),
        np.array([[1.0, 0.5, 0.25], [0.75, 0.125, 0.625]], dtype=np.float64),
        np.array([[1.0, 0.25], [1.75, 1.125]], dtype=np.float32),
        np.array([[-1.0, 0.25], [-1.75, 1.125]], dtype=np.float16),
        np.array([[-1, 2], [-7, 8]], dtype=np.int64),
        np.array([[-16, 126], [-77, 88]], dtype=np.int32),
        np.array([[16, 12], [-7, 8]], dtype=np.int16),
        np.array([[1, 2], [7, -8]], dtype=np.int8),
        np.array([[2, 3], [76, 54]], dtype=np.uint8),
        np.array([[2 + 3j, 3 + 4.5j], [7 + 6j, 5 + 4.25j]], dtype=np.complex128),
        np.array([[2 + 3.5j, 3 + 4.125], [8 + 7j, 5 + 4.75j]], dtype=np.complex64),
        np.array([], dtype=np.complex64),
        np.array([], dtype=np.float64),
    ],
)
def test_ndarray_roundtrip(value: np.ndarray):
    msg = arg_to_proto(value)
    deserialized_value = cast(np.ndarray, arg_from_proto(msg))
    np.testing.assert_array_equal(value, deserialized_value)


@pytest.mark.parametrize('value', [[], (), set(), frozenset()])
def test_empty_sequence_roundtrip(value):
    msg = arg_to_proto(value)
    deserialized_value = arg_from_proto(msg)
    assert value == deserialized_value


@pytest.mark.parametrize(
    'value', [{4, 'a'}, {'b', 5}, frozenset({4, 'a'}), {'a', ('b', 'c', 'd')}, {'a', (2, 'c', 'd')}]
)
def test_sets_roundtrip(value):
    msg = arg_to_proto(value)
    deserialized_value = arg_from_proto(msg)
    assert value == deserialized_value


@pytest.mark.parametrize(
    'value',
    [
        KeyCondition(MeasurementKey('a')),
        SympyCondition(sympy.Symbol('a') > sympy.Symbol('b')),
        SympyCondition(sympy.Symbol('a') >= sympy.Symbol('b')),
        SympyCondition(sympy.Symbol('a') < sympy.Symbol('b')),
        SympyCondition(sympy.Symbol('a') <= sympy.Symbol('b')),
        SympyCondition(sympy.Ne(sympy.Symbol('a'), sympy.Symbol('b'))),
        SympyCondition(sympy.Eq(sympy.Symbol('a'), sympy.Symbol('b'))),
        BitMaskKeyCondition('a'),
        BitMaskKeyCondition('a', bitmask=13),
        BitMaskKeyCondition('a', bitmask=13, target_value=9, equal_target=False),
        BitMaskKeyCondition('a', bitmask=13, target_value=9, equal_target=True),
        BitMaskKeyCondition.create_equal_mask(MeasurementKey('a'), 13),
        BitMaskKeyCondition.create_not_equal_mask(MeasurementKey('a'), 13),
    ],
)
def test_conditions_roundtrip(value):
    msg = v2.program_pb2.Arg()
    condition_to_proto(value, out=msg)
    deserialized_value = condition_from_proto(msg)
    assert value == deserialized_value


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
    assert float_arg_from_proto(proto) == value


def test_missing_required_arg():
    with pytest.raises(ValueError, match='blah is missing'):
        _ = float_arg_from_proto(v2.program_pb2.FloatArg(), required_arg_name='blah')
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = arg_from_proto(v2.program_pb2.Arg(), required_arg_name='blah')
    assert arg_from_proto(v2.program_pb2.Arg()) is None
    assert float_arg_from_proto(v2.program_pb2.FloatArg()) is None


def test_invalid_float_arg():
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = float_arg_from_proto(
            v2.program_pb2.Arg(arg_value=v2.program_pb2.ArgValue(float_value=0.5)),
            required_arg_name='blah',
        )


def test_invalid_sympy_func():
    with pytest.raises(ValueError, match='Unrecognized sympy function'):
        _ = arg_from_proto(v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='dingdong')))
    with pytest.raises(ValueError, match='Unrecognized Sympy expression'):
        _ = arg_to_proto(sympy.Quaternion(sympy.Symbol('a'), 1, 2, 3))


@pytest.mark.parametrize('rotation_angles_arg', [{}, {'rotation_angles': [0.1, 0.3]}])
@pytest.mark.parametrize('qid_shape_arg', [{}, {'qid_shape': [2, 2]}])
@pytest.mark.parametrize('tags_arg', [{}, {'tags': ['test1', 'test2']}])
def test_internal_gate_serialization(rotation_angles_arg, qid_shape_arg, tags_arg):
    g = cirq_google.InternalGate(
        gate_name='g',
        gate_module='test',
        num_qubits=5,
        **rotation_angles_arg,
        **qid_shape_arg,
        **tags_arg,
    )
    proto = v2.program_pb2.InternalGate()
    internal_gate_arg_to_proto(g, out=proto)
    v = internal_gate_from_proto(proto)
    assert g == v


def test_clifford_tableau():
    tests = [
        CliffordTableau(
            1,
            0,
            rs=np.array([True, False], dtype=bool),
            xs=np.array([[True], [False]], dtype=bool),
            zs=np.array([[True], [False]], dtype=bool),
        ),
        CliffordTableau(
            1,
            1,
            rs=np.array([True, True], dtype=bool),
            xs=np.array([[True], [False]], dtype=bool),
            zs=np.array([[False], [False]], dtype=bool),
        ),
    ]
    for ct in tests:
        proto = clifford_tableau_arg_to_proto(ct)
        tableau = clifford_tableau_from_proto(proto)
        assert tableau == ct


def test_serialize_with_units():
    g = cirq_google.InternalGate(
        gate_name='test', gate_module='test', parameter_with_unit=3.14 * tunits.units.ns
    )
    msg = internal_gate_arg_to_proto(g)
    v = internal_gate_from_proto(msg)
    assert g == v
