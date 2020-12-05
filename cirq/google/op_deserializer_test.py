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

from typing import Dict, List
import pytest
import sympy

from google.protobuf import json_format

import cirq
import cirq.google as cg
from cirq.google.api import v2


def op_proto(json_dict: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json_dict, op)
    return op


@cirq.value_equality
class GateWithAttribute(cirq.SingleQubitGate):
    def __init__(self, val, not_req=None):
        self.val = val
        self.not_req = not_req

    def _value_equality_values_(self):
        return (self.val,)


TEST_CASES = [
    (float, 1.0, {'arg_value': {'float_value': 1.0}}),
    (str, 'abc', {'arg_value': {'string_value': 'abc'}}),
    (float, 1, {'arg_value': {'float_value': 1.0}}),
    (List[bool], [True, False], {'arg_value': {'bool_values': {'values': [True, False]}}}),
    (sympy.Symbol, sympy.Symbol('x'), {'symbol': 'x'}),
    (
        float,
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
]


@pytest.mark.parametrize(('val_type', 'val', 'arg_value'), TEST_CASES)
def test_from_proto(val_type, val, arg_value):
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            )
        ],
    )
    serialized = op_proto(
        {'gate': {'id': 'my_gate'}, 'args': {'my_val': arg_value}, 'qubits': [{'id': '1_2'}]}
    )
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto(serialized, arg_function_language='linear')
    assert result == GateWithAttribute(val)(q)


def test_from_proto_required_missing():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            )
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'not_my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )
    with pytest.raises(Exception, match='my_val'):
        deserializer.from_proto(serialized)


def test_from_proto_unknown_function():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            )
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {
                'my_val': {
                    'func': {
                        'type': 'UNKNOWN_OPERATION',
                        'args': [
                            {'symbol': 'x'},
                            {'arg_value': {'float_value': -1.0}},
                        ],
                    }
                }
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    with pytest.raises(ValueError, match='Unrecognized function type'):
        _ = deserializer.from_proto(serialized)


def test_from_proto_value_type_not_recognized():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            )
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {
                'my_val': {
                    'arg_value': {},
                }
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    with pytest.raises(ValueError, match='Unrecognized value type'):
        _ = deserializer.from_proto(serialized)


def test_from_proto_function_argument_not_set():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            )
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {
                'my_val': {
                    'func': {
                        'type': 'mul',
                        'args': [
                            {'symbol': 'x'},
                            {},
                        ],
                    }
                }
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    with pytest.raises(ValueError, match='A multiplication argument is missing'):
        _ = deserializer.from_proto(serialized, arg_function_language='linear')


def test_from_proto_value_func():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val', constructor_arg_name='val', value_func=lambda x: x + 1
            )
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto(serialized)
    assert result == GateWithAttribute(1.125)(q)


def test_from_proto_not_required_ok():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            ),
            cg.DeserializingArg(
                serialized_name='not_req', constructor_arg_name='not_req', required=False
            ),
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto(serialized)
    assert result == GateWithAttribute(0.125)(q)


def test_from_proto_missing_required_arg():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            ),
            cg.DeserializingArg(
                serialized_name='not_req', constructor_arg_name='not_req', required=False
            ),
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'not_req': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )
    with pytest.raises(ValueError):
        deserializer.from_proto(serialized)


def test_from_proto_required_arg_not_assigned():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(
                serialized_name='my_val',
                constructor_arg_name='val',
            ),
            cg.DeserializingArg(
                serialized_name='not_req', constructor_arg_name='not_req', required=False
            ),
        ],
    )
    serialized = op_proto(
        {'gate': {'id': 'my_gate'}, 'args': {'my_val': {}}, 'qubits': [{'id': '1_2'}]}
    )
    with pytest.raises(ValueError):
        deserializer.from_proto(serialized)


def test_defaults():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(serialized_name='my_val', constructor_arg_name='val', default=1.0),
            cg.DeserializingArg(
                serialized_name='not_req',
                constructor_arg_name='not_req',
                default='hello',
                required=False,
            ),
        ],
    )
    serialized = op_proto({'gate': {'id': 'my_gate'}, 'args': {}, 'qubits': [{'id': '1_2'}]})
    g = GateWithAttribute(1.0)
    g.not_req = 'hello'
    assert deserializer.from_proto(serialized) == g(cirq.GridQubit(1, 2))


def test_token():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(serialized_name='my_val', constructor_arg_name='val'),
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 1.25}}},
            'qubits': [{'id': '1_2'}],
            'token_value': 'abc123',
        }
    )
    op = GateWithAttribute(1.25)(cirq.GridQubit(1, 2))
    op = op.with_tags(cg.CalibrationTag('abc123'))
    assert deserializer.from_proto(serialized) == op


def test_token_with_references():
    deserializer = cg.GateOpDeserializer(
        serialized_gate_id='my_gate',
        gate_constructor=GateWithAttribute,
        args=[
            cg.DeserializingArg(serialized_name='my_val', constructor_arg_name='val'),
        ],
    )
    serialized = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 1.25}}},
            'qubits': [{'id': '1_2'}],
            'token_constant_index': 1,
        }
    )
    op = GateWithAttribute(1.25)(cirq.GridQubit(1, 2))
    op = op.with_tags(cg.CalibrationTag('abc123'))
    constants = []
    constant = v2.program_pb2.Constant()
    constant.string_value = 'my_token'
    constants.append(constant)
    constant = v2.program_pb2.Constant()
    constant.string_value = 'abc123'
    constants.append(constant)
    assert deserializer.from_proto(serialized, constants=constants) == op

    with pytest.raises(ValueError, match='Proto has references to constants table'):
        deserializer.from_proto(serialized)
