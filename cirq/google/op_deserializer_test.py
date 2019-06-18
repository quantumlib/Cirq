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

from typing import List

import pytest
import sympy

import cirq
import cirq.google as cg


@cirq.value_equality
class GateWithAttribute(cirq.SingleQubitGate):

    def __init__(self, val, not_req=None):
        self.val = val
        self.not_req = not_req

    def _value_equality_values_(self):
        return (self.val,)


TEST_CASES = ((float, 1.0, {
    'arg_value': {
        'float_value': 1.0
    }
}), (str, 'abc', {
    'arg_value': {
        'string_value': 'abc'
    }
}), (float, 1, {
    'arg_value': {
        'float_value': 1.0
    }
}), (List[bool], [True, False], {
    'arg_value': {
        'bool_values': {
            'values': [True, False]
        }
    }
}), (sympy.Symbol, sympy.Symbol('x'), {
    'symbol': 'x'
}))


@pytest.mark.parametrize(('val_type', 'val', 'arg_value'), TEST_CASES)
def test_from_proto(val_type, val, arg_value):
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             )
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'my_val': arg_value
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto_dict(serialized)
    assert result == GateWithAttribute(val)(q)


def test_from_proto_required_missing():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             )
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'not_my_val': {
                'arg_value': {
                    'float_value': 0.125
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    with pytest.raises(Exception, match='my_val'):
        deserializer.from_proto_dict(serialized)


def test_from_proto_unknown_arg_type():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             )
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'my_val': {
                'arg_value': {
                    'what_value': 0.125
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    with pytest.raises(Exception, match='what_value'):
        deserializer.from_proto_dict(serialized)


def test_from_proto_value_func():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                                 value_func=lambda x: x + 1)
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'my_val': {
                'arg_value': {
                    'float_value': 0.125
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto_dict(serialized)
    assert result == GateWithAttribute(1.125)(q)


def test_from_proto_not_required_ok():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             ),
                                             cg.DeserializingArg(
                                                 serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'my_val': {
                'arg_value': {
                    'float_value': 0.125
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    result = deserializer.from_proto_dict(serialized)
    assert result == GateWithAttribute(0.125)(q)


def test_from_proto_missing_required_arg():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             ),
                                             cg.DeserializingArg(
                                                 serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'not_req': {
                'arg_value': {
                    'float_value': 0.125
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    with pytest.raises(ValueError):
        deserializer.from_proto_dict(serialized)


def test_from_proto_required_arg_not_assigned():
    deserializer = cg.GateOpDeserializer(serialized_gate_id='my_gate',
                                         gate_constructor=GateWithAttribute,
                                         args=[
                                             cg.DeserializingArg(
                                                 serialized_name='my_val',
                                                 constructor_arg_name='val',
                                             ),
                                             cg.DeserializingArg(
                                                 serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
                                         ])
    serialized = {
        'gate': {
            'id': 'my_gate'
        },
        'args': {
            'my_val': {}
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    with pytest.raises(ValueError):
        deserializer.from_proto_dict(serialized)
