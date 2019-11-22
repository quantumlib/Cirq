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

import pytest

import cirq
import cirq.google as cg

X_SERIALIZER = cg.GateOpSerializer(
    gate_type=cirq.XPowGate,
    serialized_gate_id='x_pow',
    args=[
        cg.SerializingArg(
            serialized_name='half_turns',
            serialized_type=float,
            gate_getter='exponent',
        )
    ],
)

X_DESERIALIZER = cg.GateOpDeserializer(
    serialized_gate_id='x_pow',
    gate_constructor=cirq.XPowGate,
    args=[
        cg.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
        )
    ],
)

Y_SERIALIZER = cg.GateOpSerializer(
    gate_type=cirq.YPowGate,
    serialized_gate_id='y_pow',
    args=[
        cg.SerializingArg(
            serialized_name='half_turns',
            serialized_type=float,
            gate_getter='exponent',
        )
    ],
)

Y_DESERIALIZER = cg.GateOpDeserializer(
    serialized_gate_id='y_pow',
    gate_constructor=cirq.YPowGate,
    args=[
        cg.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
        )
    ],
)

MY_GATE_SET = cg.SerializableGateSet(
    gate_set_name='my_gate_set',
    serializers=[X_SERIALIZER],
    deserializers=[X_DESERIALIZER],
)


def test_supported_gate_types():
    assert MY_GATE_SET.supported_gate_types() == (cirq.XPowGate,)


def test_is_supported_gate():
    assert MY_GATE_SET.is_supported_gate(cirq.XPowGate())
    assert MY_GATE_SET.is_supported_gate(cirq.X)
    assert not MY_GATE_SET.is_supported_gate(cirq.ZPowGate())


def test_is_supported_gate_can_serialize_predicate():
    serializer = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x_pow',
        args=[
            cg.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent',
            )
        ],
        can_serialize_predicate=lambda x: x.exponent == 1.0)
    gate_set = cg.SerializableGateSet(gate_set_name='my_gate_set',
                                      serializers=[serializer],
                                      deserializers=[X_DESERIALIZER])
    assert gate_set.is_supported_gate(cirq.XPowGate())
    assert not gate_set.is_supported_gate(cirq.XPowGate()**0.5)
    assert gate_set.is_supported_gate(cirq.X)


def test_serialize_deserialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.X(q0))

    proto = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'my_gate_set'
        },
        'circuit': {
            'scheduling_strategy':
            1,
            'moments': [
                {
                    'operations': [
                        X_SERIALIZER.to_proto_dict(cirq.X(q0)),
                        X_SERIALIZER.to_proto_dict(cirq.X(q1))
                    ]
                },
                {
                    'operations': [X_SERIALIZER.to_proto_dict(cirq.X(q0))]
                },
            ]
        },
    }
    assert proto == MY_GATE_SET.serialize_dict(circuit)
    assert MY_GATE_SET.deserialize_dict(proto) == circuit


def test_deserialize_bad_operation_id():
    proto = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'my_gate_set'
        },
        'circuit': {
            'scheduling_strategy':
            1,
            'moments': [
                {
                    'operations': [],
                },
                {
                    'operations': [
                        {
                            'gate': {
                                'id': 'UNKNOWN_GATE'
                            },
                            'args': {
                                'half_turns': {
                                    'arg_value': {
                                        'float_value': 1.0
                                    }
                                }
                            },
                            'qubits': [{
                                'id': '1_1'
                            }]
                        },
                    ]
                },
            ]
        },
    }
    with pytest.raises(ValueError,
                       match='problem in moment 1 handling an '
                       'operation with the following'):
        MY_GATE_SET.deserialize_dict(proto)


def test_serialize_deserialize_empty_circuit():
    circuit = cirq.Circuit()

    proto = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'my_gate_set'
        },
        'circuit': {
            'scheduling_strategy': 1,
            'moments': []
        },
    }
    assert proto == MY_GATE_SET.serialize_dict(circuit)
    assert MY_GATE_SET.deserialize_dict(proto) == circuit


def test_deserialize_empty_moment():
    circuit = cirq.Circuit([cirq.Moment()])

    proto = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'my_gate_set'
        },
        'circuit': {
            'scheduling_strategy': 1,
            'moments': [{}]
        },
    }
    assert MY_GATE_SET.deserialize_dict(proto) == circuit


def test_serialize_unrecognized():
    with pytest.raises(NotImplementedError, match='program type'):
        MY_GATE_SET.serialize("not quite right")


def test_serialize_deserialize_schedule_no_device():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    proto = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'my_gate_set'
        },
        'schedule': {
            'scheduled_operations': [
                {
                    'operation': X_SERIALIZER.to_proto_dict(cirq.X(q0)),
                    'start_time_picos': '0'
                },
                {
                    'operation': X_SERIALIZER.to_proto_dict(cirq.X(q1)),
                    'start_time_picos': '200000',
                },
                {
                    'operation': X_SERIALIZER.to_proto_dict(cirq.X(q0)),
                    'start_time_picos': '400000',
                },
            ]
        },
    }
    with pytest.raises(ValueError):
        MY_GATE_SET.deserialize_dict(proto)


def test_serialize_deserialize_op():
    q0 = cirq.GridQubit(1, 1)
    proto = {
        'gate': {
            'id': 'x_pow'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }
    assert proto == MY_GATE_SET.serialize_op_dict(
        cirq.XPowGate(exponent=0.125)(q0))
    assert MY_GATE_SET.deserialize_op_dict(proto) == cirq.XPowGate(
        exponent=0.125)(q0)


def test_serialize_deserialize_op_subclass():
    q0 = cirq.GridQubit(1, 1)
    proto = {
        'gate': {
            'id': 'x_pow'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 1.0
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }
    # cirq.X is a sublcass of XPowGate.
    assert proto == MY_GATE_SET.serialize_op_dict(cirq.X(q0))
    assert MY_GATE_SET.deserialize_op_dict(proto) == cirq.X(q0)


def test_multiple_serializers():
    serializer1 = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x_pow',
        args=[
            cg.SerializingArg(serialized_name='half_turns',
                              serialized_type=float,
                              gate_getter='exponent')
        ],
        can_serialize_predicate=lambda x: x.exponent != 1)
    serializer2 = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x',
        args=[
            cg.SerializingArg(serialized_name='half_turns',
                              serialized_type=float,
                              gate_getter='exponent')
        ],
        can_serialize_predicate=lambda x: x.exponent == 1)
    gate_set = cg.SerializableGateSet(gate_set_name='my_gate_set',
                                      serializers=[serializer1, serializer2],
                                      deserializers=[])
    q0 = cirq.GridQubit(1, 1)
    assert gate_set.serialize_op(cirq.X(q0)).gate.id == 'x'
    assert gate_set.serialize_op(cirq.X(q0)**0.5).gate.id == 'x_pow'


def test_gateset_with_added_gates():
    x_gateset = cg.SerializableGateSet(
        gate_set_name='x',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    xy_gateset = x_gateset.with_added_gates(
        gate_set_name='xy',
        serializers=[Y_SERIALIZER],
        deserializers=[Y_DESERIALIZER],
    )
    assert x_gateset.gate_set_name == 'x'
    assert x_gateset.is_supported_gate(cirq.X)
    assert not x_gateset.is_supported_gate(cirq.Y)

    assert xy_gateset.gate_set_name == 'xy'
    assert xy_gateset.is_supported_gate(cirq.X)
    assert xy_gateset.is_supported_gate(cirq.Y)

    # test serialization and deserialization
    proto = {
        'gate': {
            'id': 'y_pow'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }

    expected_gate = cirq.YPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
    assert xy_gateset.serialize_op_dict(expected_gate) == proto
    assert xy_gateset.deserialize_op_dict(proto) == expected_gate


def test_gateset_with_added_gates_again():
    """Verify that adding a serializer twice doesn't mess anything up."""
    x_gateset = cg.SerializableGateSet(
        gate_set_name='x',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    xx_gateset = x_gateset.with_added_gates(
        gate_set_name='xx',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )

    assert xx_gateset.gate_set_name == 'xx'
    assert xx_gateset.is_supported_gate(cirq.X)
    assert not xx_gateset.is_supported_gate(cirq.Y)

    # test serialization and deserialization
    proto = {
        'gate': {
            'id': 'x_pow'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }

    expected_gate = cirq.XPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
    assert xx_gateset.serialize_op_dict(expected_gate) == proto
    assert xx_gateset.deserialize_op_dict(proto) == expected_gate


def test_deserialize_op_invalid_gate():
    proto = {
        'gate': {},
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }
    with pytest.raises(ValueError, match='does not have a gate'):
        MY_GATE_SET.deserialize_op_dict(proto)

    proto = {
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }
    with pytest.raises(ValueError, match='does not have a gate'):
        MY_GATE_SET.deserialize_op_dict(proto)


def test_deserialize_unsupported_gate_type():
    proto = {
        'gate': {
            'id': 'no_pow'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': 0.125
                }
            },
        },
        'qubits': [{
            'id': '1_1'
        }]
    }
    with pytest.raises(ValueError, match='no_pow'):
        MY_GATE_SET.deserialize_op_dict(proto)


def test_serialize_op_unsupported_type():
    q0 = cirq.GridQubit(1, 1)
    with pytest.raises(ValueError, match='YPowGate'):
        MY_GATE_SET.serialize_op_dict(cirq.YPowGate()(q0))


def test_deserialize_invalid_gate_set():
    proto = {
        'language': {
            'gate_set': 'not_my_gate_set'
        },
        'circuit': {
            'scheduling_strategy': 1,
            'moments': []
        },
    }
    with pytest.raises(ValueError, match='not_my_gate_set'):
        MY_GATE_SET.deserialize_dict(proto)

    proto['language'] = {}
    with pytest.raises(ValueError, match='Missing gate set'):
        MY_GATE_SET.deserialize_dict(proto)

    proto = {
        'circuit': {
            'scheduling_strategy': 1,
            'moments': []
        },
    }
    with pytest.raises(ValueError, match='Missing gate set'):
        MY_GATE_SET.deserialize_dict(proto)


def test_deserialize_schedule_missing_device():
    proto = {
        'language': {
            'gate_set': 'my_gate_set'
        },
        'schedule': {
            'scheduled_operations': []
        },
    }
    with pytest.raises(ValueError, match='device'):
        MY_GATE_SET.deserialize_dict(proto)


def test_deserialize_no_operation():
    proto = {
        'language': {
            'gate_set': 'my_gate_set'
        },
        'schedule': {
            'scheduled_operations': [
                {
                    'start_time_picos': 0
                },
            ]
        },
    }
    with pytest.raises(ValueError, match='operation'):
        MY_GATE_SET.deserialize_dict(proto, cirq.google.Bristlecone)
