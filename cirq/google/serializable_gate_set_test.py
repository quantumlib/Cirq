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

from typing import Dict
import pytest
from google.protobuf import json_format

import cirq
import cirq.google as cg
from cirq.google.api import v2

X_SERIALIZER = cg.GateOpSerializer(
    gate_type=cirq.XPowGate,
    serialized_gate_id='x_pow',
    args=[
        cg.SerializingArg(
            serialized_name='half_turns',
            serialized_type=float,
            op_getter='exponent',
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
            op_getter='exponent',
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


def op_proto(json: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


def test_supported_gate_types():
    assert MY_GATE_SET.supported_gate_types() == (cirq.XPowGate,)


def test_is_supported():
    q0, q1 = cirq.GridQubit.rect(1, 2)
    assert MY_GATE_SET.is_supported(cirq.Circuit(cirq.X(q0), cirq.X(q1)))
    assert not MY_GATE_SET.is_supported(cirq.Circuit(cirq.X(q0), cirq.Z(q1)))


def test_is_supported_operation():
    q = cirq.GridQubit(1, 1)
    assert MY_GATE_SET.is_supported_operation(cirq.XPowGate()(q))
    assert MY_GATE_SET.is_supported_operation(cirq.X(q))
    assert not MY_GATE_SET.is_supported_operation(cirq.ZPowGate()(q))


def test_is_supported_operation_can_serialize_predicate():
    q = cirq.GridQubit(1, 2)
    serializer = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x_pow',
        args=[
            cg.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                op_getter='exponent',
            )
        ],
        can_serialize_predicate=lambda x: x.gate.exponent == 1.0,
    )
    gate_set = cg.SerializableGateSet(
        gate_set_name='my_gate_set', serializers=[serializer], deserializers=[X_DESERIALIZER]
    )
    assert gate_set.is_supported_operation(cirq.XPowGate()(q))
    assert not gate_set.is_supported_operation(cirq.XPowGate()(q) ** 0.5)
    assert gate_set.is_supported_operation(cirq.X(q))


def test_serialize_deserialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.X(q0))

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(
                    operations=[
                        X_SERIALIZER.to_proto(cirq.X(q0)),
                        X_SERIALIZER.to_proto(cirq.X(q1)),
                    ]
                ),
                v2.program_pb2.Moment(operations=[X_SERIALIZER.to_proto(cirq.X(q0))]),
            ],
        ),
    )
    assert proto == MY_GATE_SET.serialize(circuit)
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_serialize_deserialize_circuit_with_tokens():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    tag1 = cg.CalibrationTag('abc123')
    tag2 = cg.CalibrationTag('def456')
    circuit = cirq.Circuit(cirq.X(q0).with_tags(tag1), cirq.X(q1).with_tags(tag2), cirq.X(q0))
    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.float_value = 1.0
    op1.qubits.add().id = '1_1'
    op1.token_constant_index = 0
    op2 = v2.program_pb2.Operation()
    op2.gate.id = 'x_pow'
    op2.args['half_turns'].arg_value.float_value = 1.0
    op2.qubits.add().id = '1_2'
    op2.token_constant_index = 1
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(operations=[op1, op2]),
                v2.program_pb2.Moment(operations=[X_SERIALIZER.to_proto(cirq.X(q0))]),
            ],
        ),
        constants=[
            v2.program_pb2.Constant(string_value='abc123'),
            v2.program_pb2.Constant(string_value='def456'),
        ],
    )
    assert proto == MY_GATE_SET.serialize(circuit)
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_deserialize_bad_operation_id():
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(operations=[]),
                v2.program_pb2.Moment(
                    operations=[
                        v2.program_pb2.Operation(
                            gate=v2.program_pb2.Gate(id='UNKNOWN_GATE'),
                            args={
                                'half_turns': v2.program_pb2.Arg(
                                    arg_value=v2.program_pb2.ArgValue(float_value=1.0)
                                )
                            },
                            qubits=[v2.program_pb2.Qubit(id='1_1')],
                        )
                    ]
                ),
            ],
        ),
    )
    with pytest.raises(
        ValueError, match='problem in moment 1 handling an operation with the following'
    ):
        MY_GATE_SET.deserialize(proto)


def test_serialize_deserialize_empty_circuit():
    circuit = cirq.Circuit()

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]
        ),
    )
    assert proto == MY_GATE_SET.serialize(circuit)
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_deserialize_empty_moment():
    circuit = cirq.Circuit([cirq.Moment()])

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(),
            ],
        ),
    )
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_serialize_unrecognized():
    with pytest.raises(NotImplementedError, match='program type'):
        MY_GATE_SET.serialize("not quite right")


def test_serialize_deserialize_schedule_no_device():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        schedule=v2.program_pb2.Schedule(
            scheduled_operations=[
                v2.program_pb2.ScheduledOperation(
                    operation=X_SERIALIZER.to_proto(cirq.X(q0)), start_time_picos=0
                ),
                v2.program_pb2.ScheduledOperation(
                    operation=X_SERIALIZER.to_proto(cirq.X(q1)), start_time_picos=200000
                ),
                v2.program_pb2.ScheduledOperation(
                    operation=X_SERIALIZER.to_proto(cirq.X(q0)), start_time_picos=400000
                ),
            ]
        ),
    )
    with pytest.raises(ValueError):
        MY_GATE_SET.deserialize(proto)


def test_serialize_deserialize_op():
    q0 = cirq.GridQubit(1, 1)
    proto = op_proto(
        {
            'gate': {'id': 'x_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )
    assert proto == MY_GATE_SET.serialize_op(cirq.XPowGate(exponent=0.125)(q0))
    assert MY_GATE_SET.deserialize_op(proto) == cirq.XPowGate(exponent=0.125)(q0)


def test_serialize_deserialize_op_with_token():
    q0 = cirq.GridQubit(1, 1)
    proto = op_proto(
        {
            'gate': {'id': 'x_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
            'token_value': 'abc123',
        }
    )
    op = cirq.XPowGate(exponent=0.125)(q0).with_tags(cg.CalibrationTag('abc123'))
    assert proto == MY_GATE_SET.serialize_op(op)
    assert MY_GATE_SET.deserialize_op(proto) == op


def test_serialize_deserialize_op_with_constants():
    q0 = cirq.GridQubit(1, 1)
    proto = op_proto(
        {
            'gate': {'id': 'x_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
            'token_constant_index': 0,
        }
    )
    op = cirq.XPowGate(exponent=0.125)(q0).with_tags(cg.CalibrationTag('abc123'))
    assert proto == MY_GATE_SET.serialize_op(op, constants=[])
    constant = v2.program_pb2.Constant()
    constant.string_value = 'abc123'
    assert MY_GATE_SET.deserialize_op(proto, constants=[constant]) == op


def test_serialize_deserialize_op_subclass():
    q0 = cirq.GridQubit(1, 1)
    proto = op_proto(
        {
            'gate': {'id': 'x_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 1.0}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )
    # cirq.X is a subclass of XPowGate.
    assert proto == MY_GATE_SET.serialize_op(cirq.X(q0))
    assert MY_GATE_SET.deserialize_op(proto) == cirq.X(q0)


def test_multiple_serializers():
    serializer1 = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x_pow',
        args=[
            cg.SerializingArg(
                serialized_name='half_turns', serialized_type=float, op_getter='exponent'
            )
        ],
        can_serialize_predicate=lambda x: x.gate.exponent != 1,
    )
    serializer2 = cg.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='x',
        args=[
            cg.SerializingArg(
                serialized_name='half_turns', serialized_type=float, op_getter='exponent'
            )
        ],
        can_serialize_predicate=lambda x: x.gate.exponent == 1,
    )
    gate_set = cg.SerializableGateSet(
        gate_set_name='my_gate_set', serializers=[serializer1, serializer2], deserializers=[]
    )
    q0 = cirq.GridQubit(1, 1)
    assert gate_set.serialize_op(cirq.X(q0)).gate.id == 'x'
    assert gate_set.serialize_op(cirq.X(q0) ** 0.5).gate.id == 'x_pow'


def test_gateset_with_added_gates():
    q = cirq.GridQubit(1, 1)
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
    assert x_gateset.is_supported_operation(cirq.X(q))
    assert not x_gateset.is_supported_operation(cirq.Y(q))
    assert xy_gateset.gate_set_name == 'xy'
    assert xy_gateset.is_supported_operation(cirq.X(q))
    assert xy_gateset.is_supported_operation(cirq.Y(q))

    # test serialization and deserialization
    proto = op_proto(
        {
            'gate': {'id': 'y_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )

    expected_gate = cirq.YPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
    assert xy_gateset.serialize_op(expected_gate) == proto
    assert xy_gateset.deserialize_op(proto) == expected_gate


def test_gateset_with_added_gates_again():
    """Verify that adding a serializer twice doesn't mess anything up."""
    q = cirq.GridQubit(2, 2)
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
    assert xx_gateset.is_supported_operation(cirq.X(q))
    assert not xx_gateset.is_supported_operation(cirq.Y(q))

    # test serialization and deserialization
    proto = op_proto(
        {
            'gate': {'id': 'x_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )

    expected_gate = cirq.XPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
    assert xx_gateset.serialize_op(expected_gate) == proto
    assert xx_gateset.deserialize_op(proto) == expected_gate


def test_deserialize_op_invalid_gate():
    proto = op_proto(
        {
            'gate': {},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )
    with pytest.raises(ValueError, match='does not have a gate'):
        MY_GATE_SET.deserialize_op(proto)

    proto = op_proto(
        {
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )
    with pytest.raises(ValueError, match='does not have a gate'):
        MY_GATE_SET.deserialize_op(proto)


def test_deserialize_unsupported_gate_type():
    proto = op_proto(
        {
            'gate': {'id': 'no_pow'},
            'args': {
                'half_turns': {'arg_value': {'float_value': 0.125}},
            },
            'qubits': [{'id': '1_1'}],
        }
    )
    with pytest.raises(ValueError, match='no_pow'):
        MY_GATE_SET.deserialize_op(proto)


def test_serialize_op_unsupported_type():
    q0 = cirq.GridQubit(1, 1)
    with pytest.raises(ValueError, match='YPowGate'):
        MY_GATE_SET.serialize_op(cirq.YPowGate()(q0))


def test_deserialize_invalid_gate_set():
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(gate_set='not_my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]
        ),
    )
    with pytest.raises(ValueError, match='not_my_gate_set'):
        MY_GATE_SET.deserialize(proto)

    proto.language.gate_set = ''
    with pytest.raises(ValueError, match='Missing gate set'):
        MY_GATE_SET.deserialize(proto)

    proto = v2.program_pb2.Program(
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]
        )
    )
    with pytest.raises(ValueError, match='Missing gate set'):
        MY_GATE_SET.deserialize(proto)


def test_deserialize_schedule_missing_device():
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(gate_set='my_gate_set'),
        schedule=v2.program_pb2.Schedule(scheduled_operations=[]),
    )
    with pytest.raises(ValueError, match='device'):
        MY_GATE_SET.deserialize(proto)


def test_deserialize_no_operation():
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(gate_set='my_gate_set'),
        schedule=v2.program_pb2.Schedule(
            scheduled_operations=[v2.program_pb2.ScheduledOperation(start_time_picos=0)]
        ),
    )
    with pytest.raises(ValueError, match='operation'):
        MY_GATE_SET.deserialize(proto, cirq.google.Bristlecone)
