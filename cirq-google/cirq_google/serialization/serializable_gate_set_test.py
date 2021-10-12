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
import sympy
from google.protobuf import json_format

import cirq
from cirq.testing import assert_deprecated
import cirq_google as cg
from cirq_google.api import v2

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

CIRCUIT_OP_SERIALIZER = cg.CircuitOpSerializer()
CIRCUIT_OP_DESERIALIZER = cg.CircuitOpDeserializer()

MY_GATE_SET = cg.SerializableGateSet(
    gate_set_name='my_gate_set',
    serializers=[X_SERIALIZER, CIRCUIT_OP_SERIALIZER],
    deserializers=[X_DESERIALIZER, CIRCUIT_OP_DESERIALIZER],
)


def test_deprecated_methods():
    with assert_deprecated('Use name instead', deadline='v0.14'):
        _ = MY_GATE_SET.gate_set_name


def op_proto(json: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


def test_naming():
    assert MY_GATE_SET.name == 'my_gate_set'


def test_supported_internal_types():
    assert MY_GATE_SET.supported_internal_types() == (cirq.XPowGate, cirq.FrozenCircuit)


def test_is_supported():
    q0, q1 = cirq.GridQubit.rect(1, 2)
    assert MY_GATE_SET.is_supported(cirq.Circuit(cirq.X(q0), cirq.X(q1)))
    assert not MY_GATE_SET.is_supported(cirq.Circuit(cirq.X(q0), cirq.Z(q1)))


def test_is_supported_subcircuits():
    q0, q1 = cirq.GridQubit.rect(1, 2)
    assert MY_GATE_SET.is_supported(
        cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1))))
    )
    assert not MY_GATE_SET.is_supported(
        cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.Z(q1))))
    )


def test_is_supported_operation():
    q = cirq.GridQubit(1, 1)
    assert MY_GATE_SET.is_supported_operation(cirq.XPowGate()(q))
    assert MY_GATE_SET.is_supported_operation(cirq.X(q))
    assert not MY_GATE_SET.is_supported_operation(cirq.ZPowGate()(q))


def test_is_supported_circuit_operation():
    q = cirq.GridQubit(1, 1)
    assert MY_GATE_SET.is_supported_operation(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q))))
    assert MY_GATE_SET.is_supported_operation(
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q))).with_tags('test_tag')
    )
    assert MY_GATE_SET.is_supported_operation(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q))))
        )
    )
    assert not MY_GATE_SET.is_supported_operation(
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.Z(q)))
    )


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
                    ],
                ),
                v2.program_pb2.Moment(
                    operations=[X_SERIALIZER.to_proto(cirq.X(q0))],
                ),
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
                v2.program_pb2.Moment(
                    operations=[op1, op2],
                ),
                v2.program_pb2.Moment(
                    operations=[X_SERIALIZER.to_proto(cirq.X(q0))],
                ),
            ],
        ),
        constants=[
            v2.program_pb2.Constant(string_value='abc123'),
            v2.program_pb2.Constant(string_value='def456'),
        ],
    )
    assert proto == MY_GATE_SET.serialize(circuit)
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_serialize_deserialize_circuit_with_subcircuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    tag1 = cg.CalibrationTag('abc123')
    fcircuit = cirq.FrozenCircuit(cirq.X(q0))
    circuit = cirq.Circuit(
        cirq.X(q1).with_tags(tag1),
        cirq.CircuitOperation(fcircuit).repeat(repetition_ids=['a', 'b']),
        cirq.CircuitOperation(fcircuit).with_qubit_mapping({q0: q1}),
        cirq.X(q0),
    )

    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.float_value = 1.0
    op1.qubits.add().id = '1_2'
    op1.token_constant_index = 0

    c_op1 = v2.program_pb2.CircuitOperation()
    c_op1.circuit_constant_index = 1
    rep_spec = c_op1.repetition_specification
    rep_spec.repetition_count = 2
    rep_spec.repetition_ids.ids.extend(['a', 'b'])

    c_op2 = v2.program_pb2.CircuitOperation()
    c_op2.circuit_constant_index = 1
    c_op2.repetition_specification.repetition_count = 1
    qmap = c_op2.qubit_map.entries.add()
    qmap.key.id = '1_1'
    qmap.value.id = '1_2'

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(
                    operations=[op1],
                    circuit_operations=[c_op1],
                ),
                v2.program_pb2.Moment(
                    operations=[X_SERIALIZER.to_proto(cirq.X(q0))],
                    circuit_operations=[c_op2],
                ),
            ],
        ),
        constants=[
            v2.program_pb2.Constant(string_value='abc123'),
            v2.program_pb2.Constant(
                circuit_value=v2.program_pb2.Circuit(
                    scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
                    moments=[
                        v2.program_pb2.Moment(
                            operations=[X_SERIALIZER.to_proto(cirq.X(q0))],
                        )
                    ],
                )
            ),
        ],
    )
    assert proto == MY_GATE_SET.serialize(circuit)
    assert MY_GATE_SET.deserialize(proto) == circuit


def test_deserialize_infinite_recursion_fails():
    inf_op = cirq.CircuitOperation(cirq.FrozenCircuit())
    # Maliciously modify the CircuitOperation to be self-referencing.
    setattr(inf_op.circuit, '_moments', tuple(cirq.Circuit(inf_op).moments))
    circuit = cirq.Circuit(inf_op)
    with pytest.raises(RecursionError):
        _ = MY_GATE_SET.serialize(circuit)

    c_op1 = v2.program_pb2.CircuitOperation()
    c_op1.circuit_constant_index = 0
    rep_spec = c_op1.repetition_specification
    rep_spec.repetition_count = 2
    rep_spec.repetition_ids.ids.extend(['a', 'b'])

    # This proto is illegal: c_op1 references a constant containing c_op1.
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='my_gate_set'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(
                    circuit_operations=[c_op1],
                ),
            ],
        ),
        constants=[
            v2.program_pb2.Constant(
                circuit_value=v2.program_pb2.Circuit(
                    scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
                    moments=[
                        v2.program_pb2.Moment(
                            circuit_operations=[c_op1],
                        ),
                    ],
                )
            ),
        ],
    )
    with pytest.raises(ValueError, match="Failed to deserialize circuit"):
        _ = MY_GATE_SET.deserialize(proto)


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


def default_circuit_proto():
    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.string_value = 'k'
    op1.qubits.add().id = '1_1'

    return v2.program_pb2.Circuit(
        scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
        moments=[
            v2.program_pb2.Moment(
                operations=[op1],
            ),
        ],
    )


def default_circuit():
    return cirq.FrozenCircuit(
        cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'),
        cirq.measure(cirq.GridQubit(1, 1), key='m'),
    )


def test_serialize_circuit_op_errors():
    constants = [default_circuit_proto()]
    raw_constants = {default_circuit(): 0}

    op = cirq.CircuitOperation(default_circuit())
    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        MY_GATE_SET.serialize_op(op)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        MY_GATE_SET.serialize_op(op, constants=constants)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        MY_GATE_SET.serialize_op(op, raw_constants=raw_constants)

    NO_CIRCUIT_OP_GATE_SET = cg.SerializableGateSet(
        gate_set_name='no_circuit_op_gateset',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    with pytest.raises(ValueError, match='Cannot serialize CircuitOperation'):
        NO_CIRCUIT_OP_GATE_SET.serialize_op(op, constants=constants, raw_constants=raw_constants)


def test_deserialize_circuit_op_errors():
    constants = [default_circuit_proto()]
    deserialized_constants = [default_circuit()]

    proto = v2.program_pb2.CircuitOperation()
    proto.circuit_constant_index = 0
    proto.repetition_specification.repetition_count = 1

    NO_CIRCUIT_OP_GATE_SET = cg.SerializableGateSet(
        gate_set_name='no_circuit_op_gateset',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    with pytest.raises(ValueError, match='Unsupported serialized CircuitOperation'):
        NO_CIRCUIT_OP_GATE_SET.deserialize_op(
            proto, constants=constants, deserialized_constants=deserialized_constants
        )

    BAD_CIRCUIT_DESERIALIZER = cg.GateOpDeserializer(
        serialized_gate_id='circuit',
        gate_constructor=cirq.ZPowGate,
        args=[],
    )
    BAD_CIRCUIT_DESERIALIZER_GATE_SET = cg.SerializableGateSet(
        gate_set_name='bad_circuit_gateset',
        serializers=[CIRCUIT_OP_SERIALIZER],
        deserializers=[BAD_CIRCUIT_DESERIALIZER],
    )
    with pytest.raises(ValueError, match='Expected CircuitOpDeserializer for id "circuit"'):
        BAD_CIRCUIT_DESERIALIZER_GATE_SET.deserialize_op(
            proto, constants=constants, deserialized_constants=deserialized_constants
        )


def test_serialize_deserialize_circuit_op():
    constants = [default_circuit_proto()]
    raw_constants = {default_circuit(): 0}
    deserialized_constants = [default_circuit()]

    proto = v2.program_pb2.CircuitOperation()
    proto.circuit_constant_index = 0
    proto.repetition_specification.repetition_count = 1

    op = cirq.CircuitOperation(default_circuit())
    assert proto == MY_GATE_SET.serialize_op(op, constants=constants, raw_constants=raw_constants)
    assert (
        MY_GATE_SET.deserialize_op(
            proto, constants=constants, deserialized_constants=deserialized_constants
        )
        == op
    )


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


def test_gateset_with_added_types():
    q = cirq.GridQubit(1, 1)
    x_gateset = cg.SerializableGateSet(
        gate_set_name='x',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    xy_gateset = x_gateset.with_added_types(
        gate_set_name='xy',
        serializers=[Y_SERIALIZER],
        deserializers=[Y_DESERIALIZER],
    )
    assert x_gateset.name == 'x'
    assert x_gateset.is_supported_operation(cirq.X(q))
    assert not x_gateset.is_supported_operation(cirq.Y(q))
    assert xy_gateset.name == 'xy'
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


def test_gateset_with_added_types_again():
    """Verify that adding a serializer twice doesn't mess anything up."""
    q = cirq.GridQubit(2, 2)
    x_gateset = cg.SerializableGateSet(
        gate_set_name='x',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )
    xx_gateset = x_gateset.with_added_types(
        gate_set_name='xx',
        serializers=[X_SERIALIZER],
        deserializers=[X_DESERIALIZER],
    )

    assert xx_gateset.name == 'xx'
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


def test_deserialize_op_bad_operation_proto():
    proto = v2.program_pb2.Circuit()
    with pytest.raises(ValueError, match='Operation proto has unknown type'):
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


def test_serialize_op_bad_operation():
    class NullOperation(cirq.Operation):
        @property
        def qubits(self):
            return tuple()  # coverage: ignore

        def with_qubits(self, *qubits):
            return self  # coverage: ignore

    null_op = NullOperation()
    with pytest.raises(ValueError, match='Operation is of an unrecognized type'):
        MY_GATE_SET.serialize_op(null_op)


def test_serialize_op_bad_operation_proto():
    q0 = cirq.GridQubit(1, 1)
    msg = v2.program_pb2.Circuit()
    with pytest.raises(ValueError, match='Operation proto is of an unrecognized type'):
        MY_GATE_SET.serialize_op(cirq.X(q0), msg)


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
        MY_GATE_SET.deserialize(proto, cg.Bristlecone)
