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

import copy
import numpy as np
import pytest
import sympy

from google.protobuf import json_format

import cirq
from cirq.testing import assert_deprecated
import cirq_google as cg
from cirq_google.api import v2


DEFAULT_TOKEN = 'test_tag'


def op_proto(json: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


class GateWithAttribute(cirq.SingleQubitGate):
    def __init__(self, val):
        self.val = val


class GateWithProperty(cirq.SingleQubitGate):
    def __init__(self, val, not_req=None):
        self._val = val
        self._not_req = not_req

    @property
    def val(self):
        return self._val


class GateWithMethod(cirq.SingleQubitGate):
    def __init__(self, val):
        self._val = val

    def get_val(self):
        return self._val


class SubclassGate(GateWithAttribute):

    pass


def get_val(op):
    return op.gate.get_val()


def test_deprecated_fields():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[],
    )
    with assert_deprecated('Use serialized_id', deadline='v0.13'):
        assert serializer.serialized_gate_id == serializer.serialized_id

    with assert_deprecated('Use internal_type', deadline='v0.13'):
        assert serializer.gate_type == serializer.internal_type


TEST_CASES = (
    (float, 1.0, {'arg_value': {'float_value': 1.0}}),
    (str, 'abc', {'arg_value': {'string_value': 'abc'}}),
    (float, 1, {'arg_value': {'float_value': 1.0}}),
    (List[bool], [True, False], {'arg_value': {'bool_values': {'values': [True, False]}}}),
    (List[bool], (True, False), {'arg_value': {'bool_values': {'values': [True, False]}}}),
    (
        List[bool],
        np.array([True, False], dtype=bool),
        {'arg_value': {'bool_values': {'values': [True, False]}}},
    ),
    (sympy.Symbol, sympy.Symbol('x'), {'symbol': 'x'}),
    (float, sympy.Symbol('x'), {'symbol': 'x'}),
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
)


@pytest.mark.parametrize(('val_type', 'val', 'arg_value'), TEST_CASES)
def test_to_proto_attribute(val_type, val, arg_value):
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(serialized_name='my_val', serialized_type=val_type, op_getter='val')
        ],
    )
    q = cirq.GridQubit(1, 2)
    result = serializer.to_proto(GateWithAttribute(val)(q), arg_function_language='linear')
    expected = op_proto(
        {'gate': {'id': 'my_gate'}, 'args': {'my_val': arg_value}, 'qubits': [{'id': '1_2'}]}
    )
    assert result == expected


@pytest.mark.parametrize(('val_type', 'val', 'arg_value'), TEST_CASES)
def test_to_proto_property(val_type, val, arg_value):
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(serialized_name='my_val', serialized_type=val_type, op_getter='val')
        ],
    )
    q = cirq.GridQubit(1, 2)
    result = serializer.to_proto(GateWithProperty(val)(q), arg_function_language='linear')
    expected = op_proto(
        {'gate': {'id': 'my_gate'}, 'args': {'my_val': arg_value}, 'qubits': [{'id': '1_2'}]}
    )
    assert result == expected


@pytest.mark.parametrize(('val_type', 'val', 'arg_value'), TEST_CASES)
def test_to_proto_callable(val_type, val, arg_value):
    serializer = cg.GateOpSerializer(
        gate_type=GateWithMethod,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(serialized_name='my_val', serialized_type=val_type, op_getter=get_val)
        ],
    )
    q = cirq.GridQubit(1, 2)
    result = serializer.to_proto(GateWithMethod(val)(q), arg_function_language='linear')
    expected = op_proto(
        {'gate': {'id': 'my_gate'}, 'args': {'my_val': arg_value}, 'qubits': [{'id': '1_2'}]}
    )
    assert result == expected


def test_to_proto_gate_predicate():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
        can_serialize_predicate=lambda x: x.gate.val == 1,
    )
    q = cirq.GridQubit(1, 2)
    assert serializer.to_proto(GateWithAttribute(0)(q)) is None
    assert serializer.to_proto(GateWithAttribute(1)(q)) is not None
    assert not serializer.can_serialize_operation(GateWithAttribute(0)(q))
    assert serializer.can_serialize_operation(GateWithAttribute(1)(q))


def test_to_proto_gate_mismatch():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
    )
    q = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='GateWithAttribute.*GateWithProperty'):
        serializer.to_proto(GateWithAttribute(1.0)(q))


def test_to_proto_unsupported_type():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=bytes, op_getter='val')],
    )
    q = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='bytes'):
        serializer.to_proto(GateWithProperty(b's')(q))


def test_to_proto_named_qubit_supported():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
    )
    q = cirq.NamedQubit('a')
    arg_value = 1.0
    result = serializer.to_proto(GateWithProperty(arg_value)(q))

    expected = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': arg_value}}},
            'qubits': [{'id': 'a'}],
        }
    )
    assert result == expected


def test_to_proto_line_qubit_supported():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
    )
    q = cirq.LineQubit('10')
    arg_value = 1.0
    result = serializer.to_proto(GateWithProperty(arg_value)(q))

    expected = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': arg_value}}},
            'qubits': [{'id': '10'}],
        }
    )
    assert result == expected


def test_to_proto_required_but_not_present():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(
                serialized_name='my_val', serialized_type=float, op_getter=lambda x: None
            )
        ],
    )
    q = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='required'):
        serializer.to_proto(GateWithProperty(1.0)(q))


def test_to_proto_no_getattr():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='nope')],
    )
    q = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='does not have'):
        serializer.to_proto(GateWithProperty(1.0)(q))


def test_to_proto_not_required_ok():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val'),
            cg.SerializingArg(
                serialized_name='not_req',
                serialized_type=float,
                op_getter='not_req',
                required=False,
            ),
        ],
    )
    expected = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )

    q = cirq.GridQubit(1, 2)
    assert serializer.to_proto(GateWithProperty(0.125)(q)) == expected


@pytest.mark.parametrize(
    ('val_type', 'val'),
    (
        (float, 's'),
        (str, 1.0),
        (sympy.Symbol, 1.0),
        (List[bool], [1.0]),
        (List[bool], 'a'),
        (List[bool], (1.0,)),
    ),
)
def test_to_proto_type_mismatch(val_type, val):
    serializer = cg.GateOpSerializer(
        gate_type=GateWithProperty,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(serialized_name='my_val', serialized_type=val_type, op_getter='val')
        ],
    )
    q = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match=str(type(val))):
        serializer.to_proto(GateWithProperty(val)(q))


def test_can_serialize_operation_subclass():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
        can_serialize_predicate=lambda x: x.gate.val == 1,
    )
    q = cirq.GridQubit(1, 1)
    assert serializer.can_serialize_operation(SubclassGate(1)(q))
    assert not serializer.can_serialize_operation(SubclassGate(0)(q))


def test_defaults_not_serialized():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[
            cg.SerializingArg(
                serialized_name='my_val', serialized_type=float, default=1.0, op_getter='val'
            )
        ],
    )
    q = cirq.GridQubit(1, 2)
    no_default = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
        }
    )
    assert no_default == serializer.to_proto(GateWithAttribute(0.125)(q))
    with_default = op_proto({'gate': {'id': 'my_gate'}, 'qubits': [{'id': '1_2'}]})
    assert with_default == serializer.to_proto(GateWithAttribute(1.0)(q))


def test_token_serialization():
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
    )
    q = cirq.GridQubit(1, 2)
    tag = cg.CalibrationTag('my_token')
    expected = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
            'token_value': 'my_token',
        }
    )
    assert expected == serializer.to_proto(GateWithAttribute(0.125)(q).with_tags(tag))


ONE_CONSTANT = [v2.program_pb2.Constant(string_value='my_token')]
TWO_CONSTANTS = [
    v2.program_pb2.Constant(string_value='other_token'),
    v2.program_pb2.Constant(string_value='my_token'),
]


@pytest.mark.parametrize(
    ('constants', 'expected_index', 'expected_constants'),
    (
        ([], 0, ONE_CONSTANT),
        (ONE_CONSTANT, 0, ONE_CONSTANT),
        (TWO_CONSTANTS, 1, TWO_CONSTANTS),
    ),
)
def test_token_serialization_with_constant_reference(constants, expected_index, expected_constants):
    serializer = cg.GateOpSerializer(
        gate_type=GateWithAttribute,
        serialized_gate_id='my_gate',
        args=[cg.SerializingArg(serialized_name='my_val', serialized_type=float, op_getter='val')],
    )
    # Make a local copy since we are modifying the array in-place.
    constants = copy.copy(constants)
    q = cirq.GridQubit(1, 2)
    tag = cg.CalibrationTag('my_token')
    expected = op_proto(
        {
            'gate': {'id': 'my_gate'},
            'args': {'my_val': {'arg_value': {'float_value': 0.125}}},
            'qubits': [{'id': '1_2'}],
            'token_constant_index': expected_index,
        }
    )
    assert expected == serializer.to_proto(
        GateWithAttribute(0.125)(q).with_tags(tag), constants=constants
    )
    assert constants == expected_constants


def default_circuit_proto():
    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.string_value = 'k'
    op1.qubits.add().id = '1_1'

    op2 = v2.program_pb2.Operation()
    op2.gate.id = 'x_pow'
    op2.args['half_turns'].arg_value.float_value = 1.0
    op2.qubits.add().id = '1_2'
    op2.token_constant_index = 0

    return v2.program_pb2.Circuit(
        scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
        moments=[
            v2.program_pb2.Moment(
                operations=[op1, op2],
            ),
        ],
    )


def default_circuit():
    return cirq.FrozenCircuit(
        cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'),
        cirq.X(cirq.GridQubit(1, 2)).with_tags(DEFAULT_TOKEN),
        cirq.measure(cirq.GridQubit(1, 1), key='m'),
    )


def test_circuit_op_serializer_properties():
    serializer = cg.CircuitOpSerializer()
    assert serializer.internal_type == cirq.FrozenCircuit
    assert serializer.serialized_id == 'circuit'


def test_can_serialize_circuit_op():
    serializer = cg.CircuitOpSerializer()
    assert serializer.can_serialize_operation(cirq.CircuitOperation(default_circuit()))
    assert not serializer.can_serialize_operation(cirq.X(cirq.GridQubit(1, 1)))


def test_circuit_op_to_proto_errors():
    serializer = cg.CircuitOpSerializer()
    to_serialize = cirq.CircuitOperation(default_circuit())

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    raw_constants = {
        DEFAULT_TOKEN: 0,
        default_circuit(): 1,
    }

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer.to_proto(to_serialize)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer.to_proto(to_serialize, constants=constants)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer.to_proto(to_serialize, raw_constants=raw_constants)

    with pytest.raises(ValueError, match='Serializer expected CircuitOperation'):
        serializer.to_proto(
            v2.program_pb2.Operation(), constants=constants, raw_constants=raw_constants
        )

    bad_raw_constants = {cirq.FrozenCircuit(): 0}
    with pytest.raises(ValueError, match='Encountered a circuit not in the constants table'):
        serializer.to_proto(to_serialize, constants=constants, raw_constants=bad_raw_constants)


@pytest.mark.parametrize('repetitions', [1, 5, ['a', 'b', 'c']])
def test_circuit_op_to_proto(repetitions):
    serializer = cg.CircuitOpSerializer()
    if isinstance(repetitions, int):
        repetition_ids = None
    else:
        repetition_ids = repetitions
        repetitions = len(repetition_ids)
    to_serialize = cirq.CircuitOperation(
        circuit=default_circuit(),
        qubit_map={cirq.GridQubit(1, 1): cirq.GridQubit(1, 2)},
        measurement_key_map={'m': 'results'},
        param_resolver={'k': 1.0},
        repetitions=repetitions,
        repetition_ids=repetition_ids,
    )

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    raw_constants = {
        DEFAULT_TOKEN: 0,
        default_circuit(): 1,
    }

    repetition_spec = v2.program_pb2.RepetitionSpecification()
    if repetition_ids is None:
        repetition_spec.repetition_count = repetitions
    else:
        for rep_id in repetition_ids:
            repetition_spec.repetition_ids.ids.append(rep_id)

    qubit_map = v2.program_pb2.QubitMapping()
    q_p1 = qubit_map.entries.add()
    q_p1.key.id = '1_1'
    q_p1.value.id = '1_2'

    measurement_key_map = v2.program_pb2.MeasurementKeyMapping()
    meas_p1 = measurement_key_map.entries.add()
    meas_p1.key.string_key = 'm'
    meas_p1.value.string_key = 'results'

    arg_map = v2.program_pb2.ArgMapping()
    arg_p1 = arg_map.entries.add()
    arg_p1.key.arg_value.string_value = 'k'
    arg_p1.value.arg_value.float_value = 1.0

    expected = v2.program_pb2.CircuitOperation(
        circuit_constant_index=1,
        repetition_specification=repetition_spec,
        qubit_map=qubit_map,
        measurement_key_map=measurement_key_map,
        arg_map=arg_map,
    )
    actual = serializer.to_proto(to_serialize, constants=constants, raw_constants=raw_constants)
    assert actual == expected
