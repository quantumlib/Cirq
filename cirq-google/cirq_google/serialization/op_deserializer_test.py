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
import sympy

import cirq
import cirq_google as cg
from cirq_google.api import v2

DEFAULT_TOKEN = 'test_tag'


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
        moments=[v2.program_pb2.Moment(operations=[op1, op2])],
    )


def default_circuit():
    return cirq.FrozenCircuit(
        cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'),
        cirq.X(cirq.GridQubit(1, 2)).with_tags(DEFAULT_TOKEN),
        cirq.measure(cirq.GridQubit(1, 1), key='m'),
    )


def test_circuit_op_from_proto_errors():
    deserializer = cg.CircuitOpDeserializer()
    serialized = v2.program_pb2.CircuitOperation(circuit_constant_index=1)

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]

    bad_deserialized_constants = [DEFAULT_TOKEN]
    with pytest.raises(ValueError, match='does not appear in the deserialized_constants list'):
        deserializer.from_proto(
            serialized, constants=constants, deserialized_constants=bad_deserialized_constants
        )

    bad_deserialized_constants = [DEFAULT_TOKEN, 2]
    with pytest.raises(ValueError, match='Constant at index 1 was expected to be a circuit'):
        deserializer.from_proto(
            serialized, constants=constants, deserialized_constants=bad_deserialized_constants
        )


def test_circuit_op_arg_key_errors():
    deserializer = cg.CircuitOpDeserializer()
    arg_map = v2.program_pb2.ArgMapping()
    p1 = arg_map.entries.add()
    p1.key.arg_value.float_value = 1.0
    p1.value.arg_value.float_value = 2.0

    serialized = v2.program_pb2.CircuitOperation(circuit_constant_index=1, arg_map=arg_map)

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    deserialized_constants = [DEFAULT_TOKEN, default_circuit()]

    with pytest.raises(ValueError, match='Invalid key parameter type'):
        deserializer.from_proto(
            serialized, constants=constants, deserialized_constants=deserialized_constants
        )


def test_circuit_op_arg_val_errors():
    deserializer = cg.CircuitOpDeserializer()
    arg_map = v2.program_pb2.ArgMapping()
    p1 = arg_map.entries.add()
    p1.key.arg_value.string_value = 'k'
    p1.value.arg_value.bool_values.values.extend([True, False])

    serialized = v2.program_pb2.CircuitOperation(circuit_constant_index=1, arg_map=arg_map)

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    deserialized_constants = [DEFAULT_TOKEN, default_circuit()]

    with pytest.raises(ValueError, match='Invalid value parameter type'):
        deserializer.from_proto(
            serialized, constants=constants, deserialized_constants=deserialized_constants
        )


@pytest.mark.parametrize('repetitions', [1, 5, ['a', 'b', 'c']])
def test_circuit_op_from_proto(repetitions):
    deserializer = cg.CircuitOpDeserializer()

    repetition_spec = v2.program_pb2.RepetitionSpecification()
    if isinstance(repetitions, int):
        repetition_ids = None
        repetition_spec.repetition_count = repetitions
    else:
        repetition_ids = repetitions
        repetitions = len(repetition_ids)
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

    serialized = v2.program_pb2.CircuitOperation(
        circuit_constant_index=1,
        repetition_specification=repetition_spec,
        qubit_map=qubit_map,
        measurement_key_map=measurement_key_map,
        arg_map=arg_map,
    )

    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    deserialized_constants = [DEFAULT_TOKEN, default_circuit()]

    actual = deserializer.from_proto(
        serialized, constants=constants, deserialized_constants=deserialized_constants
    )
    expected = cirq.CircuitOperation(
        circuit=default_circuit(),
        qubit_map={cirq.GridQubit(1, 1): cirq.GridQubit(1, 2)},
        measurement_key_map={'m': 'results'},
        param_resolver={'k': 1.0},
        repetitions=repetitions,
        repetition_ids=repetition_ids,
    )
    assert actual == expected
