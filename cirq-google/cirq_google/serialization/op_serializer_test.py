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
    raw_constants = {DEFAULT_TOKEN: 0, default_circuit(): 1}

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

    with pytest.raises(ValueError, match='Cannot serialize repetitions of type'):
        serializer.to_proto(
            to_serialize ** sympy.Symbol('a'), constants=constants, raw_constants=raw_constants
        )


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
    raw_constants = {DEFAULT_TOKEN: 0, default_circuit(): 1}

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


def test_circuit_op_to_proto_complex():
    serializer = cg.CircuitOpSerializer()
    to_serialize = cirq.CircuitOperation(
        circuit=default_circuit(),
        qubit_map={cirq.GridQubit(1, 1): cirq.GridQubit(1, 2)},
        measurement_key_map={'m': 'results'},
        param_resolver={'k': 1.0j},
        repetitions=10,
        repetition_ids=None,
    )
    constants = [
        v2.program_pb2.Constant(string_value=DEFAULT_TOKEN),
        v2.program_pb2.Constant(circuit_value=default_circuit_proto()),
    ]
    raw_constants = {DEFAULT_TOKEN: 0, default_circuit(): 1}
    with pytest.raises(ValueError, match='complex value'):
        serializer.to_proto(to_serialize, constants=constants, raw_constants=raw_constants)
