# Copyright 2026 The Cirq Developers
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

import sympy
import tunits as tu

import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.ops.multi_step_multi_level_reset import MultiStepMultiLevelReset


def test_multi_step_multi_level_reset_properties():
    gate = MultiStepMultiLevelReset()
    assert cirq.num_qubits(gate) == 1
    assert gate.is_reset_gate()
    assert not cirq.has_unitary(gate)


def test_circuit_diagram():
    q = cirq.GridQubit(0, 0)
    op = MultiStepMultiLevelReset()(q)
    circuit = cirq.Circuit(op)
    cirq.testing.assert_has_diagram(
        circuit,
        """
(0, 0): ───[R (MSML)]───
""",
    )


def test_decomposition():
    q = cirq.GridQubit(0, 0)
    op = MultiStepMultiLevelReset()(q)
    decomp = cirq.decompose(op)
    assert decomp == [cirq.ResetChannel()(q)]


def test_equality_and_hashing():
    gate1 = MultiStepMultiLevelReset(
        f_start=2.5 * tu.GHz,
        coupler_amplitudes={"coupler_A": 0.5},
        lengths=(10 * tu.ns, 12 * tu.ns),
    )
    gate2 = MultiStepMultiLevelReset(
        f_start=2.5 * tu.GHz,
        coupler_amplitudes={"coupler_A": 0.5},
        lengths=[10 * tu.ns, 12 * tu.ns],
    )
    gate3 = MultiStepMultiLevelReset(
        f_start=3.0 * tu.GHz,
        coupler_amplitudes={"coupler_A": 0.5},
        lengths=(10 * tu.ns, 12 * tu.ns),
    )

    assert gate1 == gate2
    assert hash(gate1) == hash(gate2)
    assert gate1 != gate3
    assert gate1 != "other"

    # Verify gate can be used in set and in cirq.Moment
    q = cirq.GridQubit(0, 0)
    moment = cirq.Moment(gate1(q))
    assert moment == cirq.Moment(gate2(q))
    assert {gate1, gate2} == {gate1}


def test_repr():
    gate = MultiStepMultiLevelReset()
    assert repr(gate) == 'cirq_google.MultiStepMultiLevelReset()'

    gate_with_args = MultiStepMultiLevelReset(
        f_start=2.5 * tu.GHz, already_at_readout_detuning=False, coupler_amplitudes={"c": 0.5}
    )
    assert repr(gate_with_args) == (
        f"cirq_google.MultiStepMultiLevelReset(f_start={2.5 * tu.GHz!r}, "
        "already_at_readout_detuning=False, coupler_amplitudes={'c': 0.5})"
    )


def test_empty_serialization_round_trip():
    q = cirq.GridQubit(0, 0)
    gate = MultiStepMultiLevelReset()
    op = gate(q)
    circuit = cirq.Circuit(op)

    # Serialize
    proto = cg.CIRCUIT_SERIALIZER.serialize(circuit)

    # Verify proto structure
    op_protos = [c.operation_value for c in proto.constants if c.HasField('operation_value')]
    assert len(op_protos) == 1
    op_proto = op_protos[0]

    assert op_proto.WhichOneof('gate_value') == 'resetgate'
    gate_proto = op_proto.resetgate
    assert gate_proto.reset_type == "MultiStepMultiLevelReset"
    assert len(gate_proto.arguments) == 0

    # Deserialize
    deserialized_circuit = cg.CIRCUIT_SERIALIZER.deserialize(proto)
    assert deserialized_circuit == circuit

    deserialized_op = next(iter(deserialized_circuit.all_operations()))
    assert isinstance(deserialized_op.gate, MultiStepMultiLevelReset)
    assert deserialized_op.gate == gate


def _proto_for_unit(value: float, unit: tu.Value) -> v2.program_pb2.Arg:
    proto = v2.program_pb2.Arg()
    proto.arg_value.value_with_unit.MergeFrom((value * unit).to_proto())
    return proto


def test_full_serialization_round_trip():
    q = cirq.GridQubit(0, 0)
    gate = MultiStepMultiLevelReset(
        f_start=2.5 * tu.GHz,
        already_at_readout_detuning=False,
        f_end=3.0 * tu.GHz,
        end_at_idle=True,
        lengths=(10 * tu.ns, 12 * tu.ns),
        f_swaps_delta=(0.1 * tu.GHz, -0.1 * tu.GHz),
        gs=(10 * tu.MHz, 20 * tu.MHz),
        padding_before=5 * tu.ns,
        padding_after=10 * tu.ns,
        detune_to_start_freq=True,
        start_at_readout_detuning=False,
        coupler_amplitudes={"coupler_A": 0.5},
        compensate_coupled_qubit=True,
    )
    op = gate(q)
    circuit = cirq.Circuit(op)

    # Serialize
    proto = cg.CIRCUIT_SERIALIZER.serialize(circuit)

    # Verify proto structure matches the pyle format exactly
    op_protos = [c.operation_value for c in proto.constants if c.HasField('operation_value')]
    assert len(op_protos) == 1
    op_proto = op_protos[0]

    assert op_proto.WhichOneof('gate_value') == 'resetgate'
    gate_proto = op_proto.resetgate
    assert gate_proto.reset_type == "MultiStepMultiLevelReset"

    expected_args = {
        "f_start": _proto_for_unit(2.5, tu.GHz),
        "already_at_readout_detuning": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(bool_value=False)
        ),
        "f_end": _proto_for_unit(3.0, tu.GHz),
        "end_at_idle": v2.program_pb2.Arg(arg_value=v2.program_pb2.ArgValue(bool_value=True)),
        "lengths": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(
                tuple_value=v2.program_pb2.Tuple(
                    sequence_type=v2.program_pb2.Tuple.SequenceType.TUPLE,
                    values=[_proto_for_unit(10, tu.ns), _proto_for_unit(12, tu.ns)],
                )
            )
        ),
        "f_swaps_delta": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(
                tuple_value=v2.program_pb2.Tuple(
                    sequence_type=v2.program_pb2.Tuple.SequenceType.TUPLE,
                    values=[_proto_for_unit(0.1, tu.GHz), _proto_for_unit(-0.1, tu.GHz)],
                )
            )
        ),
        "gs": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(
                tuple_value=v2.program_pb2.Tuple(
                    sequence_type=v2.program_pb2.Tuple.SequenceType.TUPLE,
                    values=[_proto_for_unit(10, tu.MHz), _proto_for_unit(20, tu.MHz)],
                )
            )
        ),
        "padding_before": _proto_for_unit(5, tu.ns),
        "padding_after": _proto_for_unit(10, tu.ns),
        "detune_to_start_freq": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(bool_value=True)
        ),
        "start_at_readout_detuning": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(bool_value=False)
        ),
        "coupler_amplitudes": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(string_value='__JSON_DICT__:{"coupler_A": 0.5}')
        ),
        "compensate_coupled_qubit": v2.program_pb2.Arg(
            arg_value=v2.program_pb2.ArgValue(bool_value=True)
        ),
    }

    assert gate_proto.arguments == expected_args

    # Deserialize
    deserialized_circuit = cg.CIRCUIT_SERIALIZER.deserialize(proto)
    assert deserialized_circuit == circuit

    deserialized_op = next(iter(deserialized_circuit.all_operations()))
    assert isinstance(deserialized_op.gate, MultiStepMultiLevelReset)
    assert deserialized_op.gate == gate


def test_internal_gate_deserialization_fallback():
    # Verify backward-compatibility deserialization if serialized as internalgate
    op_proto = v2.program_pb2.Operation()
    op_proto.qubit_constant_index.append(0)
    op_proto.internalgate.name = "MultiStepMultiLevelReset"
    op_proto.internalgate.gate_args["already_at_readout_detuning"].arg_value.bool_value = True
    op_proto.internalgate.gate_args["coupler_amplitudes"].arg_value.string_value = (
        '__JSON_DICT__:{"c1": 0.25}'
    )

    program_proto = v2.program_pb2.Program()
    program_proto.constants.add(qubit=v2.program_pb2.Qubit(id="0_0"))
    program_proto.constants.add(operation_value=op_proto)
    moment = program_proto.circuit.moments.add()
    moment.operation_indices.append(1)

    circuit = cg.CIRCUIT_SERIALIZER.deserialize(program_proto)
    deserialized_op = next(iter(circuit.all_operations()))
    assert isinstance(deserialized_op.gate, MultiStepMultiLevelReset)
    assert deserialized_op.gate.already_at_readout_detuning is True
    assert deserialized_op.gate.coupler_amplitudes == {"c1": 0.25}


def test_json_serialization():
    # Empty gate
    gate = MultiStepMultiLevelReset()
    json_text = cirq.to_json(gate)
    deserialized = cirq.read_json(json_text=json_text)
    assert deserialized == gate

    # Gate with symbols/primitives
    gate2 = MultiStepMultiLevelReset(
        f_start=sympy.Symbol("f_start"),
        already_at_readout_detuning=False,
        coupler_amplitudes={"coupler_A": 0.5},
    )
    json_text2 = cirq.to_json(gate2)
    deserialized2 = cirq.read_json(json_text=json_text2)
    assert deserialized2 == gate2
