# Copyright 2018 The Cirq Developers
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
from cirq import ops
from cirq.google.common_serialized_gates import GATE_SERIALIZER, \
    GATE_DESERIALIZER, LAMBDA_HALF, LAMBDA_ZERO
from cirq.google import op_serializer, op_deserializer


def test_serialized_gate_map():
    assert GATE_SERIALIZER[
               ops.PhasedXPowGate] == op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='exp_w',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter='phase_exponent'),
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent')
        ])
    assert GATE_SERIALIZER[ops.ZPowGate] == op_serializer.GateOpSerializer(
        gate_type=ops.ZPowGate,
        serialized_gate_id='exp_z',
        args=[
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent')
        ])
    assert GATE_SERIALIZER[ops.XPowGate] == op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='exp_w',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=LAMBDA_ZERO),
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
        ])
    assert GATE_SERIALIZER[ops.YPowGate] == op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='exp_w',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=LAMBDA_HALF),
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
        ])
    assert GATE_SERIALIZER[ops.CZPowGate] == op_serializer.GateOpSerializer(
        gate_type=ops.CZPowGate,
        serialized_gate_id='exp_11',
        args=[
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent')
        ])


def test_deserialized_gate_map():
    assert GATE_DESERIALIZER[
               ops.PhasedXPowGate] == op_deserializer.GateOpDeserializer(
        serialized_gate_id='exp_w',
        gate_constructor=ops.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent'),
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent')
        ])
    assert GATE_DESERIALIZER[
               ops.ZPowGate] == op_deserializer.GateOpDeserializer(
        serialized_gate_id='exp_z',
        gate_constructor=ops.ZPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent')
        ])
    assert GATE_DESERIALIZER[
               ops.CZPowGate] == op_deserializer.GateOpDeserializer(
        serialized_gate_id='exp_11',
        gate_constructor=ops.CZPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent')
        ])
