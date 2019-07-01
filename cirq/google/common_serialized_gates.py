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
"""Standard way to serialize gates common to initial architectures.

For new devices and gate sets, prefer to use these serialized versions
in order to standardize the serialization format across multiple
gate sets.
"""

from typing import Type, cast, List

from cirq import ops, protocols
from cirq.google import op_serializer, op_deserializer


def serialize(gate_type: Type[ops.Gate], gate_name: str,
    arg_dict: dict) -> op_serializer.GateOpSerializer:
    """Initialize a gate serializer
      with a string name and dict of float arguments"""
    arg_array = []
    for key in arg_dict:
        arg_array.append(op_serializer.SerializingArg(serialized_name=key,
                                                      serialized_type=float,
                                                      gate_getter=arg_dict[
                                                          key]))
    return op_serializer.GateOpSerializer(
        gate_type=gate_type,
        serialized_gate_id=gate_name,
        args=arg_array)


def deserialize(gate_type: Type[ops.Gate], gate_name: str,
    arg_dict: dict) -> op_deserializer.GateOpDeserializer:
    """Initialize a gate deserializer
      with a string name and dict of float arguments"""
    arg_array = []
    for key in arg_dict:
        arg_array.append(op_deserializer.DeserializingArg(
            serialized_name=key,
            constructor_arg_name=arg_dict[key]))

    return op_deserializer.GateOpDeserializer(
        serialized_gate_id=gate_name,
        gate_constructor=gate_type,
        args=arg_array)


# Define mapping between a gate and its serializer in a handy dictionary
GATE_SERIALIZER = {
    ops.PhasedXPowGate: serialize(ops.PhasedXPowGate, 'exp_w',
                                  dict(axis_half_turns='phase_exponent',
                                       half_turns='exponent')),
    ops.XPowGate: serialize(ops.XPowGate, 'exp_w',
                            dict(axis_half_turns=lambda x: 0.0,
                                 half_turns='exponent')),
    ops.YPowGate: serialize(ops.YPowGate, 'exp_w',
                            dict(axis_half_turns=lambda x: 0.5,
                                 half_turns='exponent')),
    ops.ZPowGate: serialize(ops.ZPowGate, 'exp_z',
                            dict(half_turns='exponent')),
    ops.CZPowGate: serialize(ops.CZPowGate, 'exp_11',
                             dict(half_turns='exponent')),
    ops.MeasurementGate: op_serializer.GateOpSerializer(
        gate_type=ops.MeasurementGate,
        serialized_gate_id='meas',
        args=[
            op_serializer.SerializingArg(
                serialized_name='key',
                serialized_type=str,
                gate_getter=protocols.measurement_key),
            op_serializer.SerializingArg(serialized_name='invert_mask',
                                         serialized_type=List[bool],
                                         gate_getter='invert_mask')
        ])
}

# Define mapping between a gate and its serializer in a handy dictionary
GATE_DESERIALIZER = {
    ops.PhasedXPowGate: deserialize(ops.PhasedXPowGate, 'exp_w',
                                    dict(axis_half_turns='phase_exponent',
                                         half_turns='exponent')),
    ops.ZPowGate: deserialize(ops.ZPowGate, 'exp_z',
                              dict(half_turns='exponent')),
    ops.CZPowGate: deserialize(ops.CZPowGate, 'exp_11',
                               dict(half_turns='exponent')),
    ops.MeasurementGate: op_deserializer.GateOpDeserializer(
        serialized_gate_id='meas',
        gate_constructor=ops.MeasurementGate,
        args=[
            op_deserializer.DeserializingArg(serialized_name='key',
                                             constructor_arg_name='key'),
            op_deserializer.DeserializingArg(
                serialized_name='invert_mask',
                constructor_arg_name='invert_mask',
                value_func=lambda x: tuple(cast(list, x)))
        ],
        num_qubits_param='num_qubits'),
}
