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
"""Gate sets supported by Google's apis."""

from typing import cast, List

from cirq import ops, protocols
from cirq.google import op_serializer, op_deserializer, serializable_gate_set

# The xmon gate set.
XMON = serializable_gate_set.SerializableGateSet(
    gate_set_name='xmon',
    serializers=[
        op_serializer.GateOpSerializer(
            gate_type=ops.PhasedXPowGate,
            serialized_gate_id='exp_w',
            args=[
                op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                             serialized_type=float,
                                             gate_getter='phase_exponent'),
                op_serializer.SerializingArg(serialized_name='half_turns',
                                             serialized_type=float,
                                             gate_getter='exponent')
            ]),
        op_serializer.GateOpSerializer(gate_type=ops.ZPowGate,
                                       serialized_gate_id='exp_z',
                                       args=[
                                           op_serializer.SerializingArg(
                                               serialized_name='half_turns',
                                               serialized_type=float,
                                               gate_getter='exponent')
                                       ]),
        op_serializer.GateOpSerializer(
            gate_type=ops.XPowGate,
            serialized_gate_id='exp_w',
            args=[
                op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                             serialized_type=float,
                                             gate_getter=lambda x: 0.0),
                op_serializer.SerializingArg(serialized_name='half_turns',
                                             serialized_type=float,
                                             gate_getter='exponent'),
            ]),
        op_serializer.GateOpSerializer(
            gate_type=ops.YPowGate,
            serialized_gate_id='exp_w',
            args=[
                op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                             serialized_type=float,
                                             gate_getter=lambda x: 0.5),
                op_serializer.SerializingArg(serialized_name='half_turns',
                                             serialized_type=float,
                                             gate_getter='exponent'),
            ]),
        op_serializer.GateOpSerializer(gate_type=ops.CZPowGate,
                                       serialized_gate_id='exp_11',
                                       args=[
                                           op_serializer.SerializingArg(
                                               serialized_name='half_turns',
                                               serialized_type=float,
                                               gate_getter='exponent')
                                       ]),
        op_serializer.GateOpSerializer(
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
    ],
    deserializers=[
        op_deserializer.GateOpDeserializer(
            serialized_gate_id='exp_w',
            gate_constructor=ops.PhasedXPowGate,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='axis_half_turns',
                    constructor_arg_name='phase_exponent'),
                op_deserializer.DeserializingArg(
                    serialized_name='half_turns',
                    constructor_arg_name='exponent')
            ]),
        op_deserializer.GateOpDeserializer(
            serialized_gate_id='exp_z',
            gate_constructor=ops.ZPowGate,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='half_turns',
                    constructor_arg_name='exponent')
            ]),
        op_deserializer.GateOpDeserializer(
            serialized_gate_id='exp_11',
            gate_constructor=ops.CZPowGate,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='half_turns',
                    constructor_arg_name='exponent')
            ]),
        op_deserializer.GateOpDeserializer(
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
    ])
