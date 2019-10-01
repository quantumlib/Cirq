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
"""Common Serializers that can be used by APIs.

This file contains the following serializers (and corresponding deserializers)

    - SINGLE_QUBIT_SERIALIZERS: A list of GateOpSerializer for single qubit
        rotations using cirq Gates.
    - MEASUREMENT_SERIALIZER:  Single GateOpSerializer for the measurement gate
    - SINGLE_QUBIT_SERIALIZERS: A list of GateOpSerializer for single qubit
        rotations confined to half-pi increments using cirq Gates.

"""
from typing import cast, List

import sympy

from cirq import ops, protocols
from cirq.google import op_deserializer, op_serializer


def _near_mod_2(e, t, atol=1e-8):
    if isinstance(e, sympy.Symbol):
        return False
    return abs((e - t + 1) % 2 - 1) <= atol


"""Single qubit serializers for arbitrary rotations"""
SINGLE_QUBIT_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='xy',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter='phase_exponent'),
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
        ]),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy',
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
        serialized_gate_id='xy',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: 0.5),
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
        ]),
    op_serializer.GateOpSerializer(
        gate_type=ops.ZPowGate,
        serialized_gate_id='z',
        args=[
            op_serializer.SerializingArg(serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
            op_serializer.SerializingArg(
                serialized_name='type',
                serialized_type=str,
                gate_getter=lambda x: 'virtual_propagates_forward'),
        ])
]
"""Single qubit deserializers into PhasedXPowGate and ZPowGate"""
SINGLE_QUBIT_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy',
        gate_constructor=ops.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent'),
            op_deserializer.DeserializingArg(serialized_name='half_turns',
                                             constructor_arg_name='exponent')
        ]),
    op_deserializer.GateOpDeserializer(serialized_gate_id='z',
                                       gate_constructor=ops.ZPowGate,
                                       args=[
                                           op_deserializer.DeserializingArg(
                                               serialized_name='half_turns',
                                               constructor_arg_name='exponent')
                                       ]),
]
"""Measurement serializer."""
MEASUREMENT_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.MeasurementGate,
    serialized_gate_id='meas',
    args=[
        op_serializer.SerializingArg(serialized_name='key',
                                     serialized_type=str,
                                     gate_getter=protocols.measurement_key),
        op_serializer.SerializingArg(serialized_name='invert_mask',
                                     serialized_type=List[bool],
                                     gate_getter='invert_mask'),
    ])
"""Measurement deserializer."""
MEASUREMENT_DESERIALIZER = op_deserializer.GateOpDeserializer(
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
    num_qubits_param='num_qubits')
"""Serializers for single qubit rotations confined to half-pi increments"""
SINGLE_QUBIT_HALF_PI_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter='phase_exponent'),
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.PhasedXPowGate, x).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: (x.exponent - 1) / 2)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.XPowGate, x).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: x.exponent / 2)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.YPowGate, x).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: x.exponent - 0.5)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.XPowGate, x).exponent, 0.5)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: x.exponent)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.YPowGate, x).exponent, 0.5)),
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter='phase_exponent'),
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.PhasedXPowGate, x).exponent, 0.5)),
]
"""Deserializers for single qubit rotations confined to half-pi increments"""
SINGLE_QUBIT_HALF_PI_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy_pi',
        gate_constructor=ops.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent'),
            op_deserializer.DeserializingArg(serialized_name='axis_half_turns',
                                             constructor_arg_name='exponent',
                                             value_func=lambda _: 1),
        ]),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy_half_pi',
        gate_constructor=ops.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent'),
            op_deserializer.DeserializingArg(serialized_name='axis_half_turns',
                                             constructor_arg_name='exponent',
                                             value_func=lambda _: 0.5),
        ]),
]
