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
from typing import cast, List, Union

import numpy as np
import sympy

from cirq import ops, protocols, value
from cirq.google import op_deserializer, op_serializer


def _near_mod_n(e, t, n, atol=1e-8):
    if isinstance(e, sympy.Symbol):
        return False
    return abs((e - t + 1) % n - 1) <= atol


def _near_mod_2pi(e, t, atol=1e-8):
    return _near_mod_n(e, t, n=2 * np.pi, atol=atol)


def _near_mod_2(e, t, atol=1e-8):
    return _near_mod_n(e, t, n=2, atol=atol)


#############################################
#
# Single qubit serializers and deserializers
#
#############################################

#
# Single qubit serializers for arbitrary rotations
#
SINGLE_QUBIT_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='xy',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter='phase_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent',
            ),
        ],
    ),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: 0.0,
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent',
            ),
        ],
    ),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: 0.5,
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent',
            ),
        ],
    ),
    op_serializer.GateOpSerializer(
        gate_type=ops.ZPowGate,
        serialized_gate_id='z',
        args=[
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                gate_getter='exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='type',
                serialized_type=str,
                gate_getter=lambda x: 'virtual_propagates_forward',
            ),
        ],
    ),
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXZGate,
        serialized_gate_id='xyz',
        args=[
            op_serializer.SerializingArg(
                serialized_name='x_exponent',
                serialized_type=float,
                gate_getter='x_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='z_exponent',
                serialized_type=float,
                gate_getter='z_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='axis_phase_exponent',
                serialized_type=float,
                gate_getter='axis_phase_exponent',
            ),
        ],
    ),
]


#
# Single qubit deserializers for arbitrary rotations
#
SINGLE_QUBIT_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy',
        gate_constructor=ops.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent',
            ),
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent',
            ),
        ],
    ),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='z',
        gate_constructor=ops.ZPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent',
            ),
        ],
    ),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xyz',
        gate_constructor=ops.PhasedXZGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='x_exponent',
                constructor_arg_name='x_exponent',
            ),
            op_deserializer.DeserializingArg(
                serialized_name='z_exponent',
                constructor_arg_name='z_exponent',
            ),
            op_deserializer.DeserializingArg(
                serialized_name='axis_phase_exponent',
                constructor_arg_name='axis_phase_exponent',
            ),
        ],
    ),
]


#
# Measurement Serializer and Deserializer
#
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


#
# Serializers for single qubit rotations confined to half-pi increments
#
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
                gate_getter=lambda x: (cast(ops.XPowGate, x).exponent - 1) / 2)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.XPowGate, x).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: cast(ops.YPowGate, x).exponent / 2)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.YPowGate, x).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: cast(ops.XPowGate, x).exponent - 0.5)
        ],
        can_serialize_predicate=lambda x: _near_mod_2(
            cast(ops.XPowGate, x).exponent, 0.5)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                gate_getter=lambda x: cast(ops.YPowGate, x).exponent)
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

#
# Deserializers for single qubit rotations confined to half-pi increments
#
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

#############################################
#
# Two qubit serializers and deserializers
#
#############################################

#
# CZ Serializer and deserializer
#
CZ_POW_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.CZPowGate,
    serialized_gate_id='cz',
    args=[
        op_serializer.SerializingArg(serialized_name='half_turns',
                                     serialized_type=float,
                                     gate_getter='exponent')
    ])

CZ_POW_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='cz',
    gate_constructor=ops.CZPowGate,
    args=[
        op_deserializer.DeserializingArg(serialized_name='half_turns',
                                         constructor_arg_name='exponent')
    ])

#
# Sycamore Gate Serializer and deserializer
#
SYC_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.FSimGate,
    serialized_gate_id='syc',
    args=[],
    can_serialize_predicate=(
        lambda e: _near_mod_2pi(cast(ops.FSimGate, e).theta, np.pi / 2) and
        _near_mod_2pi(cast(ops.FSimGate, e).phi, np.pi / 6)))

SYC_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='syc',
    gate_constructor=lambda: ops.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
    args=[])

#
# sqrt(ISWAP) serializer and deserializer
# (e.g. ISWAP ** 0.5)
#
SQRT_ISWAP_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=ops.FSimGate,
        serialized_gate_id='fsim_pi_4',
        args=[],
        can_serialize_predicate=(
            lambda e: _near_mod_2pi(cast(ops.FSimGate, e).theta, np.pi / 4) and
            _near_mod_2pi(cast(ops.FSimGate, e).phi, 0))),
    op_serializer.GateOpSerializer(
        gate_type=ops.ISwapPowGate,
        serialized_gate_id='fsim_pi_4',
        args=[],
        can_serialize_predicate=(lambda e: _near_mod_n(
            cast(ops.ISwapPowGate, e).exponent, -0.5, 4))),
    op_serializer.GateOpSerializer(
        gate_type=ops.FSimGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[],
        can_serialize_predicate=(
            lambda e: _near_mod_2pi(cast(ops.FSimGate, e).theta, -np.pi / 4) and
            _near_mod_2pi(cast(ops.FSimGate, e).phi, 0))),
    op_serializer.GateOpSerializer(
        gate_type=ops.ISwapPowGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[],
        can_serialize_predicate=(lambda e: _near_mod_n(
            cast(ops.ISwapPowGate, e).exponent, +0.5, 4))),
]

SQRT_ISWAP_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='fsim_pi_4',
        gate_constructor=lambda: ops.FSimGate(theta=np.pi / 4, phi=0),
        args=[]),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='inv_fsim_pi_4',
        gate_constructor=lambda: ops.FSimGate(theta=-np.pi / 4, phi=0),
        args=[]),
]

#
# WaitGate serializer and deserializer
#
WAIT_GATE_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.WaitGate,
    serialized_gate_id='wait',
    args=[
        op_serializer.SerializingArg(
            serialized_name='nanos',
            serialized_type=float,
            gate_getter=lambda e: cast(ops.WaitGate, e).duration.total_nanos()),
    ])
WAIT_GATE_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='wait',
    gate_constructor=ops.WaitGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='nanos',
            constructor_arg_name='duration',
            value_func=lambda nanos: value.Duration(nanos=cast(
                Union[int, float, sympy.Basic], nanos)))
    ])
