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
from cirq.google.api import v2
from cirq.google.ops import PhysicalZTag

# Type strings used in serialization for the two types of Z operations
PHYSICAL_Z = 'physical'
VIRTUAL_Z = 'virtual_propagates_forward'


# Default tolerance for differences in floating point
# Note that Google protocol buffers use floats
# which trigger a conversion from double precision to single precision
# This results in errors possibly up to 1e-6
# (23 bits for mantissa in single precision)
_DEFAULT_ATOL = 1e-6


def _near_mod_n(e, t, n, atol=_DEFAULT_ATOL):
    if isinstance(e, sympy.Symbol):
        return False
    return abs((e - t + 1) % n - 1) <= atol


def _near_mod_2pi(e, t, atol=_DEFAULT_ATOL):
    return _near_mod_n(e, t, n=2 * np.pi, atol=atol)


def _near_mod_2(e, t, atol=_DEFAULT_ATOL):
    return _near_mod_n(e, t, n=2, atol=atol)


def _convert_physical_z(op: ops.Operation, proto: v2.program_pb2.Operation):
    if 'type' in proto.args:
        if proto.args['type'].arg_value.string_value == PHYSICAL_Z:
            return op.with_tags(PhysicalZTag())
    return op


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
                op_getter='phase_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                op_getter='exponent',
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
                op_getter=lambda op: 0.0,
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                op_getter='exponent',
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
                op_getter=lambda op: 0.5,
            ),
            op_serializer.SerializingArg(
                serialized_name='half_turns',
                serialized_type=float,
                op_getter='exponent',
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
                op_getter='exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='type',
                serialized_type=str,
                op_getter=lambda op: PHYSICAL_Z
                if PhysicalZTag() in op.tags else VIRTUAL_Z,
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
                op_getter='x_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='z_exponent',
                serialized_type=float,
                op_getter='z_exponent',
            ),
            op_serializer.SerializingArg(
                serialized_name='axis_phase_exponent',
                serialized_type=float,
                op_getter='axis_phase_exponent',
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
                default=0.0,
            ),
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent',
                default=1.0,
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
                default=1.0,
            ),
        ],
        op_wrapper=lambda op, proto: _convert_physical_z(op, proto)),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xyz',
        gate_constructor=ops.PhasedXZGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='x_exponent',
                constructor_arg_name='x_exponent',
                default=0.0,
            ),
            op_deserializer.DeserializingArg(
                serialized_name='z_exponent',
                constructor_arg_name='z_exponent',
                default=0.0,
            ),
            op_deserializer.DeserializingArg(
                serialized_name='axis_phase_exponent',
                constructor_arg_name='axis_phase_exponent',
                default=0.0,
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
                                     op_getter=protocols.measurement_key),
        op_serializer.SerializingArg(serialized_name='invert_mask',
                                     serialized_type=List[bool],
                                     op_getter='invert_mask'),
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
                                         op_getter='phase_exponent'),
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.PhasedXPowGate, op.gate).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: (cast(ops.XPowGate, op.gate).exponent - 1
                                     ) / 2)
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.XPowGate, op.gate).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(ops.YPowGate, op.gate).exponent / 2)
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.YPowGate, op.gate).exponent, 1)),
    op_serializer.GateOpSerializer(
        gate_type=ops.XPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(ops.XPowGate, op.gate).exponent - 0.5)
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.XPowGate, op.gate).exponent, 0.5)),
    op_serializer.GateOpSerializer(
        gate_type=ops.YPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(ops.YPowGate, op.gate).exponent)
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.YPowGate, op.gate).exponent, 0.5)),
    op_serializer.GateOpSerializer(
        gate_type=ops.PhasedXPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         op_getter='phase_exponent'),
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.PhasedXPowGate, op.gate).exponent, 0.5)),
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
                constructor_arg_name='phase_exponent',
            ),
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

# Only CZ
CZ_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.CZPowGate,
    serialized_gate_id='cz',
    args=[
        op_serializer.SerializingArg(serialized_name='half_turns',
                                     serialized_type=float,
                                     op_getter='exponent')
    ],
    can_serialize_predicate=lambda op: _near_mod_2(
        cast(ops.CZPowGate, op.gate).exponent, 1.0))

# CZ to any power
CZ_POW_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.CZPowGate,
    serialized_gate_id='cz',
    args=[
        op_serializer.SerializingArg(serialized_name='half_turns',
                                     serialized_type=float,
                                     op_getter='exponent')
    ])

CZ_POW_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='cz',
    gate_constructor=ops.CZPowGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
            default=1.0,
        )
    ])

#
# Sycamore Gate Serializer and deserializer
#
SYC_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=ops.FSimGate,
    serialized_gate_id='syc',
    args=[],
    can_serialize_predicate=(
        lambda op: _near_mod_2pi(cast(ops.FSimGate, op.gate).theta, np.pi / 2)
        and _near_mod_2pi(cast(ops.FSimGate, op.gate).phi, np.pi / 6)))

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
        can_serialize_predicate=(lambda op: _near_mod_2pi(
            cast(ops.FSimGate, op.gate).theta, np.pi / 4) and _near_mod_2pi(
                cast(ops.FSimGate, op.gate).phi, 0))),
    op_serializer.GateOpSerializer(
        gate_type=ops.ISwapPowGate,
        serialized_gate_id='fsim_pi_4',
        args=[],
        can_serialize_predicate=(lambda op: _near_mod_n(
            cast(ops.ISwapPowGate, op.gate).exponent, -0.5, 4))),
    op_serializer.GateOpSerializer(
        gate_type=ops.FSimGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[],
        can_serialize_predicate=(lambda op: _near_mod_2pi(
            cast(ops.FSimGate, op.gate).theta, -np.pi / 4) and _near_mod_2pi(
                cast(ops.FSimGate, op.gate).phi, 0))),
    op_serializer.GateOpSerializer(
        gate_type=ops.ISwapPowGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[],
        can_serialize_predicate=(lambda op: _near_mod_n(
            cast(ops.ISwapPowGate, op.gate).exponent, +0.5, 4))),
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
# FSim serializer
# Only allows sqrt_iswap, its inverse, identity, and sycamore
#
def _can_serialize_limited_fsim(theta: float, phi: float):
    # Symbols for LIMITED_FSIM are allowed, but may fail server-side
    # if an incorrect run context is specified
    if _near_mod_2pi(phi, 0) or isinstance(phi, sympy.Symbol):
        if isinstance(theta, sympy.Symbol):
            return True
        # Identity
        if _near_mod_2pi(theta, 0):
            return True
        # sqrt ISWAP
        if _near_mod_2pi(theta, -np.pi / 4):
            return True
        # inverse sqrt ISWAP
        if _near_mod_2pi(theta, np.pi / 4):
            return True
    # Sycamore
    if ((_near_mod_2pi(theta, np.pi / 2) or isinstance(theta, sympy.Symbol)) and
        (_near_mod_2pi(phi, np.pi / 6)) or isinstance(phi, sympy.Symbol)):
        return True
    # CZ
    if ((_near_mod_2pi(theta, 0) or isinstance(theta, sympy.Symbol)) and
        (_near_mod_2pi(phi, np.pi)) or isinstance(phi, sympy.Symbol)):
        return True
    return False


def _can_serialize_limited_iswap(exponent: float):
    # Symbols for LIMITED_FSIM are allowed, but may fail server-side
    # if an incorrect run context is specified
    if isinstance(exponent, sympy.Symbol):
        return True
    # Sqrt ISWAP
    if _near_mod_n(exponent, 0.5, 4):
        return True
    # Inverse Sqrt ISWAP
    if _near_mod_n(exponent, -0.5, 4):
        return True
    # Identity
    if _near_mod_n(exponent, 0.0, 4):
        return True
    return False


LIMITED_FSIM_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=ops.FSimGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(serialized_name='theta',
                                         serialized_type=float,
                                         op_getter='theta'),
            op_serializer.SerializingArg(serialized_name='phi',
                                         serialized_type=float,
                                         op_getter='phi')
        ],
        can_serialize_predicate=(lambda op: _can_serialize_limited_fsim(
            cast(ops.FSimGate, op.gate).theta,
            cast(ops.FSimGate, op.gate).phi))),
    op_serializer.GateOpSerializer(
        gate_type=ops.ISwapPowGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(
                serialized_name='theta',
                serialized_type=float,
                # Note that ISWAP ** 0.5 is Fsim(-pi/4,0)
                op_getter=(lambda op: cast(ops.ISwapPowGate, op.gate).exponent *
                           -np.pi / 2)),
            op_serializer.SerializingArg(serialized_name='phi',
                                         serialized_type=float,
                                         op_getter=lambda e: 0)
        ],
        can_serialize_predicate=(lambda op: _can_serialize_limited_iswap(
            cast(ops.ISwapPowGate, op.gate).exponent))),
    op_serializer.GateOpSerializer(
        gate_type=ops.CZPowGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(serialized_name='theta',
                                         serialized_type=float,
                                         op_getter=lambda e: 0),
            op_serializer.SerializingArg(serialized_name='phi',
                                         serialized_type=float,
                                         op_getter=lambda e: np.pi)
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(ops.CZPowGate, op.gate).exponent, 1.0))
]


LIMITED_FSIM_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='fsim',
    gate_constructor=ops.FSimGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='theta',
            constructor_arg_name='theta',
            default=0.0,
        ),
        op_deserializer.DeserializingArg(
            serialized_name='phi',
            constructor_arg_name='phi',
            default=0.0,
        ),
    ])


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
            op_getter=lambda op: cast(ops.WaitGate, op.gate
                                     ).duration.total_nanos()),
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
