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

import cirq
from cirq_google.api import v2
from cirq_google.experimental.ops import CouplerPulse
from cirq_google.ops import PhysicalZTag
from cirq_google.serialization import op_deserializer, op_serializer

# Type strings used in serialization for the two types of Z operations
PHYSICAL_Z = 'physical'
VIRTUAL_Z = 'virtual_propagates_forward'

# Strings used for phase matching args
PHASE_MATCH_PHYS_Z = 'phys_z'


# Default tolerance for differences in floating point
# Note that Google protocol buffers use floats
# which trigger a conversion from double precision to single precision
# This results in errors possibly up to 1e-6
# (23 bits for mantissa in single precision)
_DEFAULT_ATOL = 1e-6


def _near_mod_n(e, t, n, atol=_DEFAULT_ATOL):
    """Returns whether a value, e, translated by t, is equal to 0 mod n."""
    if isinstance(e, sympy.Symbol):
        return False
    return abs((e - t + 1) % n - 1) <= atol


def _near_mod_2pi(e, t, atol=_DEFAULT_ATOL):
    """Returns whether a value, e, translated by t, is equal to 0 mod 2 * pi."""
    return _near_mod_n(e, t, n=2 * np.pi, atol=atol)


def _near_mod_2(e, t, atol=_DEFAULT_ATOL):
    """Returns whether a value, e, translated by t, is equal to 0 mod 2."""
    return _near_mod_n(e, t, n=2, atol=atol)


def _convert_physical_z(op: cirq.Operation, proto: v2.program_pb2.Operation):
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
        gate_type=cirq.PhasedXPowGate,
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
        gate_type=cirq.XPowGate,
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
        gate_type=cirq.YPowGate,
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
        gate_type=cirq.ZPowGate,
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
                op_getter=lambda op: PHYSICAL_Z if PhysicalZTag() in op.tags else VIRTUAL_Z,
            ),
        ],
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.PhasedXZGate,
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
        gate_constructor=cirq.PhasedXPowGate,
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
        gate_constructor=cirq.ZPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='half_turns',
                constructor_arg_name='exponent',
                default=1.0,
            ),
        ],
        op_wrapper=lambda op, proto: _convert_physical_z(op, proto),
    ),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xyz',
        gate_constructor=cirq.PhasedXZGate,
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
    gate_type=cirq.MeasurementGate,
    serialized_gate_id='meas',
    args=[
        op_serializer.SerializingArg(
            serialized_name='key', serialized_type=str, op_getter=cirq.measurement_key_name
        ),
        op_serializer.SerializingArg(
            serialized_name='invert_mask', serialized_type=List[bool], op_getter='invert_mask'
        ),
    ],
)
MEASUREMENT_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='meas',
    gate_constructor=cirq.MeasurementGate,
    args=[
        op_deserializer.DeserializingArg(serialized_name='key', constructor_arg_name='key'),
        op_deserializer.DeserializingArg(
            serialized_name='invert_mask',
            constructor_arg_name='invert_mask',
            value_func=lambda x: tuple(cast(list, x)),
        ),
    ],
    num_qubits_param='num_qubits',
)


#
# Serializers for single qubit rotations confined to half-pi increments
#
SINGLE_QUBIT_HALF_PI_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=cirq.PhasedXPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns', serialized_type=float, op_getter='phase_exponent'
            ),
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(cirq.PhasedXPowGate, op.gate).exponent, 1
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: (cast(cirq.XPowGate, op.gate).exponent - 1) / 2,
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.XPowGate, op.gate).exponent, 1),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.YPowGate,
        serialized_gate_id='xy_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(cirq.YPowGate, op.gate).exponent / 2,
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.YPowGate, op.gate).exponent, 1),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(cirq.XPowGate, op.gate).exponent - 0.5,
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.XPowGate, op.gate).exponent, 0.5),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.YPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns',
                serialized_type=float,
                op_getter=lambda op: cast(cirq.YPowGate, op.gate).exponent,
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.YPowGate, op.gate).exponent, 0.5),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.PhasedXPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            op_serializer.SerializingArg(
                serialized_name='axis_half_turns', serialized_type=float, op_getter='phase_exponent'
            ),
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(cirq.PhasedXPowGate, op.gate).exponent, 0.5
        ),
    ),
]

#
# Deserializers for single qubit rotations confined to half-pi increments
#
SINGLE_QUBIT_HALF_PI_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy_pi',
        gate_constructor=cirq.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='phase_exponent',
            ),
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='exponent',
                value_func=lambda _: 1,
            ),
        ],
    ),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='xy_half_pi',
        gate_constructor=cirq.PhasedXPowGate,
        args=[
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns', constructor_arg_name='phase_exponent'
            ),
            op_deserializer.DeserializingArg(
                serialized_name='axis_half_turns',
                constructor_arg_name='exponent',
                value_func=lambda _: 0.5,
            ),
        ],
    ),
]

#############################################
#
# Two qubit serializers and deserializers
#
#############################################

_phase_match_arg = op_serializer.SerializingArg(
    serialized_name='phase_match',
    serialized_type=str,
    op_getter=lambda op: PHASE_MATCH_PHYS_Z if PhysicalZTag() in op.tags else None,
    required=False,
)


def _add_phase_match(op: cirq.Operation, proto: v2.program_pb2.Operation):
    if 'phase_match' in proto.args:
        if proto.args['phase_match'].arg_value.string_value == PHASE_MATCH_PHYS_Z:
            return op.with_tags(PhysicalZTag())
    return op


#
# CZ Serializer and deserializer
#

# Only CZ
CZ_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.CZPowGate,
    serialized_gate_id='cz',
    args=[
        op_serializer.SerializingArg(
            serialized_name='half_turns', serialized_type=float, op_getter='exponent'
        ),
        _phase_match_arg,
    ],
    can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.CZPowGate, op.gate).exponent, 1.0),
)

# CZ to any power
CZ_POW_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.CZPowGate,
    serialized_gate_id='cz',
    args=[
        op_serializer.SerializingArg(
            serialized_name='half_turns', serialized_type=float, op_getter='exponent'
        ),
        _phase_match_arg,
    ],
)

CZ_POW_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='cz',
    gate_constructor=cirq.CZPowGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
            default=1.0,
        ),
    ],
    op_wrapper=lambda op, proto: _add_phase_match(op, proto),
)

#
# Sycamore Gate Serializer and deserializer
#
SYC_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.FSimGate,
    serialized_gate_id='syc',
    args=[_phase_match_arg],
    can_serialize_predicate=(
        lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, np.pi / 2)
        and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, np.pi / 6)
    ),
)

SYC_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='syc',
    gate_constructor=lambda: cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
    args=[],
    op_wrapper=lambda op, proto: _add_phase_match(op, proto),
)

#
# sqrt(ISWAP) serializer and deserializer
# (e.g. ISWAP ** 0.5)
#
SQRT_ISWAP_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, np.pi / 4)
            and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, 0)
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_n(cast(cirq.ISwapPowGate, op.gate).exponent, -0.5, 4)
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, -np.pi / 4)
            and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, 0)
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_n(cast(cirq.ISwapPowGate, op.gate).exponent, +0.5, 4)
        ),
    ),
]

SQRT_ISWAP_DESERIALIZERS = [
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='fsim_pi_4',
        gate_constructor=lambda: cirq.FSimGate(theta=np.pi / 4, phi=0),
        args=[],
        op_wrapper=lambda op, proto: _add_phase_match(op, proto),
    ),
    op_deserializer.GateOpDeserializer(
        serialized_gate_id='inv_fsim_pi_4',
        gate_constructor=lambda: cirq.FSimGate(theta=-np.pi / 4, phi=0),
        args=[],
        op_wrapper=lambda op, proto: _add_phase_match(op, proto),
    ),
]


#
# FSim serializer
# Only allows iswap, sqrt_iswap and their inverses, iswap, CZ, identity, and sycamore
# Note that not all combinations may not be available on all processors
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
        # ISWAP
        if _near_mod_2pi(theta, -np.pi / 2):
            return True
        # Inverse ISWAP
        if _near_mod_2pi(theta, np.pi / 2):
            return True
    # Sycamore
    if (
        (_near_mod_2pi(theta, np.pi / 2) or isinstance(theta, sympy.Symbol))
        and (_near_mod_2pi(phi, np.pi / 6))
        or isinstance(phi, sympy.Symbol)
    ):
        return True
    # CZ
    if (
        (_near_mod_2pi(theta, 0) or isinstance(theta, sympy.Symbol))
        and (_near_mod_2pi(phi, np.pi))
        or isinstance(phi, sympy.Symbol)
    ):
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
    # ISWAP
    if _near_mod_n(exponent, -1.0, 4):
        return True
    # Inverse ISWAP
    if _near_mod_n(exponent, 1.0, 4):
        return True
    # Identity
    if _near_mod_n(exponent, 0.0, 4):
        return True
    return False


LIMITED_FSIM_SERIALIZERS = [
    op_serializer.GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(
                serialized_name='theta', serialized_type=float, op_getter='theta'
            ),
            op_serializer.SerializingArg(
                serialized_name='phi', serialized_type=float, op_getter='phi'
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=(
            lambda op: _can_serialize_limited_fsim(
                cast(cirq.FSimGate, op.gate).theta, cast(cirq.FSimGate, op.gate).phi
            )
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(
                serialized_name='theta',
                serialized_type=float,
                # Note that ISWAP ** 0.5 is Fsim(-pi/4,0)
                op_getter=(lambda op: cast(cirq.ISwapPowGate, op.gate).exponent * -np.pi / 2),
            ),
            op_serializer.SerializingArg(
                serialized_name='phi', serialized_type=float, op_getter=lambda e: 0
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=(
            lambda op: _can_serialize_limited_iswap(cast(cirq.ISwapPowGate, op.gate).exponent)
        ),
    ),
    op_serializer.GateOpSerializer(
        gate_type=cirq.CZPowGate,
        serialized_gate_id='fsim',
        args=[
            op_serializer.SerializingArg(
                serialized_name='theta', serialized_type=float, op_getter=lambda e: 0
            ),
            op_serializer.SerializingArg(
                serialized_name='phi', serialized_type=float, op_getter=lambda e: np.pi
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.CZPowGate, op.gate).exponent, 1.0),
    ),
]


LIMITED_FSIM_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='fsim',
    gate_constructor=cirq.FSimGate,
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
    ],
    op_wrapper=lambda op, proto: _add_phase_match(op, proto),
)

#############################################
#
# Miscellaneous serializers and deserializers
#
#############################################

#
# Coupler Pulse serializer and deserializer
#

COUPLER_PULSE_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=CouplerPulse,
    serialized_gate_id='coupler_pulse',
    args=[
        op_serializer.SerializingArg(
            serialized_name='coupling_mhz', serialized_type=float, op_getter='coupling_mhz'
        ),
        op_serializer.SerializingArg(
            serialized_name='hold_time_ns',
            serialized_type=float,
            op_getter=lambda op: cast(CouplerPulse, op.gate).hold_time.total_nanos(),
        ),
        op_serializer.SerializingArg(
            serialized_name='rise_time_ns',
            serialized_type=float,
            op_getter=lambda op: cast(CouplerPulse, op.gate).rise_time.total_nanos(),
        ),
        op_serializer.SerializingArg(
            serialized_name='padding_time_ns',
            serialized_type=float,
            op_getter=lambda op: cast(CouplerPulse, op.gate).padding_time.total_nanos(),
        ),
    ],
)
COUPLER_PULSE_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='coupler_pulse',
    gate_constructor=CouplerPulse,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='coupling_mhz',
            constructor_arg_name='coupling_mhz',
        ),
        op_deserializer.DeserializingArg(
            serialized_name='hold_time_ns',
            constructor_arg_name='hold_time',
            value_func=lambda nanos: cirq.Duration(
                nanos=cast(Union[int, float, sympy.Basic], nanos)
            ),
        ),
        op_deserializer.DeserializingArg(
            serialized_name='rise_time_ns',
            constructor_arg_name='rise_time',
            value_func=lambda nanos: cirq.Duration(
                nanos=cast(Union[int, float, sympy.Basic], nanos)
            ),
        ),
        op_deserializer.DeserializingArg(
            serialized_name='padding_time_ns',
            constructor_arg_name='padding_time',
            value_func=lambda nanos: cirq.Duration(
                nanos=cast(Union[int, float, sympy.Basic], nanos)
            ),
        ),
    ],
)

#
# WaitGate serializer and deserializer
#
WAIT_GATE_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.WaitGate,
    serialized_gate_id='wait',
    args=[
        op_serializer.SerializingArg(
            serialized_name='nanos',
            serialized_type=float,
            op_getter=lambda op: cast(cirq.WaitGate, op.gate).duration.total_nanos(),
        ),
    ],
)
WAIT_GATE_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='wait',
    gate_constructor=cirq.WaitGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='nanos',
            constructor_arg_name='duration',
            value_func=lambda nanos: cirq.Duration(
                nanos=cast(Union[int, float, sympy.Basic], nanos)
            ),
        )
    ],
    num_qubits_param='num_qubits',
)

#
# CircuitOperation serializer and deserializer
#
CIRCUIT_OP_SERIALIZER = op_serializer.CircuitOpSerializer()
CIRCUIT_OP_DESERIALIZER = op_deserializer.CircuitOpDeserializer()
