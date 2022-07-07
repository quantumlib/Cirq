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
from cirq_google.ops import PhysicalZTag, SYC, fsim_gate_family
from cirq_google.serialization import op_deserializer, op_serializer

# Type strings used in serialization for the two types of Z operations
PHYSICAL_Z = 'physical'
VIRTUAL_Z = 'virtual_propagates_forward'

# Strings used for phase matching args
PHASE_MATCH_PHYS_Z = 'phys_z'


def _near_mod_n(e, t, n, atol=fsim_gate_family.DEFAULT_ATOL):
    """Returns whether a value, e, translated by t, is equal to 0 mod n."""
    if isinstance(e, sympy.Symbol):
        return False
    return abs((e - t + 1) % n - 1) <= atol


def _near_mod_2pi(e, t, atol=fsim_gate_family.DEFAULT_ATOL):
    """Returns whether a value, e, translated by t, is equal to 0 mod 2 * pi."""
    return _near_mod_n(e, t, n=2 * np.pi, atol=atol)


def _near_mod_2(e, t, atol=fsim_gate_family.DEFAULT_ATOL):
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
_SINGLE_QUBIT_SERIALIZERS = [
    op_serializer._GateOpSerializer(
        gate_type=cirq.PhasedXPowGate,
        serialized_gate_id='xy',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter='phase_exponent',
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='half_turns', serialized_type=float, op_getter='exponent'
                ),
            ),
        ],
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='xy',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: 0.0,
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='half_turns', serialized_type=float, op_getter='exponent'
                ),
            ),
        ],
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.YPowGate,
        serialized_gate_id='xy',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: 0.5,
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='half_turns', serialized_type=float, op_getter='exponent'
                ),
            ),
        ],
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.ZPowGate,
        serialized_gate_id='z',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='half_turns', serialized_type=float, op_getter='exponent'
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='type',
                    serialized_type=str,
                    op_getter=lambda op: PHYSICAL_Z if PhysicalZTag() in op.tags else VIRTUAL_Z,
                ),
            ),
        ],
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.PhasedXZGate,
        serialized_gate_id='xyz',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='x_exponent', serialized_type=float, op_getter='x_exponent'
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='z_exponent', serialized_type=float, op_getter='z_exponent'
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_phase_exponent',
                    serialized_type=float,
                    op_getter='axis_phase_exponent',
                ),
            ),
        ],
    ),
]

#
# Single qubit deserializers for arbitrary rotations
#
_SINGLE_QUBIT_DESERIALIZERS = [
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='xy',
        gate_constructor=cirq.PhasedXPowGate,
        args=[
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_half_turns',
                    constructor_arg_name='phase_exponent',
                    default=0.0,
                ),
            ),
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='half_turns', constructor_arg_name='exponent', default=1.0
                ),
            ),
        ],
    ),
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='z',
        gate_constructor=cirq.ZPowGate,
        args=[
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='half_turns', constructor_arg_name='exponent', default=1.0
                ),
            )
        ],
        op_wrapper=lambda op, proto: _convert_physical_z(op, proto),
    ),
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='xyz',
        gate_constructor=cirq.PhasedXZGate,
        args=[
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='x_exponent', constructor_arg_name='x_exponent', default=0.0
                ),
            ),
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='z_exponent', constructor_arg_name='z_exponent', default=0.0
                ),
            ),
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_phase_exponent',
                    constructor_arg_name='axis_phase_exponent',
                    default=0.0,
                ),
            ),
        ],
    ),
]


#
# Measurement Serializer and Deserializer
#
_MEASUREMENT_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=cirq.MeasurementGate,
    serialized_gate_id='meas',
    args=[
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='key', serialized_type=str, op_getter=cirq.measurement_key_name
            ),
        ),
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='invert_mask', serialized_type=List[bool], op_getter='invert_mask'
            ),
        ),
    ],
)
_MEASUREMENT_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='meas',
    gate_constructor=cirq.MeasurementGate,
    args=[
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(serialized_name='key', constructor_arg_name='key'),
        ),
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='invert_mask',
                constructor_arg_name='invert_mask',
                value_func=lambda x: tuple(cast(list, x)),
            ),
        ),
    ],
    num_qubits_param='num_qubits',
)


#
# Serializers for single qubit rotations confined to half-pi increments
#
_SINGLE_QUBIT_HALF_PI_SERIALIZERS = [
    op_serializer._GateOpSerializer(
        gate_type=cirq.PhasedXPowGate,
        serialized_gate_id='xy_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter='phase_exponent',
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(cirq.PhasedXPowGate, op.gate).exponent, 1
        ),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='xy_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: (cast(cirq.XPowGate, op.gate).exponent - 1) / 2,
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.XPowGate, op.gate).exponent, 1),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.YPowGate,
        serialized_gate_id='xy_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: cast(cirq.YPowGate, op.gate).exponent / 2,
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.YPowGate, op.gate).exponent, 1),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.XPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: cast(cirq.XPowGate, op.gate).exponent - 0.5,
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.XPowGate, op.gate).exponent, 0.5),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.YPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter=lambda op: cast(cirq.YPowGate, op.gate).exponent,
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.YPowGate, op.gate).exponent, 0.5),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.PhasedXPowGate,
        serialized_gate_id='xy_half_pi',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='axis_half_turns',
                    serialized_type=float,
                    op_getter='phase_exponent',
                ),
            )
        ],
        can_serialize_predicate=lambda op: _near_mod_2(
            cast(cirq.PhasedXPowGate, op.gate).exponent, 0.5
        ),
    ),
]

#
# Deserializers for single qubit rotations confined to half-pi increments
#
_SINGLE_QUBIT_HALF_PI_DESERIALIZERS = [
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='xy_pi',
        gate_constructor=cirq.PhasedXPowGate,
        args=[
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_half_turns', constructor_arg_name='phase_exponent'
                ),
            ),
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_half_turns',
                    constructor_arg_name='exponent',
                    value_func=lambda _: 1,
                ),
            ),
        ],
    ),
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='xy_half_pi',
        gate_constructor=cirq.PhasedXPowGate,
        args=[
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_half_turns', constructor_arg_name='phase_exponent'
                ),
            ),
            cast(
                op_deserializer.DeserializingArg,
                op_deserializer._DeserializingArg(
                    serialized_name='axis_half_turns',
                    constructor_arg_name='exponent',
                    value_func=lambda _: 0.5,
                ),
            ),
        ],
    ),
]

#############################################
#
# Two qubit serializers and deserializers
#
#############################################

_phase_match_arg = cast(
    op_serializer.SerializingArg,
    op_serializer._SerializingArg(
        serialized_name='phase_match',
        serialized_type=str,
        op_getter=lambda op: PHASE_MATCH_PHYS_Z if PhysicalZTag() in op.tags else None,
        required=False,
    ),
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
CZ_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=cirq.CZPowGate,
    serialized_gate_id='cz',
    args=[
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='half_turns', serialized_type=float, op_getter='exponent'
            ),
        ),
        _phase_match_arg,
    ],
    can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.CZPowGate, op.gate).exponent, 1.0),
)

# CZ to any power
_CZ_POW_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=cirq.CZPowGate,
    serialized_gate_id='cz',
    args=[
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='half_turns', serialized_type=float, op_getter='exponent'
            ),
        ),
        _phase_match_arg,
    ],
)

_CZ_POW_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='cz',
    gate_constructor=cirq.CZPowGate,
    args=[
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='half_turns', constructor_arg_name='exponent', default=1.0
            ),
        )
    ],
    op_wrapper=lambda op, proto: _add_phase_match(op, proto),
)

#
# Sycamore Gate Serializer and deserializer
#
_SYC_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=cirq.FSimGate,
    serialized_gate_id='syc',
    args=[_phase_match_arg],
    can_serialize_predicate=(
        lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, np.pi / 2)
        and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, np.pi / 6)
    ),
)

_SYC_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='syc',
    gate_constructor=lambda: cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
    args=[],
    op_wrapper=lambda op, proto: _add_phase_match(op, proto),
)

#
# sqrt(ISWAP) serializer and deserializer
# (e.g. ISWAP ** 0.5)
#
_SQRT_ISWAP_SERIALIZERS = [
    op_serializer._GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, np.pi / 4)
            and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, 0)
        ),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_n(cast(cirq.ISwapPowGate, op.gate).exponent, -0.5, 4)
        ),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_2pi(cast(cirq.FSimGate, op.gate).theta, -np.pi / 4)
            and _near_mod_2pi(cast(cirq.FSimGate, op.gate).phi, 0)
        ),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='inv_fsim_pi_4',
        args=[_phase_match_arg],
        can_serialize_predicate=(
            lambda op: _near_mod_n(cast(cirq.ISwapPowGate, op.gate).exponent, +0.5, 4)
        ),
    ),
]

_SQRT_ISWAP_DESERIALIZERS = [
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='fsim_pi_4',
        gate_constructor=lambda: cirq.FSimGate(theta=np.pi / 4, phi=0),
        args=[],
        op_wrapper=lambda op, proto: _add_phase_match(op, proto),
    ),
    op_deserializer._GateOpDeserializer(
        serialized_gate_id='inv_fsim_pi_4',
        gate_constructor=lambda: cirq.FSimGate(theta=-np.pi / 4, phi=0),
        args=[],
        op_wrapper=lambda op, proto: _add_phase_match(op, proto),
    ),
]

_LIMITED_FSIM_GATE_FAMILY = fsim_gate_family.FSimGateFamily(
    gates_to_accept=[
        cirq.IdentityGate(2),
        cirq.SQRT_ISWAP_INV,
        cirq.SQRT_ISWAP,
        cirq.ISWAP,
        cirq.ISWAP_INV,
        SYC,
        cirq.CZ,
    ],
    gate_types_to_check=[cirq.FSimGate],
    allow_symbols=True,
)
_LIMITED_ISWAP_GATE_FAMILY = fsim_gate_family.FSimGateFamily(
    gates_to_accept=[
        cirq.IdentityGate(2),
        cirq.SQRT_ISWAP_INV,
        cirq.SQRT_ISWAP,
        cirq.ISWAP,
        cirq.ISWAP_INV,
    ],
    gate_types_to_check=[cirq.ISwapPowGate],
    allow_symbols=True,
)
_LIMITED_FSIM_SERIALIZERS = [
    op_serializer._GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id='fsim',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='theta', serialized_type=float, op_getter='theta'
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='phi', serialized_type=float, op_getter='phi'
                ),
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=(lambda op: op in _LIMITED_FSIM_GATE_FAMILY),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.ISwapPowGate,
        serialized_gate_id='fsim',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='theta',
                    serialized_type=float,
                    # Note that ISWAP ** 0.5 is Fsim(-pi/4,0)
                    op_getter=(lambda op: cast(cirq.ISwapPowGate, op.gate).exponent * -np.pi / 2),
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='phi', serialized_type=float, op_getter=lambda e: 0
                ),
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=(lambda op: op in _LIMITED_ISWAP_GATE_FAMILY),
    ),
    op_serializer._GateOpSerializer(
        gate_type=cirq.CZPowGate,
        serialized_gate_id='fsim',
        args=[
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='theta', serialized_type=float, op_getter=lambda e: 0
                ),
            ),
            cast(
                op_serializer.SerializingArg,
                op_serializer._SerializingArg(
                    serialized_name='phi', serialized_type=float, op_getter=lambda e: np.pi
                ),
            ),
            _phase_match_arg,
        ],
        can_serialize_predicate=lambda op: _near_mod_2(cast(cirq.CZPowGate, op.gate).exponent, 1.0),
    ),
]


_LIMITED_FSIM_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='fsim',
    gate_constructor=cirq.FSimGate,
    args=[
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='theta', constructor_arg_name='theta', default=0.0
            ),
        ),
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='phi', constructor_arg_name='phi', default=0.0
            ),
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

_COUPLER_PULSE_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=CouplerPulse,
    serialized_gate_id='coupler_pulse',
    args=[
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='coupling_mhz', serialized_type=float, op_getter='coupling_mhz'
            ),
        ),
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='hold_time_ns',
                serialized_type=float,
                op_getter=lambda op: cast(CouplerPulse, op.gate).hold_time.total_nanos(),
            ),
        ),
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='rise_time_ns',
                serialized_type=float,
                op_getter=lambda op: cast(CouplerPulse, op.gate).rise_time.total_nanos(),
            ),
        ),
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='padding_time_ns',
                serialized_type=float,
                op_getter=lambda op: cast(CouplerPulse, op.gate).padding_time.total_nanos(),
            ),
        ),
    ],
)
_COUPLER_PULSE_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='coupler_pulse',
    gate_constructor=CouplerPulse,
    args=[
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='coupling_mhz', constructor_arg_name='coupling_mhz'
            ),
        ),
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='hold_time_ns',
                constructor_arg_name='hold_time',
                value_func=lambda nanos: cirq.Duration(
                    nanos=cast(Union[int, float, sympy.Expr], nanos)
                ),
            ),
        ),
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='rise_time_ns',
                constructor_arg_name='rise_time',
                value_func=lambda nanos: cirq.Duration(
                    nanos=cast(Union[int, float, sympy.Expr], nanos)
                ),
            ),
        ),
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='padding_time_ns',
                constructor_arg_name='padding_time',
                value_func=lambda nanos: cirq.Duration(
                    nanos=cast(Union[int, float, sympy.Expr], nanos)
                ),
            ),
        ),
    ],
)

#
# WaitGate serializer and deserializer
#
_WAIT_GATE_SERIALIZER = op_serializer._GateOpSerializer(
    gate_type=cirq.WaitGate,
    serialized_gate_id='wait',
    args=[
        cast(
            op_serializer.SerializingArg,
            op_serializer._SerializingArg(
                serialized_name='nanos',
                serialized_type=float,
                op_getter=lambda op: cast(cirq.WaitGate, op.gate).duration.total_nanos(),
            ),
        )
    ],
)
_WAIT_GATE_DESERIALIZER = op_deserializer._GateOpDeserializer(
    serialized_gate_id='wait',
    gate_constructor=cirq.WaitGate,
    args=[
        cast(
            op_deserializer.DeserializingArg,
            op_deserializer._DeserializingArg(
                serialized_name='nanos',
                constructor_arg_name='duration',
                value_func=lambda nanos: cirq.Duration(
                    nanos=cast(Union[int, float, sympy.Expr], nanos)
                ),
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


# HACK: to allow these to be used in gate_sets.py without throwing deprecation warnings during
# module load.
SINGLE_QUBIT_SERIALIZERS = _SINGLE_QUBIT_SERIALIZERS
SINGLE_QUBIT_DESERIALIZERS = _SINGLE_QUBIT_DESERIALIZERS
SINGLE_QUBIT_HALF_PI_SERIALIZERS = _SINGLE_QUBIT_HALF_PI_SERIALIZERS
SINGLE_QUBIT_HALF_PI_DESERIALIZERS = _SINGLE_QUBIT_HALF_PI_DESERIALIZERS
MEASUREMENT_SERIALIZER = _MEASUREMENT_SERIALIZER
MEASUREMENT_DESERIALIZER = _MEASUREMENT_DESERIALIZER
CZ_POW_SERIALIZER = _CZ_POW_SERIALIZER
CZ_POW_DESERIALIZER = _CZ_POW_DESERIALIZER
SYC_SERIALIZER = _SYC_SERIALIZER
SYC_DESERIALIZER = _SYC_DESERIALIZER
SQRT_ISWAP_SERIALIZERS = _SQRT_ISWAP_SERIALIZERS
SQRT_ISWAP_DESERIALIZERS = _SQRT_ISWAP_DESERIALIZERS
LIMITED_FSIM_SERIALIZERS = _LIMITED_FSIM_SERIALIZERS
LIMITED_FSIM_DESERIALIZER = _LIMITED_FSIM_DESERIALIZER
COUPLER_PULSE_SERIALIZER = _COUPLER_PULSE_SERIALIZER
COUPLER_PULSE_DESERIALIZER = _COUPLER_PULSE_DESERIALIZER
WAIT_GATE_SERIALIZER = _WAIT_GATE_SERIALIZER
WAIT_GATE_DESERIALIZER = _WAIT_GATE_DESERIALIZER


cirq._compat.deprecate_attributes(
    __name__,
    {
        'SINGLE_QUBIT_SERIALIZERS': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SINGLE_QUBIT_DESERIALIZERS': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SINGLE_QUBIT_HALF_PI_SERIALIZERS': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SINGLE_QUBIT_HALF_PI_DESERIALIZERS': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'MEASUREMENT_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'MEASUREMENT_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'CZ_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'CZ_POW_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'CZ_POW_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SYC_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SYC_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SQRT_ISWAP_SERIALIZERS': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'SQRT_ISWAP_DESERIALIZERS': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'LIMITED_FSIM_SERIALIZERS': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'LIMITED_FSIM_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'COUPLER_PULSE_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'COUPLER_PULSE_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'WAIT_GATE_SERIALIZER': (
            'v0.16',
            'GateOpSerializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
        'WAIT_GATE_DESERIALIZER': (
            'v0.16',
            'GateOpDeserializer will no longer be available.'
            ' CircuitSerializer will be the only supported circuit serializer going forward.',
        ),
    },
)
