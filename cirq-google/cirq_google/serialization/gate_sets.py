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
from typing import cast

from cirq._doc import document
from cirq import _compat
from cirq_google.serialization import serializable_gate_set
from cirq_google.serialization.common_serializers import (
    _SINGLE_QUBIT_SERIALIZERS,
    _SINGLE_QUBIT_DESERIALIZERS,
    _SINGLE_QUBIT_HALF_PI_SERIALIZERS,
    _SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
    _MEASUREMENT_SERIALIZER,
    _MEASUREMENT_DESERIALIZER,
    _CZ_POW_SERIALIZER,
    _CZ_POW_DESERIALIZER,
    _SYC_SERIALIZER,
    _SYC_DESERIALIZER,
    _SQRT_ISWAP_SERIALIZERS,
    _SQRT_ISWAP_DESERIALIZERS,
    _LIMITED_FSIM_SERIALIZERS,
    _LIMITED_FSIM_DESERIALIZER,
    _COUPLER_PULSE_SERIALIZER,
    _COUPLER_PULSE_DESERIALIZER,
    _WAIT_GATE_SERIALIZER,
    _WAIT_GATE_DESERIALIZER,
    CIRCUIT_OP_SERIALIZER,
    CIRCUIT_OP_DESERIALIZER,
)

SYC_GATESET = cast(
    serializable_gate_set.SerializableGateSet,
    serializable_gate_set._SerializableGateSet(
        gate_set_name='sycamore',
        serializers=[
            _SYC_SERIALIZER,
            *_SINGLE_QUBIT_SERIALIZERS,
            *_SINGLE_QUBIT_HALF_PI_SERIALIZERS,
            _MEASUREMENT_SERIALIZER,
            _WAIT_GATE_SERIALIZER,
            CIRCUIT_OP_SERIALIZER,
        ],
        deserializers=[
            _SYC_DESERIALIZER,
            *_SINGLE_QUBIT_DESERIALIZERS,
            *_SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
            _MEASUREMENT_DESERIALIZER,
            _WAIT_GATE_DESERIALIZER,
            CIRCUIT_OP_DESERIALIZER,
        ],
    ),
)
document(SYC_GATESET, """Gate set with fsim(pi/2, pi/6) as the core 2 qubit interaction.""")

SQRT_ISWAP_GATESET = cast(
    serializable_gate_set.SerializableGateSet,
    serializable_gate_set._SerializableGateSet(
        gate_set_name='sqrt_iswap',
        serializers=[
            *_SQRT_ISWAP_SERIALIZERS,
            *_SINGLE_QUBIT_SERIALIZERS,
            _MEASUREMENT_SERIALIZER,
            _WAIT_GATE_SERIALIZER,
            CIRCUIT_OP_SERIALIZER,
        ],
        deserializers=[
            *_SQRT_ISWAP_DESERIALIZERS,
            *_SINGLE_QUBIT_DESERIALIZERS,
            _MEASUREMENT_DESERIALIZER,
            _WAIT_GATE_DESERIALIZER,
            CIRCUIT_OP_DESERIALIZER,
        ],
    ),
)
document(SQRT_ISWAP_GATESET, """Gate set with sqrt(iswap) as the core 2 qubit interaction.""")


FSIM_GATESET = cast(
    serializable_gate_set.SerializableGateSet,
    serializable_gate_set._SerializableGateSet(
        gate_set_name='fsim',
        serializers=[
            *_LIMITED_FSIM_SERIALIZERS,
            *_SINGLE_QUBIT_SERIALIZERS,
            _MEASUREMENT_SERIALIZER,
            _WAIT_GATE_SERIALIZER,
            CIRCUIT_OP_SERIALIZER,
        ],
        deserializers=[
            _LIMITED_FSIM_DESERIALIZER,
            *_SINGLE_QUBIT_DESERIALIZERS,
            _MEASUREMENT_DESERIALIZER,
            _WAIT_GATE_DESERIALIZER,
            CIRCUIT_OP_DESERIALIZER,
        ],
    ),
)
document(FSIM_GATESET, """Gate set that combines sqrt(iswap) and syc as one fsim id.""")


EXPERIMENTAL_PULSE_GATESET = cast(
    serializable_gate_set.SerializableGateSet,
    serializable_gate_set._SerializableGateSet(
        gate_set_name='pulse',
        serializers=[
            _COUPLER_PULSE_SERIALIZER,
            *_LIMITED_FSIM_SERIALIZERS,
            *_SINGLE_QUBIT_SERIALIZERS,
            _MEASUREMENT_SERIALIZER,
            _WAIT_GATE_SERIALIZER,
            CIRCUIT_OP_SERIALIZER,
        ],
        deserializers=[
            _COUPLER_PULSE_DESERIALIZER,
            _LIMITED_FSIM_DESERIALIZER,
            *_SINGLE_QUBIT_DESERIALIZERS,
            _MEASUREMENT_DESERIALIZER,
            _WAIT_GATE_DESERIALIZER,
            CIRCUIT_OP_DESERIALIZER,
        ],
    ),
)
document(
    EXPERIMENTAL_PULSE_GATESET,
    "Experimental only.  Includes CouplerPulseGate with other fsim gates.",
)


# The xmon gate set.
XMON = cast(
    serializable_gate_set.SerializableGateSet,
    serializable_gate_set._SerializableGateSet(
        gate_set_name='xmon',
        serializers=[
            *_SINGLE_QUBIT_SERIALIZERS,
            _CZ_POW_SERIALIZER,
            _MEASUREMENT_SERIALIZER,
            CIRCUIT_OP_SERIALIZER,
        ],
        deserializers=[
            *_SINGLE_QUBIT_DESERIALIZERS,
            _CZ_POW_DESERIALIZER,
            _MEASUREMENT_DESERIALIZER,
            CIRCUIT_OP_DESERIALIZER,
        ],
    ),
)
document(XMON, """Gate set for XMON devices.""")

NAMED_GATESETS = {'sqrt_iswap': SQRT_ISWAP_GATESET, 'sycamore': SYC_GATESET, 'fsim': FSIM_GATESET}

document(NAMED_GATESETS, """A convenience mapping from gateset names to gatesets""")

GOOGLE_GATESETS = [SYC_GATESET, SQRT_ISWAP_GATESET, FSIM_GATESET, XMON]

document(GOOGLE_GATESETS, """All Google gatesets""")


_SERIALIZABLE_GATESET_DEPRECATION_MESSAGE = (
    'SerializableGateSet and associated classes (GateOpSerializer, GateOpDeserializer,'
    ' SerializingArgs, DeserializingArgs) will no longer be supported.'
    ' In cirq_google.GridDevice, the new representation of Google devices, the gateset of a device'
    ' is represented as a cirq.Gateset and is available as'
    ' GridDevice.metadata.gateset.'
    ' Engine methods no longer require gate sets to be passed in.'
    ' In addition, circuit serialization is replaced by cirq_google.CircuitSerializer.'
)


_compat.deprecate_attributes(
    __name__,
    {
        'EXPERIMENTAL_PULSE_GATESET': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
        'GOOGLE_GATESETS': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
    },
)
