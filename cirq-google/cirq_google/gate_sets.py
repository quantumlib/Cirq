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
from cirq._doc import document
from cirq_google import serializable_gate_set
from cirq_google.common_serializers import (
    SINGLE_QUBIT_SERIALIZERS,
    SINGLE_QUBIT_DESERIALIZERS,
    SINGLE_QUBIT_HALF_PI_SERIALIZERS,
    SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
    MEASUREMENT_SERIALIZER,
    MEASUREMENT_DESERIALIZER,
    CZ_POW_SERIALIZER,
    CZ_POW_DESERIALIZER,
    SYC_SERIALIZER,
    SYC_DESERIALIZER,
    SQRT_ISWAP_SERIALIZERS,
    SQRT_ISWAP_DESERIALIZERS,
    LIMITED_FSIM_SERIALIZERS,
    LIMITED_FSIM_DESERIALIZER,
    COUPLER_PULSE_SERIALIZER,
    COUPLER_PULSE_DESERIALIZER,
    WAIT_GATE_SERIALIZER,
    WAIT_GATE_DESERIALIZER,
    CIRCUIT_OP_SERIALIZER,
    CIRCUIT_OP_DESERIALIZER,
)

SYC_GATESET = serializable_gate_set.SerializableGateSet(
    gate_set_name='sycamore',
    serializers=[
        SYC_SERIALIZER,
        *SINGLE_QUBIT_SERIALIZERS,
        *SINGLE_QUBIT_HALF_PI_SERIALIZERS,
        MEASUREMENT_SERIALIZER,
        WAIT_GATE_SERIALIZER,
    ],
    deserializers=[
        SYC_DESERIALIZER,
        *SINGLE_QUBIT_DESERIALIZERS,
        *SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
        MEASUREMENT_DESERIALIZER,
        WAIT_GATE_DESERIALIZER,
    ],
)
document(SYC_GATESET, """Gate set with fsim(pi/2, pi/6) as the core 2 qubit interaction.""")


# TODO: merge with SYC_GATESET once server-side support is available.
SYC_GATESET_WITH_CIRCUITOP = SYC_GATESET.with_added_types(
    gate_set_name='sycamore_co',
    serializers=[CIRCUIT_OP_SERIALIZER],
    deserializers=[CIRCUIT_OP_DESERIALIZER],
)
document(SYC_GATESET_WITH_CIRCUITOP, """SYC_GATESET with CircuitOperation support.""")

SQRT_ISWAP_GATESET = serializable_gate_set.SerializableGateSet(
    gate_set_name='sqrt_iswap',
    serializers=[
        *SQRT_ISWAP_SERIALIZERS,
        *SINGLE_QUBIT_SERIALIZERS,
        MEASUREMENT_SERIALIZER,
        WAIT_GATE_SERIALIZER,
    ],
    deserializers=[
        *SQRT_ISWAP_DESERIALIZERS,
        *SINGLE_QUBIT_DESERIALIZERS,
        MEASUREMENT_DESERIALIZER,
        WAIT_GATE_DESERIALIZER,
    ],
)
document(SQRT_ISWAP_GATESET, """Gate set with sqrt(iswap) as the core 2 qubit interaction.""")


# TODO: merge with SQRT_ISWAP_GATESET once server-side support is available.
SQRT_ISWAP_GATESET_WITH_CIRCUITOP = SQRT_ISWAP_GATESET.with_added_types(
    gate_set_name='sqrt_iswap_co',
    serializers=[CIRCUIT_OP_SERIALIZER],
    deserializers=[CIRCUIT_OP_DESERIALIZER],
)
document(SQRT_ISWAP_GATESET_WITH_CIRCUITOP, """SQRT_ISWAP_GATESET with CircuitOperation support.""")


FSIM_GATESET = serializable_gate_set.SerializableGateSet(
    gate_set_name='fsim',
    serializers=[
        *LIMITED_FSIM_SERIALIZERS,
        *SINGLE_QUBIT_SERIALIZERS,
        MEASUREMENT_SERIALIZER,
        WAIT_GATE_SERIALIZER,
    ],
    deserializers=[
        LIMITED_FSIM_DESERIALIZER,
        *SINGLE_QUBIT_DESERIALIZERS,
        MEASUREMENT_DESERIALIZER,
        WAIT_GATE_DESERIALIZER,
    ],
)
document(FSIM_GATESET, """Gate set that combines sqrt(iswap) and syc as one fsim id.""")


# TODO: merge with FSIM_GATESET once server-side support is available.
FSIM_GATESET_WITH_CIRCUITOP = FSIM_GATESET.with_added_types(
    gate_set_name='fsim_co',
    serializers=[CIRCUIT_OP_SERIALIZER],
    deserializers=[CIRCUIT_OP_DESERIALIZER],
)
document(FSIM_GATESET_WITH_CIRCUITOP, """FSIM_GATESET with CircuitOperation support.""")


EXPERIMENTAL_PULSE_GATESET = serializable_gate_set.SerializableGateSet(
    gate_set_name='pulse',
    serializers=[
        COUPLER_PULSE_SERIALIZER,
        *LIMITED_FSIM_SERIALIZERS,
        *SINGLE_QUBIT_SERIALIZERS,
        MEASUREMENT_SERIALIZER,
        WAIT_GATE_SERIALIZER,
    ],
    deserializers=[
        COUPLER_PULSE_DESERIALIZER,
        LIMITED_FSIM_DESERIALIZER,
        *SINGLE_QUBIT_DESERIALIZERS,
        MEASUREMENT_DESERIALIZER,
        WAIT_GATE_DESERIALIZER,
    ],
)
document(
    EXPERIMENTAL_PULSE_GATESET,
    "Experimental only.  Includes CouplerPulseGate with other fsim gates.",
)


# The xmon gate set.
XMON = serializable_gate_set.SerializableGateSet(
    gate_set_name='xmon',
    serializers=[
        *SINGLE_QUBIT_SERIALIZERS,
        CZ_POW_SERIALIZER,
        MEASUREMENT_SERIALIZER,
    ],
    deserializers=[
        *SINGLE_QUBIT_DESERIALIZERS,
        CZ_POW_DESERIALIZER,
        MEASUREMENT_DESERIALIZER,
    ],
)
document(XMON, """Gate set for XMON devices.""")


# TODO: merge with XMON once server-side support is available.
XMON_WITH_CIRCUITOP = XMON.with_added_types(
    gate_set_name='xmon_co',
    serializers=[CIRCUIT_OP_SERIALIZER],
    deserializers=[CIRCUIT_OP_DESERIALIZER],
)
document(XMON_WITH_CIRCUITOP, """XMON with CircuitOperation support.""")

NAMED_GATESETS = {
    'sqrt_iswap': SQRT_ISWAP_GATESET,
    'sycamore': SYC_GATESET,
    'fsim': FSIM_GATESET,
}

document(NAMED_GATESETS, """A convenience mapping from gateset names to gatesets""")

GOOGLE_GATESETS = [
    SYC_GATESET,
    SQRT_ISWAP_GATESET,
    FSIM_GATESET,
    XMON,
]

document(GOOGLE_GATESETS, """All Google gatesets""")

GOOGLE_GATESETS_WITH_CIRCUITOP = [
    SYC_GATESET_WITH_CIRCUITOP,
    SQRT_ISWAP_GATESET_WITH_CIRCUITOP,
    FSIM_GATESET_WITH_CIRCUITOP,
    XMON_WITH_CIRCUITOP,
]

document(GOOGLE_GATESETS_WITH_CIRCUITOP, """All Google gatesets with subcircuit support""")
