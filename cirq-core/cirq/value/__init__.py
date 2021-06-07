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

"""Value conversion utilities and types for time and quantum states."""
from cirq.value.abc_alt import (
    ABCMetaImplementAnyOneOf,
    alternative,
    GenericMetaImplementAnyOneOf,
)

from cirq.value.angle import (
    canonicalize_half_turns,
    chosen_angle_to_canonical_half_turns,
    chosen_angle_to_half_turns,
)

from cirq.value.digits import (
    big_endian_bits_to_int,
    big_endian_digits_to_int,
    big_endian_int_to_bits,
    big_endian_int_to_digits,
)

from cirq.value.duration import (
    Duration,
    DURATION_LIKE,
)

from cirq.value.linear_dict import (
    LinearDict,
    Scalar,
)

from cirq.value.measurement_key import (
    MEASUREMENT_KEY_SEPARATOR,
    MeasurementKey,
)

from cirq.value.probability import (
    validate_probability,
)

from cirq.value.product_state import (
    ProductState,
    KET_PLUS,
    KET_MINUS,
    KET_IMAG,
    KET_MINUS_IMAG,
    KET_ZERO,
    KET_ONE,
    PAULI_STATES,
)

from cirq.value.periodic_value import (
    PeriodicValue,
)

from cirq.value.random_state import (
    parse_random_state,
    RANDOM_STATE_OR_SEED_LIKE,
)

from cirq.value.timestamp import (
    Timestamp,
)

from cirq.value.type_alias import (
    TParamKey,
    TParamVal,
)

from cirq.value.value_equality_attr import (
    value_equality,
)
