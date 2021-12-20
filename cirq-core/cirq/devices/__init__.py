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

"""Types for devices, device-specific qubits, and noise models."""
from cirq.devices.device import (
    Device,
    SymmetricalQidPair,
)

from cirq.devices.grid_qubit import (
    GridQid,
    GridQubit,
)

from cirq.devices.line_qubit import (
    LineQubit,
    LineQid,
)

from cirq.devices.unconstrained_device import (
    UNCONSTRAINED_DEVICE,
)

from cirq.devices.noise_model import (
    NO_NOISE,
    NOISE_MODEL_LIKE,
    NoiseModel,
    ConstantQubitNoiseModel,
)

from cirq.devices.named_topologies import (
    NamedTopology,
    draw_gridlike,
    LineTopology,
    TiltedSquareLattice,
    get_placements,
    draw_placements,
)

from cirq.devices.insertion_noise_model import (
    InsertionNoiseModel,
)

from cirq.devices.noise_utils import (
    OpIdentifier,
    decay_constant_to_xeb_fidelity,
    decay_constant_to_pauli_error,
    pauli_error_to_decay_constant,
    xeb_fidelity_to_decay_constant,
    pauli_error_from_t1,
    pauli_error_from_depolarization,
    average_error,
    decoherence_pauli_error,
)
