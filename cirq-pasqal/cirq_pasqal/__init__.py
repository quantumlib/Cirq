# Copyright 2020 The Cirq Developers
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

"""Devices, qubits, and sampler for Pasqal's neutral atom device."""

from cirq_pasqal.pasqal_qubits import (
    ThreeDQubit,
    TwoDQubit,
)

from cirq_pasqal.pasqal_device import (
    PasqalDevice,
    PasqalVirtualDevice,
)

from cirq_pasqal.pasqal_noise_model import (
    PasqalNoiseModel,
)

from cirq_pasqal.pasqal_sampler import (
    PasqalSampler,
)


def _register_resolver() -> None:
    """Registers the cirq_google's public classes for JSON serialization."""
    from cirq.protocols.json_serialization import _internal_register_resolver
    from cirq_pasqal.json_resolver_cache import _class_resolver_dictionary

    _internal_register_resolver(_class_resolver_dictionary)


_register_resolver()
