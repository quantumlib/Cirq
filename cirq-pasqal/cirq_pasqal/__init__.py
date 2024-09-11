# Copyright 2021 The Cirq Developers
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

from cirq_pasqal._version import __version__ as __version__

from cirq_pasqal.pasqal_qubits import ThreeDQubit as ThreeDQubit, TwoDQubit as TwoDQubit

from cirq_pasqal.pasqal_gateset import PasqalGateset as PasqalGateset

from cirq_pasqal.pasqal_device import (
    PasqalDevice as PasqalDevice,
    PasqalVirtualDevice as PasqalVirtualDevice,
)

from cirq_pasqal.pasqal_noise_model import PasqalNoiseModel as PasqalNoiseModel

from cirq_pasqal.pasqal_sampler import PasqalSampler as PasqalSampler

# Register cirq_pasqal's public classes for JSON serialization.
from cirq.protocols.json_serialization import _register_resolver
from cirq_pasqal.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)
