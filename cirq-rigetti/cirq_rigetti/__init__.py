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

from cirq_rigetti._version import __version__ as __version__

from cirq_rigetti.sampler import (
    RigettiQCSSampler as RigettiQCSSampler,
    get_rigetti_qcs_sampler as get_rigetti_qcs_sampler,
)

from cirq_rigetti.service import (
    RigettiQCSService as RigettiQCSService,
    get_rigetti_qcs_service as get_rigetti_qcs_service,
)

from cirq_rigetti import circuit_sweep_executors, quil_output
from cirq_rigetti import circuit_transformers
from cirq_rigetti.aspen_device import (
    RigettiQCSAspenDevice as RigettiQCSAspenDevice,
    AspenQubit as AspenQubit,
    OctagonalQubit as OctagonalQubit,
    UnsupportedQubit as UnsupportedQubit,
    UnsupportedRigettiQCSOperation as UnsupportedRigettiQCSOperation,
    UnsupportedRigettiQCSQuantumProcessor as UnsupportedRigettiQCSQuantumProcessor,
)

from cirq_rigetti.quil_input import circuit_from_quil as circuit_from_quil

# Registers the cirq_rigetti's public classes for JSON serialization.
from cirq.protocols.json_serialization import _register_resolver
from cirq_rigetti.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)
