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

"""Classes for working with Google's Quantum Engine API."""

import sys
from cirq_google import api

from cirq_google._version import __version__

from cirq_google.devices import (
    GoogleNoiseProperties,
    GridDevice,
    NoiseModelFromGoogleNoiseProperties,
    Sycamore,
    Sycamore23,
)

from cirq_google.engine import (
    Calibration,
    CalibrationLayer,
    CalibrationResult,
    Engine,
    EngineJob,
    EngineProgram,
    EngineProcessor,
    EngineResult,
    ProtoVersion,
    ProcessorSampler,
    ValidatingSampler,
    get_engine,
    get_engine_calibration,
    get_engine_device,
    get_engine_sampler,
    noise_properties_from_calibration,
)

from cirq_google.line import (
    AnnealSequenceSearchStrategy,
    GreedySequenceSearchStrategy,
    line_on_device,
    LinePlacementStrategy,
)

from cirq_google.ops import (
    CalibrationTag,
    FSimGateFamily,
    FSimViaModelTag,
    InternalGate,
    PhysicalZTag,
    SYC,
    SycamoreGate,
)

from cirq_google.transformers import (
    known_2q_op_to_sycamore_operations,
    two_qubit_matrix_to_sycamore_operations,
    GoogleCZTargetGateset,
    SycamoreTargetGateset,
)

from cirq_google.serialization import (
    arg_from_proto,
    CIRCUIT_SERIALIZER,
    CircuitSerializer,
    CircuitOpDeserializer,
    CircuitOpSerializer,
    Serializer,
)

from cirq_google.workflow import (
    ExecutableSpec,
    KeyValueExecutableSpec,
    QuantumExecutable,
    QuantumExecutableGroup,
    BitstringsMeasurement,
    SharedRuntimeInfo,
    RuntimeInfo,
    ExecutableResult,
    ExecutableGroupResult,
    ExecutableGroupResultFilesystemRecord,
    QuantumRuntimeConfiguration,
    execute,
    QubitPlacer,
    CouldNotPlaceError,
    NaiveQubitPlacer,
    RandomDevicePlacer,
    HardcodedQubitPlacer,
    ProcessorRecord,
    EngineProcessorRecord,
    SimulatedProcessorRecord,
    SimulatedProcessorWithLocalDeviceRecord,
)

from cirq_google import study

from cirq_google import experimental


# Register cirq_google's public classes for JSON serialization.
from cirq.protocols.json_serialization import _register_resolver
from cirq_google.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)
