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

from cirq_google._version import __version__ as __version__

from cirq_google.devices import (
    GoogleNoiseProperties as GoogleNoiseProperties,
    GridDevice as GridDevice,
    NoiseModelFromGoogleNoiseProperties as NoiseModelFromGoogleNoiseProperties,
    Sycamore as Sycamore,
    Sycamore23 as Sycamore23,
)

from cirq_google.engine import (
    Calibration as Calibration,
    CalibrationLayer as CalibrationLayer,
    CalibrationResult as CalibrationResult,
    Engine as Engine,
    EngineJob as EngineJob,
    EngineProgram as EngineProgram,
    EngineProcessor as EngineProcessor,
    EngineResult as EngineResult,
    ProtoVersion as ProtoVersion,
    ProcessorSampler as ProcessorSampler,
    ValidatingSampler as ValidatingSampler,
    get_engine as get_engine,
    get_engine_calibration as get_engine_calibration,
    get_engine_device as get_engine_device,
    get_engine_sampler as get_engine_sampler,
    noise_properties_from_calibration as noise_properties_from_calibration,
)

from cirq_google.line import (
    AnnealSequenceSearchStrategy as AnnealSequenceSearchStrategy,
    GreedySequenceSearchStrategy as GreedySequenceSearchStrategy,
    line_on_device as line_on_device,
    LinePlacementStrategy as LinePlacementStrategy,
)

from cirq_google.ops import (
    CalibrationTag as CalibrationTag,
    Coupler as Coupler,
    FSimGateFamily as FSimGateFamily,
    FSimViaModelTag as FSimViaModelTag,
    InternalGate as InternalGate,
    InternalTag as InternalTag,
    PhysicalZTag as PhysicalZTag,
    SYC as SYC,
    SycamoreGate as SycamoreGate,
    WILLOW as WILLOW,
    WillowGate as WillowGate,
)

from cirq_google.transformers import (
    known_2q_op_to_sycamore_operations as known_2q_op_to_sycamore_operations,
    two_qubit_matrix_to_sycamore_operations as two_qubit_matrix_to_sycamore_operations,
    GoogleCZTargetGateset as GoogleCZTargetGateset,
    SycamoreTargetGateset as SycamoreTargetGateset,
)

from cirq_google.serialization import (
    arg_from_proto as arg_from_proto,
    CIRCUIT_SERIALIZER as CIRCUIT_SERIALIZER,
    CircuitSerializer as CircuitSerializer,
    CircuitOpDeserializer as CircuitOpDeserializer,
    CircuitOpSerializer as CircuitOpSerializer,
    Serializer as Serializer,
)

from cirq_google.workflow import (
    ExecutableSpec as ExecutableSpec,
    KeyValueExecutableSpec as KeyValueExecutableSpec,
    QuantumExecutable as QuantumExecutable,
    QuantumExecutableGroup as QuantumExecutableGroup,
    BitstringsMeasurement as BitstringsMeasurement,
    SharedRuntimeInfo as SharedRuntimeInfo,
    RuntimeInfo as RuntimeInfo,
    ExecutableResult as ExecutableResult,
    ExecutableGroupResult as ExecutableGroupResult,
    ExecutableGroupResultFilesystemRecord as ExecutableGroupResultFilesystemRecord,
    QuantumRuntimeConfiguration as QuantumRuntimeConfiguration,
    execute as execute,
    QubitPlacer as QubitPlacer,
    CouldNotPlaceError as CouldNotPlaceError,
    NaiveQubitPlacer as NaiveQubitPlacer,
    RandomDevicePlacer as RandomDevicePlacer,
    HardcodedQubitPlacer as HardcodedQubitPlacer,
    ProcessorRecord as ProcessorRecord,
    EngineProcessorRecord as EngineProcessorRecord,
    SimulatedProcessorRecord as SimulatedProcessorRecord,
    SimulatedProcessorWithLocalDeviceRecord as SimulatedProcessorWithLocalDeviceRecord,
)

from cirq_google import study

from cirq_google import experimental


# Register cirq_google's public classes for JSON serialization.
from cirq.protocols.json_serialization import _register_resolver
from cirq_google.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)
