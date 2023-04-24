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
from cirq import _compat
from cirq_google import api

from cirq_google._version import __version__

from cirq_google.calibration import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    CircuitWithCalibration,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    FSimPhaseCorrections,
    PhasedFSimCalibrationError,
    PhasedFSimCalibrationOptions,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    PhasedFSimEngineSimulator,
    XEBPhasedFSimCalibrationOptions,
    XEBPhasedFSimCalibrationRequest,
    LocalXEBPhasedFSimCalibrationOptions,
    LocalXEBPhasedFSimCalibrationRequest,
    SQRT_ISWAP_INV_PARAMETERS,
    THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    make_zeta_chi_gamma_compensation_for_moments,
    make_zeta_chi_gamma_compensation_for_operations,
    merge_matching_results,
    prepare_characterization_for_circuits_moments,
    prepare_floquet_characterization_for_moments,
    prepare_characterization_for_moments,
    prepare_floquet_characterization_for_moment,
    prepare_characterization_for_moment,
    prepare_floquet_characterization_for_operations,
    prepare_characterization_for_operations,
    run_calibrations,
    run_floquet_characterization_for_moments,
    run_zeta_chi_gamma_compensation_for_moments,
    try_convert_sqrt_iswap_to_fsim,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
)

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

from cirq_google.ops import CalibrationTag, FSimGateFamily, PhysicalZTag, SycamoreGate, SYC

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

from cirq_google import experimental


# Register cirq_google's public classes for JSON serialization.
from cirq.protocols.json_serialization import _register_resolver
from cirq_google.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)


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
        'XMON': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
        'FSIM_GATESET': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
        'SQRT_ISWAP_GATESET': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
        'SYC_GATESET': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
        'NAMED_GATESETS': ('v0.16', _SERIALIZABLE_GATESET_DEPRECATION_MESSAGE),
    },
)
