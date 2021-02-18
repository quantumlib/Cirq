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

from cirq.google import api

from cirq.google.arg_func_langs import (
    arg_from_proto,
)

from cirq.google.calibration import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    CircuitWithCalibration,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    FSimPhaseCorrections,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    PhasedFSimEngineSimulator,
    SQRT_ISWAP_PARAMETERS,
    THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    make_zeta_chi_gamma_compensation_for_moments,
    merge_matching_results,
    prepare_floquet_characterization_for_moments,
    prepare_floquet_characterization_for_moment,
    run_calibrations,
    run_floquet_characterization_for_moments,
    run_zeta_chi_gamma_compensation_for_moments,
    try_convert_sqrt_iswap_to_fsim,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
)

from cirq.google.devices import (
    Bristlecone,
    Foxtail,
    SerializableDevice,
    Sycamore,
    Sycamore23,
    XmonDevice,
)

from cirq.google.engine import (
    Calibration,
    CalibrationLayer,
    CalibrationResult,
    Engine,
    engine_from_environment,
    EngineJob,
    EngineProgram,
    EngineProcessor,
    EngineTimeSlot,
    ProtoVersion,
    QuantumEngineSampler,
    get_engine,
    get_engine_calibration,
    get_engine_device,
    get_engine_sampler,
)

from cirq.google.gate_sets import (
    XMON,
    FSIM_GATESET,
    SQRT_ISWAP_GATESET,
    SYC_GATESET,
    NAMED_GATESETS,
)

from cirq.google.line import (
    AnnealSequenceSearchStrategy,
    GreedySequenceSearchStrategy,
    line_on_device,
    LinePlacementStrategy,
)

from cirq.google.ops import (
    CalibrationTag,
    PhysicalZTag,
    SycamoreGate,
    SYC,
)

from cirq.google.optimizers import (
    ConvertToXmonGates,
    ConvertToSqrtIswapGates,
    ConvertToSycamoreGates,
    GateTabulation,
    optimized_for_sycamore,
    optimized_for_xmon,
)

from cirq.google.op_deserializer import (
    DeserializingArg,
    GateOpDeserializer,
)

from cirq.google.op_serializer import (
    GateOpSerializer,
    SerializingArg,
)

from cirq.google.serializable_gate_set import (
    SerializableGateSet,
)

from cirq.google import experimental


def _register_resolver() -> None:
    """Registers the cirq.google's public classes for JSON serialization."""
    from cirq.protocols.json_serialization import _internal_register_resolver
    from cirq.google.json_resolver_cache import _class_resolver_dictionary

    _internal_register_resolver(_class_resolver_dictionary)


_register_resolver()
