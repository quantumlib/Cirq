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

from cirq_google import api

from cirq_google.devices import (
    Bristlecone,
    Foxtail,
    SerializableDevice,
    Sycamore,
    Sycamore23,
    XmonDevice,
)

from cirq_google.engine import (
    Calibration,
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

from cirq_google.gate_sets import (
    XMON,
    FSIM_GATESET,
    SQRT_ISWAP_GATESET,
    SYC_GATESET,
    NAMED_GATESETS,
)

from cirq_google.line import (
    AnnealSequenceSearchStrategy,
    GreedySequenceSearchStrategy,
    line_on_device,
    LinePlacementStrategy,
)

from cirq_google.ops import (
    CalibrationTag,
    PhysicalZTag,
    SycamoreGate,
    SYC,
)

from cirq_google.optimizers import (
    ConvertToXmonGates,
    ConvertToSqrtIswapGates,
    ConvertToSycamoreGates,
    GateTabulation,
    optimized_for_sycamore,
    optimized_for_xmon,
)

from cirq_google.op_deserializer import (
    DeserializingArg,
    GateOpDeserializer,
)

from cirq_google.op_serializer import (
    GateOpSerializer,
    SerializingArg,
)

from cirq_google.serializable_gate_set import (
    SerializableGateSet,)

from cirq_google.experiments import (
    generate_boixo_2018_supremacy_circuits_v2,
    generate_boixo_2018_supremacy_circuits_v2_bristlecone,
    generate_boixo_2018_supremacy_circuits_v2_grid,
)