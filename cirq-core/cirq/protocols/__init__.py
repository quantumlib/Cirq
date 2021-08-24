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

"""Methods and classes that define cirq's protocols."""

from cirq.protocols.act_on_protocol import (
    act_on,
    SupportsActOn,
    SupportsActOnQubits,
)
from cirq.protocols.apply_unitary_protocol import (
    apply_unitaries,
    apply_unitary,
    ApplyUnitaryArgs,
    SupportsConsistentApplyUnitary,
)
from cirq.protocols.apply_channel_protocol import (
    apply_channel,
    ApplyChannelArgs,
    SupportsApplyChannel,
)
from cirq.protocols.apply_mixture_protocol import (
    apply_mixture,
    ApplyMixtureArgs,
    SupportsApplyMixture,
)
from cirq.protocols.approximate_equality_protocol import (
    approx_eq,
    SupportsApproximateEquality,
)
from cirq.protocols.kraus_protocol import (
    channel,
    kraus,
    has_channel,
    has_kraus,
    SupportsChannel,
    SupportsKraus,
)
from cirq.protocols.commutes_protocol import (
    commutes,
    definitely_commutes,
    SupportsCommutes,
)
from cirq.protocols.circuit_diagram_info_protocol import (
    circuit_diagram_info,
    CircuitDiagramInfo,
    CircuitDiagramInfoArgs,
    SupportsCircuitDiagramInfo,
)
from cirq.protocols.decompose_protocol import (
    decompose,
    decompose_once,
    decompose_once_with_qubits,
    SupportsDecompose,
    SupportsDecomposeWithQubits,
)
from cirq.protocols.equal_up_to_global_phase_protocol import (
    equal_up_to_global_phase,
    SupportsEqualUpToGlobalPhase,
)
from cirq.protocols.has_stabilizer_effect_protocol import (
    has_stabilizer_effect,
)
from cirq.protocols.has_unitary_protocol import (
    has_unitary,
    SupportsExplicitHasUnitary,
)
from cirq.protocols.inverse_protocol import (
    inverse,
)
from cirq.protocols.json_serialization import (
    DEFAULT_RESOLVERS,
    JsonResolver,
    json_serializable_dataclass,
    to_json_gzip,
    read_json_gzip,
    to_json,
    read_json,
    obj_to_dict_helper,
    dataclass_json_dict,
    SerializableByKey,
    SupportsJSON,
)
from cirq.protocols.measurement_key_protocol import (
    is_measurement,
    measurement_key,
    measurement_key_name,
    measurement_keys,
    measurement_key_names,
    with_key_path,
    with_measurement_key_mapping,
    SupportsMeasurementKey,
)
from cirq.protocols.mixture_protocol import (
    has_mixture,
    mixture,
    SupportsMixture,
    validate_mixture,
)
from cirq.protocols.mul_protocol import (
    mul,
)
from cirq.protocols.pauli_expansion_protocol import (
    pauli_expansion,
    SupportsPauliExpansion,
)

# pylint: disable=redefined-builtin
from cirq.protocols.pow_protocol import (
    pow,
)

# pylint: enable=redefined-builtin
from cirq.protocols.qasm import (
    qasm,
    QasmArgs,
    SupportsQasm,
    SupportsQasmWithArgs,
    SupportsQasmWithArgsAndQubits,
)
from cirq.protocols.quil import quil, QuilFormatter
from cirq.protocols.trace_distance_bound import (
    SupportsTraceDistanceBound,
    trace_distance_bound,
    trace_distance_from_angle_list,
)
from cirq.protocols.resolve_parameters import (
    is_parameterized,
    parameter_names,
    parameter_symbols,
    resolve_parameters,
    resolve_parameters_once,
    SupportsParameterization,
)
from cirq.protocols.phase_protocol import (
    phase_by,
    SupportsPhase,
)
from cirq.protocols.qid_shape_protocol import (
    num_qubits,
    qid_shape,
    SupportsExplicitQidShape,
    SupportsExplicitNumQubits,
)
from cirq.protocols.unitary_protocol import (
    SupportsUnitary,
    unitary,
)
