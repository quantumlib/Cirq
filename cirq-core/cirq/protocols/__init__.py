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

"""Protocols (structural subtyping) supported in Cirq."""

from cirq.protocols.act_on_protocol import (
    act_on as act_on,
    SupportsActOn as SupportsActOn,
    SupportsActOnQubits as SupportsActOnQubits,
)

from cirq.protocols.apply_unitary_protocol import (
    apply_unitaries as apply_unitaries,
    apply_unitary as apply_unitary,
    ApplyUnitaryArgs as ApplyUnitaryArgs,
    SupportsConsistentApplyUnitary as SupportsConsistentApplyUnitary,
)

from cirq.protocols.apply_channel_protocol import (
    apply_channel as apply_channel,
    ApplyChannelArgs as ApplyChannelArgs,
    SupportsApplyChannel as SupportsApplyChannel,
)

from cirq.protocols.apply_mixture_protocol import (
    apply_mixture as apply_mixture,
    ApplyMixtureArgs as ApplyMixtureArgs,
    SupportsApplyMixture as SupportsApplyMixture,
)

from cirq.protocols.approximate_equality_protocol import (
    approx_eq as approx_eq,
    SupportsApproximateEquality as SupportsApproximateEquality,
)

from cirq.protocols.kraus_protocol import (
    kraus as kraus,
    has_kraus as has_kraus,
    SupportsKraus as SupportsKraus,
)

from cirq.protocols.commutes_protocol import (
    commutes as commutes,
    definitely_commutes as definitely_commutes,
    SupportsCommutes as SupportsCommutes,
)

from cirq.protocols.control_key_protocol import (
    control_keys as control_keys,
    measurement_keys_touched as measurement_keys_touched,
    SupportsControlKey as SupportsControlKey,
)

from cirq.protocols.circuit_diagram_info_protocol import (
    circuit_diagram_info as circuit_diagram_info,
    CircuitDiagramInfo as CircuitDiagramInfo,
    CircuitDiagramInfoArgs as CircuitDiagramInfoArgs,
    LabelEntity as LabelEntity,
    SupportsCircuitDiagramInfo as SupportsCircuitDiagramInfo,
)

from cirq.protocols.decompose_protocol import (
    decompose as decompose,
    decompose_once as decompose_once,
    decompose_once_with_qubits as decompose_once_with_qubits,
    DecompositionContext as DecompositionContext,
    SupportsDecompose as SupportsDecompose,
    SupportsDecomposeWithQubits as SupportsDecomposeWithQubits,
)

from cirq.protocols.equal_up_to_global_phase_protocol import (
    equal_up_to_global_phase as equal_up_to_global_phase,
    SupportsEqualUpToGlobalPhase as SupportsEqualUpToGlobalPhase,
)

from cirq.protocols.has_stabilizer_effect_protocol import (
    has_stabilizer_effect as has_stabilizer_effect,
)

from cirq.protocols.has_unitary_protocol import (
    has_unitary as has_unitary,
    SupportsExplicitHasUnitary as SupportsExplicitHasUnitary,
)

from cirq.protocols.inverse_protocol import inverse as inverse

from cirq.protocols.json_serialization import (
    cirq_type_from_json as cirq_type_from_json,
    DEFAULT_RESOLVERS as DEFAULT_RESOLVERS,
    HasJSONNamespace as HasJSONNamespace,
    JsonResolver as JsonResolver,
    json_cirq_type as json_cirq_type,
    json_namespace as json_namespace,
    to_json_gzip as to_json_gzip,
    read_json_gzip as read_json_gzip,
    to_json as to_json,
    read_json as read_json,
    obj_to_dict_helper as obj_to_dict_helper,
    dataclass_json_dict as dataclass_json_dict,
    SerializableByKey as SerializableByKey,
    SupportsJSON as SupportsJSON,
)

from cirq.protocols.measurement_key_protocol import (
    is_measurement as is_measurement,
    measurement_key_name as measurement_key_name,
    measurement_key_obj as measurement_key_obj,
    measurement_key_names as measurement_key_names,
    measurement_key_objs as measurement_key_objs,
    with_key_path as with_key_path,
    with_key_path_prefix as with_key_path_prefix,
    with_measurement_key_mapping as with_measurement_key_mapping,
    with_rescoped_keys as with_rescoped_keys,
    SupportsMeasurementKey as SupportsMeasurementKey,
)

from cirq.protocols.mixture_protocol import (
    has_mixture as has_mixture,
    mixture as mixture,
    SupportsMixture as SupportsMixture,
    validate_mixture as validate_mixture,
)

from cirq.protocols.mul_protocol import mul as mul

from cirq.protocols.pauli_expansion_protocol import (
    pauli_expansion as pauli_expansion,
    SupportsPauliExpansion as SupportsPauliExpansion,
)

# pylint: disable=redefined-builtin
from cirq.protocols.pow_protocol import pow as pow

# pylint: enable=redefined-builtin

from cirq.protocols.qasm import (
    qasm as qasm,
    QasmArgs as QasmArgs,
    SupportsQasm as SupportsQasm,
    SupportsQasmWithArgs as SupportsQasmWithArgs,
    SupportsQasmWithArgsAndQubits as SupportsQasmWithArgsAndQubits,
)

from cirq.protocols.trace_distance_bound import (
    SupportsTraceDistanceBound as SupportsTraceDistanceBound,
    trace_distance_bound as trace_distance_bound,
    trace_distance_from_angle_list as trace_distance_from_angle_list,
)

from cirq.protocols.resolve_parameters import (
    is_parameterized as is_parameterized,
    parameter_names as parameter_names,
    parameter_symbols as parameter_symbols,
    resolve_parameters as resolve_parameters,
    resolve_parameters_once as resolve_parameters_once,
    SupportsParameterization as SupportsParameterization,
)

from cirq.protocols.phase_protocol import phase_by as phase_by, SupportsPhase as SupportsPhase

from cirq.protocols.qid_shape_protocol import (
    num_qubits as num_qubits,
    qid_shape as qid_shape,
    SupportsExplicitQidShape as SupportsExplicitQidShape,
    SupportsExplicitNumQubits as SupportsExplicitNumQubits,
)

from cirq.protocols.unitary_protocol import SupportsUnitary as SupportsUnitary, unitary as unitary
