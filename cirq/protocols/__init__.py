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


from cirq.protocols.apply_unitary import (
    apply_unitary,
    ApplyUnitaryArgs,
    SupportsApplyUnitary,
)
from cirq.protocols.approximate_equality import (
    approx_eq,
    SupportsApproximateEquality,
)
from cirq.protocols.channel import (
    channel,
    has_channel,
    SupportsChannel,
)
from cirq.protocols.circuit_diagram_info import (
    CircuitDiagramInfo,
    CircuitDiagramInfoArgs,
    circuit_diagram_info,
    SupportsCircuitDiagramInfo,
)
from cirq.protocols.decompose import (
    decompose,
    decompose_once,
    decompose_once_with_qubits,
    SupportsDecompose,
    SupportsDecomposeWithQubits,
)
from cirq.protocols.inverse import (
    inverse,
)
from cirq.protocols.measurement_key import (
    measurement_key,
)
from cirq.protocols.mixture import (
    mixture,
    mixture_channel,
    has_mixture,
    has_mixture_channel,
    SupportsMixture,
    validate_mixture,
)
from cirq.protocols.mul import (
    mul,
)
from cirq.protocols.pauli_expansion import (
    pauli_expansion,
)
# pylint: disable=redefined-builtin
from cirq.protocols.pow import (
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
from cirq.protocols.trace_distance_bound import (
    SupportsTraceDistanceBound,
    trace_distance_bound,
)
from cirq.protocols.resolve_parameters import (
    SupportsParameterization,
    is_parameterized,
    resolve_parameters,
)
from cirq.protocols.phase import (
    SupportsPhase,
    phase_by,
)
from cirq.protocols.unitary import (
    SupportsUnitary,
    has_unitary,
    unitary,
)
