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

"""Types for representing and methods for manipulating circuit operation trees.
"""

from cirq.ops.clifford_gate import (
    PauliTransform,
    SingleQubitCliffordGate,
)

from cirq.ops.common_channels import (
    amplitude_damp,
    AmplitudeDampingChannel,
    asymmetric_depolarize,
    AsymmetricDepolarizingChannel,
    bit_flip,
    BitFlipChannel,
    depolarize,
    DepolarizingChannel,
    generalized_amplitude_damp,
    GeneralizedAmplitudeDampingChannel,
    phase_damp,
    phase_flip,
    PhaseDampingChannel,
    PhaseFlipChannel,
)

from cirq.ops.common_gates import (
    CNOT,
    CNotPowGate,
    CX,
    CZ,
    CZPowGate,
    H,
    HPowGate,
    I,
    IdentityGate,
    ISWAP,
    ISwapPowGate,
    measure,
    measure_each,
    MeasurementGate,
    Rx,
    Ry,
    Rz,
    S,
    SWAP,
    SwapPowGate,
    T,
    XPowGate,
    YPowGate,
    ZPowGate,
)

from cirq.ops.controlled_gate import (
    ControlledGate,)

from cirq.ops.display import (
    ApproxPauliStringExpectation,
    DensityMatrixDisplay,
    pauli_string_expectation,
    PauliStringExpectation,
    SamplesDisplay,
    WaveFunctionDisplay,
)

from cirq.ops.eigen_gate import (
    EigenGate,)

from cirq.ops.fsim_gate import (
    FSimGate,)

from cirq.ops.gate_features import (
    InterchangeableQubitsGate,
    SingleQubitGate,
    ThreeQubitGate,
    TwoQubitGate,
)

from cirq.ops.gate_operation import (
    GateOperation,
    op_gate_of_type
)

from cirq.ops.linear_combinations import (
    LinearCombinationOfGates,)

from cirq.ops.parallel_gate_operation import (
    ParallelGateOperation,)

from cirq.ops.controlled_operation import (
    ControlledOperation,)

from cirq.ops.qubit_order import (
    QubitOrder,)

from cirq.ops.qubit_order_or_list import (
    QubitOrderOrList,)

from cirq.ops.matrix_gates import (
    SingleQubitMatrixGate,
    TwoQubitMatrixGate,
)

from cirq.ops.moment import (
    Moment,)

from cirq.ops.named_qubit import (
    NamedQubit,)

from cirq.ops.op_tree import (
    flatten_op_tree,
    freeze_op_tree,
    OP_TREE,
    transform_op_tree,
)

from cirq.ops.parity_gates import (
    XX,
    XXPowGate,
    YY,
    YYPowGate,
    ZZ,
    ZZPowGate,
)

from cirq.ops.pauli_gates import (
    Pauli,
    X,
    Y,
    Z,
)

from cirq.ops.pauli_interaction_gate import (
    PauliInteractionGate,)

from cirq.ops.pauli_string import (
    PauliString,
    SingleQubitPauliStringGateOperation,
)
from cirq.ops.pauli_string_phasor import (
    PauliStringPhasor,)
from cirq.ops.pauli_string_raw_types import (
    PauliStringGateOperation,)
from cirq.ops.phased_x_gate import (
    PhasedXPowGate,)

from cirq.ops.raw_types import (
    Gate,
    Operation,
    Qid,
)

from cirq.ops.three_qubit_gates import (
    CCNOT,
    CCX,
    CCXPowGate,
    CCZ,
    CCZPowGate,
    CSWAP,
    CSwapGate,
    FREDKIN,
    TOFFOLI,
)
