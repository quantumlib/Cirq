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

from cirq.ops.arithmetic_operation import (
    ArithmeticOperation,
)

from cirq.ops.clifford_gate import (
    PauliTransform,
    SingleQubitCliffordGate,
)

from cirq.ops.dense_pauli_string import (
    BaseDensePauliString,
    DensePauliString,
    MutableDensePauliString,
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
    reset,
    reset_each,
    ResetChannel,
)

from cirq.ops.common_gates import (
    CNOT,
    CNotPowGate,
    cphase,
    CX,
    CXPowGate,
    CZ,
    CZPowGate,
    H,
    HPowGate,
    Rx,
    Ry,
    Rz,
    rx,
    ry,
    rz,
    S,
    T,
    XPowGate,
    YPowGate,
    ZPowGate,
)

from cirq.ops.controlled_gate import (
    ControlledGate,
)

from cirq.ops.diagonal_gate import (
    DiagonalGate,
)

from cirq.ops.eigen_gate import (
    EigenGate,
)

from cirq.ops.fourier_transform import (
    PhaseGradientGate,
    qft,
    QuantumFourierTransformGate,
)

from cirq.ops.fsim_gate import (
    FSimGate,
    PhasedFSimGate,
)

from cirq.ops.gate_features import (
    InterchangeableQubitsGate,
    SingleQubitGate,
    ThreeQubitGate,
    TwoQubitGate,
)

from cirq.ops.gate_operation import (
    GateOperation,
)

from cirq.ops.identity import (
    I,
    identity_each,
    IdentityGate,
)

from cirq.ops.global_phase_op import (
    GlobalPhaseOperation,
)

from cirq.ops.linear_combinations import (
    LinearCombinationOfGates,
    LinearCombinationOfOperations,
    PauliSum,
    PauliSumLike,
)

from cirq.ops.pauli_sum_exponential import (
    PauliSumExponential,
)

from cirq.ops.parallel_gate_operation import (
    ParallelGateOperation,
)

from cirq.ops.controlled_operation import (
    ControlledOperation,
)

from cirq.ops.qubit_order import (
    QubitOrder,
)

from cirq.ops.qubit_order_or_list import (
    QubitOrderOrList,
)

from cirq.ops.matrix_gates import (
    MatrixGate,
)

from cirq.ops.measure_util import (
    measure,
    measure_each,
)

from cirq.ops.measurement_gate import (
    MeasurementGate,
)

from cirq.ops.moment import (
    Moment,
)

from cirq.ops.named_qubit import (
    NamedQubit,
    NamedQid,
)

from cirq.ops.op_tree import (
    flatten_op_tree,
    freeze_op_tree,
    flatten_to_ops,
    flatten_to_ops_or_moments,
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
    PauliInteractionGate,
)

from cirq.ops.pauli_string import (
    MutablePauliString,
    PAULI_GATE_LIKE,
    PAULI_STRING_LIKE,
    PauliString,
    SingleQubitPauliStringGateOperation,
)

from cirq.ops.pauli_string_phasor import (
    PauliStringPhasor,
)

from cirq.ops.pauli_string_raw_types import (
    PauliStringGateOperation,
)

from cirq.ops.permutation_gate import (
    QubitPermutationGate,
)

from cirq.ops.phased_iswap_gate import (
    givens,
    PhasedISwapPowGate,
)

from cirq.ops.phased_x_gate import (
    PhasedXPowGate,
)

from cirq.ops.phased_x_z_gate import (
    PhasedXZGate,
)

from cirq.ops.random_gate_channel import (
    RandomGateChannel,
)

from cirq.ops.raw_types import (
    Gate,
    Operation,
    Qid,
    TaggedOperation,
)

from cirq.ops.swap_gates import (
    ISWAP,
    ISwapPowGate,
    riswap,
    SQRT_ISWAP,
    SQRT_ISWAP_INV,
    SWAP,
    SwapPowGate,
)

from cirq.ops.tags import (
    VirtualTag,
)

from cirq.ops.three_qubit_gates import (
    CCNOT,
    CCNotPowGate,
    CCX,
    CCXPowGate,
    CCZ,
    CCZPowGate,
    CSWAP,
    CSwapGate,
    FREDKIN,
    ThreeQubitDiagonalGate,
    TOFFOLI,
)

from cirq.ops.two_qubit_diagonal_gate import (
    TwoQubitDiagonalGate,
)

from cirq.ops.wait_gate import (
    wait,
    WaitGate,
)
