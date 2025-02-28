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
"""Gates (unitary and non-unitary), operations, base types, and gate sets."""

from cirq.ops.arithmetic_operation import ArithmeticGate as ArithmeticGate

from cirq.ops.clifford_gate import (
    CliffordGate as CliffordGate,
    SingleQubitCliffordGate as SingleQubitCliffordGate,
)

from cirq.ops.dense_pauli_string import (
    BaseDensePauliString as BaseDensePauliString,
    DensePauliString as DensePauliString,
    MutableDensePauliString as MutableDensePauliString,
)

from cirq.ops.boolean_hamiltonian import BooleanHamiltonianGate as BooleanHamiltonianGate

from cirq.ops.common_channels import (
    amplitude_damp as amplitude_damp,
    AmplitudeDampingChannel as AmplitudeDampingChannel,
    asymmetric_depolarize as asymmetric_depolarize,
    AsymmetricDepolarizingChannel as AsymmetricDepolarizingChannel,
    bit_flip as bit_flip,
    BitFlipChannel as BitFlipChannel,
    depolarize as depolarize,
    DepolarizingChannel as DepolarizingChannel,
    generalized_amplitude_damp as generalized_amplitude_damp,
    GeneralizedAmplitudeDampingChannel as GeneralizedAmplitudeDampingChannel,
    phase_damp as phase_damp,
    phase_flip as phase_flip,
    PhaseDampingChannel as PhaseDampingChannel,
    PhaseFlipChannel as PhaseFlipChannel,
    R as R,
    reset as reset,
    reset_each as reset_each,
    ResetChannel as ResetChannel,
)

from cirq.ops.common_gates import (
    CNOT as CNOT,
    CNotPowGate as CNotPowGate,
    cphase as cphase,
    CX as CX,
    CXPowGate as CXPowGate,
    CZ as CZ,
    CZPowGate as CZPowGate,
    H as H,
    HPowGate as HPowGate,
    Rx as Rx,
    Ry as Ry,
    Rz as Rz,
    rx as rx,
    ry as ry,
    rz as rz,
    S as S,
    T as T,
    XPowGate as XPowGate,
    YPowGate as YPowGate,
    ZPowGate as ZPowGate,
)

from cirq.ops.common_gate_families import (
    AnyUnitaryGateFamily as AnyUnitaryGateFamily,
    AnyIntegerPowerGateFamily as AnyIntegerPowerGateFamily,
    ParallelGateFamily as ParallelGateFamily,
)

from cirq.ops.classically_controlled_operation import (
    ClassicallyControlledOperation as ClassicallyControlledOperation,
)

from cirq.ops.controlled_gate import ControlledGate as ControlledGate

from cirq.ops.diagonal_gate import DiagonalGate as DiagonalGate

from cirq.ops.eigen_gate import EigenGate as EigenGate

from cirq.ops.fourier_transform import (
    PhaseGradientGate as PhaseGradientGate,
    qft as qft,
    QuantumFourierTransformGate as QuantumFourierTransformGate,
)

from cirq.ops.fsim_gate import FSimGate as FSimGate, PhasedFSimGate as PhasedFSimGate

from cirq.ops.gate_features import InterchangeableQubitsGate as InterchangeableQubitsGate

from cirq.ops.gate_operation import GateOperation as GateOperation

from cirq.ops.gateset import GateFamily as GateFamily, Gateset as Gateset

from cirq.ops.identity import I as I, identity_each as identity_each, IdentityGate as IdentityGate

from cirq.ops.global_phase_op import (
    GlobalPhaseGate as GlobalPhaseGate,
    global_phase_operation as global_phase_operation,
)

from cirq.ops.kraus_channel import KrausChannel as KrausChannel

from cirq.ops.linear_combinations import (
    LinearCombinationOfGates as LinearCombinationOfGates,
    LinearCombinationOfOperations as LinearCombinationOfOperations,
    PauliSum as PauliSum,
    PauliSumLike as PauliSumLike,
    ProjectorSum as ProjectorSum,
)

from cirq.ops.mixed_unitary_channel import MixedUnitaryChannel as MixedUnitaryChannel

from cirq.ops.pauli_sum_exponential import PauliSumExponential as PauliSumExponential

from cirq.ops.pauli_measurement_gate import PauliMeasurementGate as PauliMeasurementGate

from cirq.ops.parallel_gate import (
    ParallelGate as ParallelGate,
    parallel_gate_op as parallel_gate_op,
)

from cirq.ops.projector import ProjectorString as ProjectorString

from cirq.ops.controlled_operation import ControlledOperation as ControlledOperation

from cirq.ops.qubit_manager import (
    BorrowableQubit as BorrowableQubit,
    CleanQubit as CleanQubit,
    QubitManager as QubitManager,
    SimpleQubitManager as SimpleQubitManager,
)

from cirq.ops.greedy_qubit_manager import GreedyQubitManager as GreedyQubitManager

from cirq.ops.qubit_order import QubitOrder as QubitOrder

from cirq.ops.qubit_order_or_list import QubitOrderOrList as QubitOrderOrList

from cirq.ops.matrix_gates import MatrixGate as MatrixGate

from cirq.ops.measure_util import (
    M as M,
    measure as measure,
    measure_each as measure_each,
    measure_paulistring_terms as measure_paulistring_terms,
    measure_single_paulistring as measure_single_paulistring,
)

from cirq.ops.measurement_gate import MeasurementGate as MeasurementGate

from cirq.ops.named_qubit import NamedQubit as NamedQubit, NamedQid as NamedQid

from cirq.ops.op_tree import (
    flatten_op_tree as flatten_op_tree,
    freeze_op_tree as freeze_op_tree,
    flatten_to_ops as flatten_to_ops,
    flatten_to_ops_or_moments as flatten_to_ops_or_moments,
    OP_TREE as OP_TREE,
    transform_op_tree as transform_op_tree,
)

from cirq.ops.parity_gates import (
    XX as XX,
    XXPowGate as XXPowGate,
    YY as YY,
    YYPowGate as YYPowGate,
    ZZ as ZZ,
    ZZPowGate as ZZPowGate,
    MSGate as MSGate,
    ms as ms,
)

from cirq.ops.pauli_gates import Pauli as Pauli, X as X, Y as Y, Z as Z

from cirq.ops.pauli_interaction_gate import PauliInteractionGate as PauliInteractionGate

from cirq.ops.pauli_string import (
    MutablePauliString as MutablePauliString,
    PAULI_GATE_LIKE as PAULI_GATE_LIKE,
    PAULI_STRING_LIKE as PAULI_STRING_LIKE,
    PauliString as PauliString,
    SingleQubitPauliStringGateOperation as SingleQubitPauliStringGateOperation,
)

from cirq.ops.pauli_string_phasor import (
    PauliStringPhasor as PauliStringPhasor,
    PauliStringPhasorGate as PauliStringPhasorGate,
)

from cirq.ops.pauli_string_raw_types import PauliStringGateOperation as PauliStringGateOperation

from cirq.ops.permutation_gate import QubitPermutationGate as QubitPermutationGate

from cirq.ops.phased_iswap_gate import givens as givens, PhasedISwapPowGate as PhasedISwapPowGate

from cirq.ops.phased_x_gate import PhasedXPowGate as PhasedXPowGate

from cirq.ops.phased_x_z_gate import PhasedXZGate as PhasedXZGate

from cirq.ops.qid_util import q as q

from cirq.ops.random_gate_channel import RandomGateChannel as RandomGateChannel

from cirq.ops.raw_types import (
    Gate as Gate,
    Operation as Operation,
    Qid as Qid,
    TaggedOperation as TaggedOperation,
)

from cirq.ops.swap_gates import (
    ISWAP as ISWAP,
    ISwapPowGate as ISwapPowGate,
    ISWAP_INV as ISWAP_INV,
    riswap as riswap,
    SQRT_ISWAP as SQRT_ISWAP,
    SQRT_ISWAP_INV as SQRT_ISWAP_INV,
    SWAP as SWAP,
    SwapPowGate as SwapPowGate,
)

from cirq.ops.tags import RoutingSwapTag as RoutingSwapTag, VirtualTag as VirtualTag

from cirq.ops.three_qubit_gates import (
    CCNOT as CCNOT,
    CCNotPowGate as CCNotPowGate,
    CCX as CCX,
    CCXPowGate as CCXPowGate,
    CCZ as CCZ,
    CCZPowGate as CCZPowGate,
    CSWAP as CSWAP,
    CSwapGate as CSwapGate,
    FREDKIN as FREDKIN,
    ThreeQubitDiagonalGate as ThreeQubitDiagonalGate,
    TOFFOLI as TOFFOLI,
)

from cirq.ops.two_qubit_diagonal_gate import TwoQubitDiagonalGate as TwoQubitDiagonalGate

from cirq.ops.wait_gate import wait as wait, WaitGate as WaitGate

from cirq.ops.state_preparation_channel import StatePreparationChannel as StatePreparationChannel

from cirq.ops.control_values import (
    AbstractControlValues as AbstractControlValues,
    ProductOfSums as ProductOfSums,
    SumOfProducts as SumOfProducts,
)

from cirq.ops.uniform_superposition_gate import UniformSuperpositionGate as UniformSuperpositionGate
