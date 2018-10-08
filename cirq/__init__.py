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

# Flattened sub-modules.

from cirq.circuits import (
    Circuit,
    CircuitDag,
    ConvertToCzAndSingleGates,
    DropEmptyMoments,
    DropNegligible,
    ExpandComposite,
    InsertStrategy,
    MergeInteractions,
    MergeSingleQubitGates,
    Moment,
    OptimizationPass,
    PointOptimizationSummary,
    PointOptimizer,
    QasmOutput,
    TextDiagramDrawer,
    Unique,
)

from cirq.decompositions import (
    controlled_op_to_operations,
    is_negligible_turn,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_op_to_framed_phase_form,
    two_qubit_matrix_to_operations,
)

from cirq.devices import (
    Device,
    GridQubit,
    UnconstrainedDevice,
)

from cirq.extension import (
    can_cast,
    cast,
    Extensions,
    PotentialImplementation,
    try_cast,
)

from cirq.linalg import (
    allclose_up_to_global_phase,
    apply_matrix_to_slices,
    bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals,
    slice_for_qubits_equal_to,
    block_diag,
    match_global_phase,
    commutes,
    CONTROL_TAG,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix,
    dot,
    is_diagonal,
    is_hermitian,
    is_orthogonal,
    is_special_orthogonal,
    is_special_unitary,
    is_unitary,
    kak_canonicalize_vector,
    kak_decomposition,
    kron,
    kron_factor_4x4_to_2x2s,
    kron_with_controls,
    map_eigenvalues,
    reflection_matrix_pow,
    so4_to_magic_su2s,
    targeted_left_multiply,
    Tolerance,
)

from cirq.line import (
    AnnealSequenceSearchStrategy,
    GreedySequenceSearchStrategy,
    LinePlacementStrategy,
    LineQubit,
    line_on_device,
)

from cirq.ops import (
    BoundedEffect,
    CCX,
    CCZ,
    SingleQubitCliffordGate,
    CNOT,
    CNotGate,
    CompositeGate,
    CompositeOperation,
    ControlledGate,
    CSWAP,
    CZ,
    EigenGate,
    ExtrapolatableEffect,
    flatten_op_tree,
    FREDKIN,
    freeze_op_tree,
    Gate,
    GateOperation,
    H,
    HGate,
    InterchangeableQubitsGate,
    ISWAP,
    ISwapGate,
    measure,
    measure_each,
    MeasurementGate,
    NamedQubit,
    OP_TREE,
    Operation,
    ParameterizableEffect,
    Pauli,
    PauliInteractionGate,
    PauliString,
    PauliTransform,
    PhaseableEffect,
    QasmConvertibleGate,
    QasmConvertibleOperation,
    QasmOutputArgs,
    QubitId,
    QubitOrder,
    QubitOrderOrList,
    ReversibleCompositeGate,
    ReversibleEffect,
    Rot11Gate,
    RotXGate,
    RotYGate,
    RotZGate,
    S,
    SingleQubitGate,
    SingleQubitMatrixGate,
    SWAP,
    SwapGate,
    T,
    TextDiagrammable,
    TextDiagramInfo,
    TextDiagramInfoArgs,
    ThreeQubitGate,
    TOFFOLI,
    transform_op_tree,
    TwoQubitGate,
    TwoQubitMatrixGate,
    X,
    Y,
    Z,
)

from cirq.schedules import (
    Schedule,
    ScheduledOperation,
    moment_by_moment_schedule,
)

from cirq.sim import (
    check_state,
    decode_initial_state,
    pretty_state,
    RunSimulator,
    SimulateTrialResult,
    StepResult,
    WaveFunctionSimulator,
)

from cirq.study import (
    Linspace,
    ParamResolver,
    plot_state_histogram,
    Points,
    Sweep,
    Sweepable,
    TrialResult,
    UnitSweep,
)

from cirq.value import (
    canonicalize_half_turns,
    chosen_angle_to_canonical_half_turns,
    chosen_angle_to_half_turns,
    Duration,
    Symbol,
    Timestamp,
)

from cirq.protocols import (
    apply_unitary_to_tensor,
    CircuitDiagramInfo,
    CircuitDiagramInfoArgs,
    circuit_diagram_info,
    inverse,
    SupportsApplyUnitaryToTensor,
    SupportsCircuitDiagramInfo,
    SupportsUnitary,
    unitary,
)

# Unflattened sub-modules.

from cirq import (
    contrib,
    google,
    testing,
)

# Import version last since it is a relative import.
from ._version import __version__
