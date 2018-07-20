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

# Flatten sub-modules.

from cirq.circuits import (
    Circuit,
    DropEmptyMoments,
    DropNegligible,
    ExpandComposite,
    InsertStrategy,
    Moment,
    OptimizationPass,
    PointOptimizationSummary,
    PointOptimizer,
    QasmOutput,
    TextDiagramDrawer,
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
    bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals,
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
    inverse,
    ISWAP,
    ISwapGate,
    KnownMatrix,
    measure,
    measure_each,
    MeasurementGate,
    NamedQubit,
    OP_TREE,
    Operation,
    QubitOrder,
    QubitOrderOrList,
    ReversibleCompositeGate,
    ReversibleEffect,
    ParameterizableEffect,
    Pauli,
    CliffordGate,
    PauliTransform,
    PauliInteractionGate,
    PauliString,
    PhaseableEffect,
    QasmConvertableGate,
    QasmConvertableOperation,
    QasmOutputArgs,
    QubitId,
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

from cirq.study import (
    Linspace,
    ParamResolver,
    Points,
    Sweep,
    Sweepable,
    TrialResult,
)

from cirq.study.visualize import (
    plot_state_histogram,
)

from cirq.value import (
    canonicalize_half_turns,
    chosen_angle_to_canonical_half_turns,
    chosen_angle_to_half_turns,
    Duration,
    Symbol,
    Timestamp,
)

# Unflattened sub-modules.
from cirq import (
    api,
    google,
    testing,
)

# Import version last since it is a relative import.
from ._version import __version__
