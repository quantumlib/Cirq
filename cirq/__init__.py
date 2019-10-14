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

from cirq import _import

# A module can only depend on modules imported earlier in this list of modules
# at import time.  Pytest will fail otherwise (enforced by
# dev_tools/import_test.py).
# Begin dependency order list of sub-modules.
from cirq import (
    # Low level
    _version,
    _compat,
    type_workarounds,
)
with _import.delay_import('cirq.protocols'):
    from cirq import (
        # Core
        protocols,
        value,
        linalg,
        ops,
        devices,
        study,
    )
from cirq import (
    # Core
    circuits,
    schedules,
    # Optimize and run
    optimizers,
    work,
    sim,
    vis,
    # Hardware specific
    ion,
    neutral_atoms,
    api,
    google,
    # Applications
    experiments,
    # Extra (nothing should depend on these)
    testing,
    contrib,
)
# End dependency order list of sub-modules

from cirq._version import (
    __version__,
)

# Flattened sub-modules.

from cirq.circuits import (
    Circuit,
    CircuitDag,
    InsertStrategy,
    PointOptimizationSummary,
    PointOptimizer,
    QasmOutput,
    TextDiagramDrawer,
    Unique,
)

from cirq.devices import (
    ConstantQubitNoiseModel,
    Device,
    GridQubit,
    LineQid,
    LineQubit,
    NO_NOISE,
    NOISE_MODEL_LIKE,
    NoiseModel,
    UNCONSTRAINED_DEVICE,
)

from cirq.experiments import (
    linear_xeb_fidelity,
    generate_boixo_2018_supremacy_circuits_v2,
    generate_boixo_2018_supremacy_circuits_v2_bristlecone,
    generate_boixo_2018_supremacy_circuits_v2_grid,
)

from cirq.linalg import (
    all_near_zero,
    all_near_zero_mod,
    allclose_up_to_global_phase,
    apply_matrix_to_slices,
    axis_angle,
    AxisAngleDecomposition,
    bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals,
    block_diag,
    commutes,
    CONTROL_TAG,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix,
    dot,
    expand_matrix_in_orthogonal_basis,
    hilbert_schmidt_inner_product,
    eye_tensor,
    is_diagonal,
    is_hermitian,
    is_orthogonal,
    is_special_orthogonal,
    is_special_unitary,
    is_unitary,
    kak_canonicalize_vector,
    kak_decomposition,
    kak_vector,
    KakDecomposition,
    subwavefunction,
    kron,
    kron_bases,
    kron_factor_4x4_to_2x2s,
    kron_with_controls,
    map_eigenvalues,
    match_global_phase,
    matrix_from_basis_coefficients,
    one_hot,
    partial_trace,
    PAULI_BASIS,
    scatter_plot_normalized_kak_interaction_coefficients,
    pow_pauli_combination,
    reflection_matrix_pow,
    slice_for_qubits_equal_to,
    so4_to_magic_su2s,
    targeted_conjugate_about,
    targeted_left_multiply,
    wavefunction_partial_trace_as_mixture,
)

from cirq.ops import (
    amplitude_damp,
    AmplitudeDampingChannel,
    ApproxPauliStringExpectation,
    ArithmeticOperation,
    asymmetric_depolarize,
    AsymmetricDepolarizingChannel,
    BaseDensePauliString,
    bit_flip,
    BitFlipChannel,
    CCX,
    CCXPowGate,
    CCZ,
    CCZPowGate,
    CCNOT,
    CNOT,
    CNotPowGate,
    ControlledGate,
    ControlledOperation,
    CSWAP,
    CSwapGate,
    CX,
    CZ,
    CZPowGate,
    DensePauliString,
    DensityMatrixDisplay,
    depolarize,
    DepolarizingChannel,
    EigenGate,
    flatten_op_tree,
    flatten_to_ops,
    flatten_to_ops_or_moments,
    FREDKIN,
    freeze_op_tree,
    FSimGate,
    Gate,
    GateOperation,
    generalized_amplitude_damp,
    GeneralizedAmplitudeDampingChannel,
    GivensRotation,
    GlobalPhaseOperation,
    H,
    HPowGate,
    I,
    identity,
    IdentityGate,
    InterchangeableQubitsGate,
    ISWAP,
    ISwapPowGate,
    LinearCombinationOfGates,
    LinearCombinationOfOperations,
    measure,
    measure_each,
    MeasurementGate,
    Moment,
    MutableDensePauliString,
    NamedQubit,
    op_gate_isinstance,
    op_gate_of_type,
    OP_TREE,
    Operation,
    ParallelGateOperation,
    Pauli,
    approx_pauli_string_expectation,
    PAULI_STRING_LIKE,
    PauliInteractionGate,
    PauliString,
    PauliStringGateOperation,
    PauliStringPhasor,
    PauliSum,
    PauliSumLike,
    PauliTransform,
    phase_damp,
    phase_flip,
    PhaseDampingChannel,
    PhaseGradientGate,
    PhasedISwapPowGate,
    PhasedXPowGate,
    PhaseFlipChannel,
    QFT,
    Qid,
    QuantumFourierTransformGate,
    QubitOrder,
    QubitOrderOrList,
    reset,
    ResetChannel,
    Rx,
    Ry,
    Rz,
    S,
    SamplesDisplay,
    SingleQubitCliffordGate,
    SingleQubitGate,
    SingleQubitPauliStringGateOperation,
    SingleQubitMatrixGate,
    SWAP,
    SwapPowGate,
    T,
    ThreeQubitGate,
    ThreeQubitDiagonalGate,
    TOFFOLI,
    transform_op_tree,
    TwoQubitGate,
    TwoQubitMatrixGate,
    WaitGate,
    WaveFunctionDisplay,
    X,
    XPowGate,
    XX,
    XXPowGate,
    Y,
    YPowGate,
    YY,
    YYPowGate,
    Z,
    ZPowGate,
    ZZ,
    ZZPowGate,
)

from cirq.optimizers import (
    ConvertToCzAndSingleGates,
    DropEmptyMoments,
    DropNegligible,
    EjectPhasedPaulis,
    EjectZ,
    ExpandComposite,
    is_negligible_turn,
    merge_single_qubit_gates_into_phased_x_z,
    MergeInteractions,
    MergeSingleQubitGates,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z,
    single_qubit_op_to_framed_phase_form,
    SynchronizeTerminalMeasurements,
    two_qubit_matrix_to_operations,
)

from cirq.schedules import (
    moment_by_moment_schedule,
    Schedule,
    ScheduledOperation,
)

from cirq.sim import (
    bloch_vector_from_state_vector,
    density_matrix_from_state_vector,
    DensityMatrixSimulator,
    DensityMatrixSimulatorState,
    DensityMatrixStepResult,
    DensityMatrixTrialResult,
    dirac_notation,
    measure_density_matrix,
    measure_state_vector,
    final_wavefunction,
    sample,
    sample_density_matrix,
    sample_state_vector,
    sample_sweep,
    SimulatesAmplitudes,
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulatesIntermediateWaveFunction,
    SimulatesSamples,
    SimulationTrialResult,
    Simulator,
    SparseSimulatorStep,
    StateVectorMixin,
    StepResult,
    to_valid_density_matrix,
    to_valid_state_vector,
    validate_normalized_state,
    von_neumann_entropy,
    WaveFunctionSimulatorState,
    WaveFunctionStepResult,
    WaveFunctionTrialResult,
)

from cirq.study import (
    ComputeDisplaysResult,
    ExpressionMap,
    flatten,
    flatten_with_params,
    flatten_with_sweep,
    Linspace,
    ListSweep,
    ParamDictType,
    ParamResolver,
    ParamResolverOrSimilarType,
    plot_state_histogram,
    Points,
    Product,
    SampleResult,
    Sweep,
    Sweepable,
    to_resolvers,
    to_sweep,
    TrialResult,
    UnitSweep,
    Zip,
)

from cirq.value import (
    ABCMetaImplementAnyOneOf,
    alternative,
    big_endian_bits_to_int,
    big_endian_digits_to_int,
    big_endian_int_to_bits,
    big_endian_int_to_digits,
    canonicalize_half_turns,
    chosen_angle_to_canonical_half_turns,
    chosen_angle_to_half_turns,
    Duration,
    DURATION_LIKE,
    LinearDict,
    PeriodicValue,
    Timestamp,
    TParamVal,
    validate_probability,
    value_equality,
)

# pylint: disable=redefined-builtin
from cirq.protocols import (
    apply_channel,
    apply_unitaries,
    apply_unitary,
    ApplyChannelArgs,
    ApplyUnitaryArgs,
    approx_eq,
    channel,
    circuit_diagram_info,
    CircuitDiagramInfo,
    CircuitDiagramInfoArgs,
    decompose,
    decompose_once,
    decompose_once_with_qubits,
    equal_up_to_global_phase,
    has_channel,
    has_mixture,
    has_mixture_channel,
    has_unitary,
    inverse,
    is_measurement,
    is_parameterized,
    measurement_key,
    mixture,
    mixture_channel,
    mul,
    num_qubits,
    pauli_expansion,
    phase_by,
    pow,
    qasm,
    QasmArgs,
    qid_shape,
    read_json,
    resolve_parameters,
    SupportsApplyChannel,
    SupportsConsistentApplyUnitary,
    SupportsApproximateEquality,
    SupportsChannel,
    SupportsCircuitDiagramInfo,
    SupportsDecompose,
    SupportsDecomposeWithQubits,
    SupportsExplicitHasUnitary,
    SupportsExplicitQidShape,
    SupportsExplicitNumQubits,
    SupportsMixture,
    SupportsParameterization,
    SupportsPhase,
    SupportsQasm,
    SupportsQasmWithArgs,
    SupportsQasmWithArgsAndQubits,
    SupportsTraceDistanceBound,
    SupportsUnitary,
    to_json,
    obj_to_dict_helper,
    trace_distance_bound,
    unitary,
    validate_mixture,
)

from cirq.ion import (
    ConvertToIonGates,
    IonDevice,
    MS,
    two_qubit_matrix_to_ion_operations,
)
from cirq.neutral_atoms import (
    ConvertToNeutralAtomGates,
    is_native_neutral_atom_gate,
    is_native_neutral_atom_op,
    NeutralAtomDevice,
)

from cirq.vis import (
    Heatmap,)

from cirq.work import (
    CircuitSampleJob,
    PauliSumCollector,
    Sampler,
    Collector,
)

# pylint: enable=redefined-builtin

# Unflattened sub-modules.

from cirq import (
    contrib,
    google,
    testing,
)
