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
from logging import warning

from cirq import _import

# A module can only depend on modules imported earlier in this list of modules
# at import time.  Pytest will fail otherwise (enforced by
# dev_tools/import_test.py).
# Begin dependency order list of sub-modules.
from cirq import (
    # Low level
    _version,
    _compat,
    _doc,
    type_workarounds,
)

with _import.delay_import('cirq.protocols'):
    from cirq import (
        # Core
        protocols,
        value,
        linalg,
        qis,
        ops,
        devices,
        study,
    )
from cirq import (
    # Core
    circuits,
    # Optimize and run
    optimizers,
    work,
    sim,
    vis,
    # Hardware specific
    ion,
    neutral_atoms,
    interop,
    # Applications
    experiments,
    # Extra (nothing should depend on these)
    testing,
)

# End dependency order list of sub-modules

from cirq._version import (
    __version__,
)

# Flattened sub-modules.

from cirq.circuits import (
    AbstractCircuit,
    Alignment,
    Circuit,
    CircuitDag,
    CircuitOperation,
    FrozenCircuit,
    InsertStrategy,
    PointOptimizationSummary,
    PointOptimizer,
    QasmOutput,
    QuilOutput,
    TextDiagramDrawer,
    Unique,
)

from cirq.devices import (
    ConstantQubitNoiseModel,
    Device,
    GridQid,
    GridQubit,
    LineQid,
    LineQubit,
    NO_NOISE,
    NOISE_MODEL_LIKE,
    NoiseModel,
    UNCONSTRAINED_DEVICE,
)

from cirq.experiments import (
    estimate_single_qubit_readout_errors,
    hog_score_xeb_fidelity_from_probabilities,
    least_squares_xeb_fidelity_from_expectations,
    least_squares_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity,
    linear_xeb_fidelity_from_probabilities,
    log_xeb_fidelity,
    log_xeb_fidelity_from_probabilities,
    generate_boixo_2018_supremacy_circuits_v2,
    generate_boixo_2018_supremacy_circuits_v2_bristlecone,
    generate_boixo_2018_supremacy_circuits_v2_grid,
    xeb_fidelity,
)

from cirq.interop import (
    quirk_json_to_circuit,
    quirk_url_to_circuit,
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
    CONTROL_TAG,
    deconstruct_single_qubit_matrix_into_angles,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix,
    dot,
    expand_matrix_in_orthogonal_basis,
    hilbert_schmidt_inner_product,
    is_diagonal,
    is_hermitian,
    is_normal,
    is_orthogonal,
    is_special_orthogonal,
    is_special_unitary,
    is_unitary,
    kak_canonicalize_vector,
    kak_decomposition,
    kak_vector,
    KakDecomposition,
    kron,
    kron_bases,
    kron_factor_4x4_to_2x2s,
    kron_with_controls,
    map_eigenvalues,
    match_global_phase,
    matrix_commutes,
    matrix_from_basis_coefficients,
    num_cnots_required,
    partial_trace,
    partial_trace_of_state_vector_as_mixture,
    PAULI_BASIS,
    scatter_plot_normalized_kak_interaction_coefficients,
    pow_pauli_combination,
    reflection_matrix_pow,
    slice_for_qubits_equal_to,
    so4_to_magic_su2s,
    sub_state_vector,
    targeted_conjugate_about,
    targeted_left_multiply,
    to_special,
    unitary_eig,
)

from cirq.ops import (
    amplitude_damp,
    AmplitudeDampingChannel,
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
    CCNotPowGate,
    CNOT,
    CNotPowGate,
    ControlledGate,
    ControlledOperation,
    cphase,
    CSWAP,
    CSwapGate,
    CX,
    CXPowGate,
    CZ,
    CZPowGate,
    DensePauliString,
    depolarize,
    DepolarizingChannel,
    DiagonalGate,
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
    givens,
    GlobalPhaseOperation,
    H,
    HPowGate,
    I,
    identity_each,
    IdentityGate,
    InterchangeableQubitsGate,
    ISWAP,
    ISwapPowGate,
    LinearCombinationOfGates,
    LinearCombinationOfOperations,
    MatrixGate,
    measure,
    measure_each,
    MeasurementGate,
    Moment,
    MutableDensePauliString,
    MutablePauliString,
    NamedQubit,
    NamedQid,
    OP_TREE,
    Operation,
    ParallelGateOperation,
    Pauli,
    PAULI_GATE_LIKE,
    PAULI_STRING_LIKE,
    PauliInteractionGate,
    PauliString,
    PauliStringGateOperation,
    PauliStringPhasor,
    PauliSum,
    PauliSumExponential,
    PauliSumLike,
    PauliTransform,
    phase_damp,
    phase_flip,
    PhaseDampingChannel,
    PhaseGradientGate,
    PhasedFSimGate,
    PhasedISwapPowGate,
    PhasedXPowGate,
    PhasedXZGate,
    PhaseFlipChannel,
    RandomGateChannel,
    qft,
    Qid,
    QuantumFourierTransformGate,
    QubitOrder,
    QubitOrderOrList,
    QubitPermutationGate,
    reset,
    ResetChannel,
    riswap,
    Rx,
    Ry,
    Rz,
    rx,
    ry,
    rz,
    S,
    SingleQubitCliffordGate,
    SingleQubitGate,
    SingleQubitPauliStringGateOperation,
    SWAP,
    SwapPowGate,
    T,
    TaggedOperation,
    ThreeQubitGate,
    ThreeQubitDiagonalGate,
    TOFFOLI,
    transform_op_tree,
    TwoQubitDiagonalGate,
    TwoQubitGate,
    VirtualTag,
    wait,
    WaitGate,
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
    AlignLeft,
    AlignRight,
    compute_cphase_exponents_for_fsim_decomposition,
    ConvertToCzAndSingleGates,
    decompose_cphase_into_two_fsim,
    decompose_multi_controlled_x,
    decompose_multi_controlled_rotation,
    decompose_two_qubit_interaction_into_four_fsim_gates,
    DropEmptyMoments,
    DropNegligible,
    EjectPhasedPaulis,
    EjectZ,
    ExpandComposite,
    is_negligible_turn,
    merge_single_qubit_gates_into_phased_x_z,
    merge_single_qubit_gates_into_phxz,
    MergeInteractions,
    MergeSingleQubitGates,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form,
    stratified_circuit,
    SynchronizeTerminalMeasurements,
    two_qubit_matrix_to_operations,
    two_qubit_matrix_to_diagonal_and_operations,
    three_qubit_matrix_to_operations,
)

from cirq.qis import (
    bloch_vector_from_state_vector,
    CliffordTableau,
    density_matrix,
    density_matrix_from_state_vector,
    dirac_notation,
    eye_tensor,
    fidelity,
    one_hot,
    QUANTUM_STATE_LIKE,
    QuantumState,
    quantum_state,
    STATE_VECTOR_LIKE,
    to_valid_density_matrix,
    to_valid_state_vector,
    validate_density_matrix,
    validate_indices,
    validate_normalized_state_vector,
    validate_qid_shape,
    von_neumann_entropy,
)

from cirq.sim import (
    ActOnArgs,
    ActOnCliffordTableauArgs,
    ActOnDensityMatrixArgs,
    ActOnStabilizerCHFormArgs,
    ActOnStateVectorArgs,
    StabilizerStateChForm,
    CIRCUIT_LIKE,
    CliffordSimulator,
    CliffordState,
    CliffordSimulatorStepResult,
    CliffordTrialResult,
    DensityMatrixSimulator,
    DensityMatrixSimulatorState,
    DensityMatrixStepResult,
    DensityMatrixTrialResult,
    measure_density_matrix,
    measure_state_vector,
    final_density_matrix,
    final_state_vector,
    sample,
    sample_density_matrix,
    sample_state_vector,
    sample_sweep,
    SimulatesAmplitudes,
    SimulatesExpectationValues,
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulatesIntermediateStateVector,
    SimulatesSamples,
    SimulationTrialResult,
    Simulator,
    SimulatorBase,
    SparseSimulatorStep,
    StabilizerSampler,
    StateVectorMixin,
    StateVectorSimulatorState,
    StateVectorStepResult,
    StateVectorTrialResult,
    StepResult,
)

from cirq.study import (
    dict_to_product_sweep,
    dict_to_zip_sweep,
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
    Sweep,
    Sweepable,
    to_resolvers,
    to_sweep,
    to_sweeps,
    Result,
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
    GenericMetaImplementAnyOneOf,
    LinearDict,
    MEASUREMENT_KEY_SEPARATOR,
    MeasurementKey,
    PeriodicValue,
    RANDOM_STATE_OR_SEED_LIKE,
    Timestamp,
    TParamKey,
    TParamVal,
    validate_probability,
    value_equality,
    KET_PLUS,
    KET_MINUS,
    KET_IMAG,
    KET_MINUS_IMAG,
    KET_ZERO,
    KET_ONE,
    PAULI_STATES,
    ProductState,
)

# pylint: disable=redefined-builtin
from cirq.protocols import (
    act_on,
    apply_channel,
    apply_mixture,
    apply_unitaries,
    apply_unitary,
    ApplyChannelArgs,
    ApplyMixtureArgs,
    ApplyUnitaryArgs,
    approx_eq,
    channel,
    circuit_diagram_info,
    CircuitDiagramInfo,
    CircuitDiagramInfoArgs,
    commutes,
    decompose,
    decompose_once,
    decompose_once_with_qubits,
    DEFAULT_RESOLVERS,
    definitely_commutes,
    equal_up_to_global_phase,
    has_channel,
    has_mixture,
    has_stabilizer_effect,
    has_unitary,
    inverse,
    is_measurement,
    is_parameterized,
    JsonResolver,
    json_serializable_dataclass,
    measurement_key,
    measurement_keys,
    mixture,
    mul,
    num_qubits,
    parameter_names,
    parameter_symbols,
    pauli_expansion,
    phase_by,
    pow,
    qasm,
    QasmArgs,
    qid_shape,
    quil,
    QuilFormatter,
    read_json_gzip,
    read_json,
    resolve_parameters,
    resolve_parameters_once,
    SerializableByKey,
    SupportsActOn,
    SupportsApplyChannel,
    SupportsApplyMixture,
    SupportsApproximateEquality,
    SupportsConsistentApplyUnitary,
    SupportsChannel,
    SupportsCircuitDiagramInfo,
    SupportsCommutes,
    SupportsDecompose,
    SupportsDecomposeWithQubits,
    SupportsEqualUpToGlobalPhase,
    SupportsExplicitHasUnitary,
    SupportsExplicitQidShape,
    SupportsExplicitNumQubits,
    SupportsJSON,
    SupportsMeasurementKey,
    SupportsMixture,
    SupportsParameterization,
    SupportsPauliExpansion,
    SupportsPhase,
    SupportsQasm,
    SupportsQasmWithArgs,
    SupportsQasmWithArgsAndQubits,
    SupportsTraceDistanceBound,
    SupportsUnitary,
    to_json_gzip,
    to_json,
    obj_to_dict_helper,
    trace_distance_bound,
    trace_distance_from_angle_list,
    unitary,
    validate_mixture,
    with_key_path,
    with_measurement_key_mapping,
)

from cirq.ion import (
    ConvertToIonGates,
    IonDevice,
    ms,
    two_qubit_matrix_to_ion_operations,
)
from cirq.neutral_atoms import (
    ConvertToNeutralAtomGates,
    is_native_neutral_atom_gate,
    is_native_neutral_atom_op,
    NeutralAtomDevice,
)

from cirq.vis import (
    Heatmap,
    TwoQubitInteractionHeatmap,
    get_state_histogram,
    integrated_histogram,
)

from cirq.work import (
    CircuitSampleJob,
    PauliSumCollector,
    Sampler,
    Collector,
    ZerosSampler,
)

# pylint: enable=redefined-builtin

# Unflattened sub-modules.

from cirq import (
    ionq,
    pasqal,
    testing,
)

try:
    _compat.deprecated_submodule(
        new_module_name='cirq_google',
        old_parent=__name__,
        old_child='google',
        deadline="v0.14",
        create_attribute=True,
    )
except ImportError as ex:
    # coverage: ignore
    warning("Can't import cirq.google: ", ex)


def _register_resolver() -> None:
    """Registers the cirq module's public classes for JSON serialization."""
    from cirq.protocols.json_serialization import _internal_register_resolver
    from cirq.json_resolver_cache import _class_resolver_dictionary

    _internal_register_resolver(_class_resolver_dictionary)


_register_resolver()

# contrib's json resolver cache depends on cirq.DEFAULT_RESOLVER

# pylint: disable=wrong-import-position
from cirq import (
    contrib,
)

# pylint: enable=wrong-import-position
