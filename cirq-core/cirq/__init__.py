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

"""Cirq is a framework for creating, editing, and invoking quantum circuits."""

from cirq import _import

from cirq._compat import __cirq_debug__ as __cirq_debug__, with_debug as with_debug

# A module can only depend on modules imported earlier in this list of modules
# at import time.  Pytest will fail otherwise (enforced by
# dev_tools/import_test.py).
# Begin dependency order list of sub-modules.
from cirq import (
    # Low level
    _version,
    _doc,
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

from cirq._version import __version__ as __version__

# Flattened sub-modules.

from cirq.circuits import (
    AbstractCircuit as AbstractCircuit,
    Alignment as Alignment,
    Circuit as Circuit,
    CircuitOperation as CircuitOperation,
    FrozenCircuit as FrozenCircuit,
    InsertStrategy as InsertStrategy,
    Moment as Moment,
    PointOptimizationSummary as PointOptimizationSummary,
    PointOptimizer as PointOptimizer,
    QasmOutput as QasmOutput,
    TextDiagramDrawer as TextDiagramDrawer,
)

from cirq.devices import (
    ConstantQubitNoiseModel as ConstantQubitNoiseModel,
    Device as Device,
    DeviceMetadata as DeviceMetadata,
    GridDeviceMetadata as GridDeviceMetadata,
    GridQid as GridQid,
    GridQubit as GridQubit,
    LineQid as LineQid,
    LineQubit as LineQubit,
    NO_NOISE as NO_NOISE,
    NOISE_MODEL_LIKE as NOISE_MODEL_LIKE,
    NoiseModel as NoiseModel,
    NoiseModelFromNoiseProperties as NoiseModelFromNoiseProperties,
    NoiseProperties as NoiseProperties,
    OpIdentifier as OpIdentifier,
    SuperconductingQubitsNoiseProperties as SuperconductingQubitsNoiseProperties,
    UNCONSTRAINED_DEVICE as UNCONSTRAINED_DEVICE,
    NamedTopology as NamedTopology,
    draw_gridlike as draw_gridlike,
    LineTopology as LineTopology,
    TiltedSquareLattice as TiltedSquareLattice,
    get_placements as get_placements,
    is_valid_placement as is_valid_placement,
    draw_placements as draw_placements,
)

from cirq.experiments import (
    TensoredConfusionMatrices as TensoredConfusionMatrices,
    estimate_parallel_single_qubit_readout_errors as estimate_parallel_single_qubit_readout_errors,
    estimate_single_qubit_readout_errors as estimate_single_qubit_readout_errors,
    hog_score_xeb_fidelity_from_probabilities as hog_score_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity as linear_xeb_fidelity,
    linear_xeb_fidelity_from_probabilities as linear_xeb_fidelity_from_probabilities,
    log_xeb_fidelity as log_xeb_fidelity,
    log_xeb_fidelity_from_probabilities as log_xeb_fidelity_from_probabilities,
    measure_confusion_matrix as measure_confusion_matrix,
    xeb_fidelity as xeb_fidelity,
)

from cirq.interop import (
    quirk_json_to_circuit as quirk_json_to_circuit,
    quirk_url_to_circuit as quirk_url_to_circuit,
)

from cirq.linalg import (
    # pylint: disable=line-too-long
    all_near_zero as all_near_zero,
    all_near_zero_mod as all_near_zero_mod,
    allclose_up_to_global_phase as allclose_up_to_global_phase,
    apply_matrix_to_slices as apply_matrix_to_slices,
    axis_angle as axis_angle,
    AxisAngleDecomposition as AxisAngleDecomposition,
    bidiagonalize_real_matrix_pair_with_symmetric_products as bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals as bidiagonalize_unitary_with_special_orthogonals,
    block_diag as block_diag,
    CONTROL_TAG as CONTROL_TAG,
    deconstruct_single_qubit_matrix_into_angles as deconstruct_single_qubit_matrix_into_angles,
    density_matrix_kronecker_product as density_matrix_kronecker_product,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices as diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix as diagonalize_real_symmetric_matrix,
    dot as dot,
    expand_matrix_in_orthogonal_basis as expand_matrix_in_orthogonal_basis,
    hilbert_schmidt_inner_product as hilbert_schmidt_inner_product,
    is_cptp as is_cptp,
    is_diagonal as is_diagonal,
    is_hermitian as is_hermitian,
    is_normal as is_normal,
    is_orthogonal as is_orthogonal,
    is_special_orthogonal as is_special_orthogonal,
    is_special_unitary as is_special_unitary,
    is_unitary as is_unitary,
    kak_canonicalize_vector as kak_canonicalize_vector,
    kak_decomposition as kak_decomposition,
    kak_vector as kak_vector,
    KakDecomposition as KakDecomposition,
    kron as kron,
    kron_bases as kron_bases,
    kron_factor_4x4_to_2x2s as kron_factor_4x4_to_2x2s,
    kron_with_controls as kron_with_controls,
    map_eigenvalues as map_eigenvalues,
    match_global_phase as match_global_phase,
    matrix_commutes as matrix_commutes,
    matrix_from_basis_coefficients as matrix_from_basis_coefficients,
    num_cnots_required as num_cnots_required,
    partial_trace as partial_trace,
    partial_trace_of_state_vector_as_mixture as partial_trace_of_state_vector_as_mixture,
    PAULI_BASIS as PAULI_BASIS,
    scatter_plot_normalized_kak_interaction_coefficients as scatter_plot_normalized_kak_interaction_coefficients,
    pow_pauli_combination as pow_pauli_combination,
    reflection_matrix_pow as reflection_matrix_pow,
    slice_for_qubits_equal_to as slice_for_qubits_equal_to,
    state_vector_kronecker_product as state_vector_kronecker_product,
    so4_to_magic_su2s as so4_to_magic_su2s,
    sub_state_vector as sub_state_vector,
    targeted_conjugate_about as targeted_conjugate_about,
    targeted_left_multiply as targeted_left_multiply,
    to_special as to_special,
    unitary_eig as unitary_eig,
)

from cirq.ops import (
    amplitude_damp as amplitude_damp,
    AmplitudeDampingChannel as AmplitudeDampingChannel,
    AnyIntegerPowerGateFamily as AnyIntegerPowerGateFamily,
    AnyUnitaryGateFamily as AnyUnitaryGateFamily,
    ArithmeticGate as ArithmeticGate,
    asymmetric_depolarize as asymmetric_depolarize,
    AsymmetricDepolarizingChannel as AsymmetricDepolarizingChannel,
    BaseDensePauliString as BaseDensePauliString,
    bit_flip as bit_flip,
    BitFlipChannel as BitFlipChannel,
    BooleanHamiltonianGate as BooleanHamiltonianGate,
    CCX as CCX,
    CCXPowGate as CCXPowGate,
    CCZ as CCZ,
    CCZPowGate as CCZPowGate,
    CCNOT as CCNOT,
    CCNotPowGate as CCNotPowGate,
    ClassicallyControlledOperation as ClassicallyControlledOperation,
    CliffordGate as CliffordGate,
    CNOT as CNOT,
    CNotPowGate as CNotPowGate,
    ControlledGate as ControlledGate,
    ControlledOperation as ControlledOperation,
    cphase as cphase,
    CSWAP as CSWAP,
    CSwapGate as CSwapGate,
    CX as CX,
    CXPowGate as CXPowGate,
    CZ as CZ,
    CZPowGate as CZPowGate,
    DensePauliString as DensePauliString,
    depolarize as depolarize,
    DepolarizingChannel as DepolarizingChannel,
    DiagonalGate as DiagonalGate,
    EigenGate as EigenGate,
    flatten_op_tree as flatten_op_tree,
    flatten_to_ops as flatten_to_ops,
    flatten_to_ops_or_moments as flatten_to_ops_or_moments,
    FREDKIN as FREDKIN,
    freeze_op_tree as freeze_op_tree,
    FSimGate as FSimGate,
    Gate as Gate,
    GateFamily as GateFamily,
    GateOperation as GateOperation,
    Gateset as Gateset,
    generalized_amplitude_damp as generalized_amplitude_damp,
    GeneralizedAmplitudeDampingChannel as GeneralizedAmplitudeDampingChannel,
    givens as givens,
    GlobalPhaseGate as GlobalPhaseGate,
    global_phase_operation as global_phase_operation,
    GreedyQubitManager as GreedyQubitManager,
    H as H,
    HPowGate as HPowGate,
    I as I,
    identity_each as identity_each,
    IdentityGate as IdentityGate,
    InterchangeableQubitsGate as InterchangeableQubitsGate,
    ISWAP as ISWAP,
    ISwapPowGate as ISwapPowGate,
    ISWAP_INV as ISWAP_INV,
    KrausChannel as KrausChannel,
    LinearCombinationOfGates as LinearCombinationOfGates,
    LinearCombinationOfOperations as LinearCombinationOfOperations,
    MatrixGate as MatrixGate,
    MixedUnitaryChannel as MixedUnitaryChannel,
    M as M,
    MSGate as MSGate,
    measure as measure,
    measure_each as measure_each,
    measure_paulistring_terms as measure_paulistring_terms,
    measure_single_paulistring as measure_single_paulistring,
    MeasurementGate as MeasurementGate,
    MutableDensePauliString as MutableDensePauliString,
    MutablePauliString as MutablePauliString,
    ms as ms,
    NamedQubit as NamedQubit,
    NamedQid as NamedQid,
    OP_TREE as OP_TREE,
    Operation as Operation,
    ParallelGate as ParallelGate,
    ParallelGateFamily as ParallelGateFamily,
    parallel_gate_op as parallel_gate_op,
    Pauli as Pauli,
    PAULI_GATE_LIKE as PAULI_GATE_LIKE,
    PAULI_STRING_LIKE as PAULI_STRING_LIKE,
    PauliInteractionGate as PauliInteractionGate,
    PauliMeasurementGate as PauliMeasurementGate,
    PauliString as PauliString,
    PauliStringGateOperation as PauliStringGateOperation,
    PauliStringPhasor as PauliStringPhasor,
    PauliStringPhasorGate as PauliStringPhasorGate,
    PauliSum as PauliSum,
    PauliSumExponential as PauliSumExponential,
    PauliSumLike as PauliSumLike,
    phase_damp as phase_damp,
    phase_flip as phase_flip,
    PhaseDampingChannel as PhaseDampingChannel,
    PhaseGradientGate as PhaseGradientGate,
    PhasedFSimGate as PhasedFSimGate,
    PhasedISwapPowGate as PhasedISwapPowGate,
    PhasedXPowGate as PhasedXPowGate,
    PhasedXZGate as PhasedXZGate,
    PhaseFlipChannel as PhaseFlipChannel,
    StatePreparationChannel as StatePreparationChannel,
    ProductOfSums as ProductOfSums,
    ProjectorString as ProjectorString,
    ProjectorSum as ProjectorSum,
    RandomGateChannel as RandomGateChannel,
    q as q,
    qft as qft,
    Qid as Qid,
    QuantumFourierTransformGate as QuantumFourierTransformGate,
    QubitManager as QubitManager,
    QubitOrder as QubitOrder,
    QubitOrderOrList as QubitOrderOrList,
    QubitPermutationGate as QubitPermutationGate,
    R as R,
    reset as reset,
    reset_each as reset_each,
    ResetChannel as ResetChannel,
    RoutingSwapTag as RoutingSwapTag,
    riswap as riswap,
    Rx as Rx,
    Ry as Ry,
    Rz as Rz,
    rx as rx,
    ry as ry,
    rz as rz,
    S as S,
    SimpleQubitManager as SimpleQubitManager,
    SingleQubitCliffordGate as SingleQubitCliffordGate,
    SingleQubitPauliStringGateOperation as SingleQubitPauliStringGateOperation,
    SQRT_ISWAP as SQRT_ISWAP,
    SQRT_ISWAP_INV as SQRT_ISWAP_INV,
    SWAP as SWAP,
    SwapPowGate as SwapPowGate,
    SumOfProducts as SumOfProducts,
    T as T,
    TaggedOperation as TaggedOperation,
    ThreeQubitDiagonalGate as ThreeQubitDiagonalGate,
    TOFFOLI as TOFFOLI,
    transform_op_tree as transform_op_tree,
    TwoQubitDiagonalGate as TwoQubitDiagonalGate,
    VirtualTag as VirtualTag,
    wait as wait,
    WaitGate as WaitGate,
    X as X,
    XPowGate as XPowGate,
    XX as XX,
    XXPowGate as XXPowGate,
    Y as Y,
    YPowGate as YPowGate,
    YY as YY,
    YYPowGate as YYPowGate,
    Z as Z,
    ZPowGate as ZPowGate,
    ZZ as ZZ,
    ZZPowGate as ZZPowGate,
    UniformSuperpositionGate as UniformSuperpositionGate,
)


from cirq.transformers import (
    # pylint: disable=line-too-long
    AbstractInitialMapper as AbstractInitialMapper,
    add_dynamical_decoupling as add_dynamical_decoupling,
    align_left as align_left,
    align_right as align_right,
    CompilationTargetGateset as CompilationTargetGateset,
    CZTargetGateset as CZTargetGateset,
    compute_cphase_exponents_for_fsim_decomposition as compute_cphase_exponents_for_fsim_decomposition,
    create_transformer_with_kwargs as create_transformer_with_kwargs,
    decompose_clifford_tableau_to_operations as decompose_clifford_tableau_to_operations,
    decompose_cphase_into_two_fsim as decompose_cphase_into_two_fsim,
    decompose_multi_controlled_x as decompose_multi_controlled_x,
    decompose_multi_controlled_rotation as decompose_multi_controlled_rotation,
    decompose_two_qubit_interaction_into_four_fsim_gates as decompose_two_qubit_interaction_into_four_fsim_gates,
    defer_measurements as defer_measurements,
    dephase_measurements as dephase_measurements,
    drop_empty_moments as drop_empty_moments,
    drop_negligible_operations as drop_negligible_operations,
    drop_terminal_measurements as drop_terminal_measurements,
    eject_phased_paulis as eject_phased_paulis,
    eject_z as eject_z,
    expand_composite as expand_composite,
    HardCodedInitialMapper as HardCodedInitialMapper,
    is_negligible_turn as is_negligible_turn,
    LineInitialMapper as LineInitialMapper,
    MappingManager as MappingManager,
    map_clean_and_borrowable_qubits as map_clean_and_borrowable_qubits,
    map_moments as map_moments,
    map_operations as map_operations,
    map_operations_and_unroll as map_operations_and_unroll,
    merge_k_qubit_unitaries as merge_k_qubit_unitaries,
    merge_k_qubit_unitaries_to_circuit_op as merge_k_qubit_unitaries_to_circuit_op,
    merge_moments as merge_moments,
    merge_operations as merge_operations,
    merge_operations_to_circuit_op as merge_operations_to_circuit_op,
    merge_single_qubit_gates_to_phased_x_and_z as merge_single_qubit_gates_to_phased_x_and_z,
    merge_single_qubit_gates_to_phxz as merge_single_qubit_gates_to_phxz,
    merge_single_qubit_moments_to_phxz as merge_single_qubit_moments_to_phxz,
    optimize_for_target_gateset as optimize_for_target_gateset,
    parameterized_2q_op_to_sqrt_iswap_operations as parameterized_2q_op_to_sqrt_iswap_operations,
    prepare_two_qubit_state_using_cz as prepare_two_qubit_state_using_cz,
    prepare_two_qubit_state_using_iswap as prepare_two_qubit_state_using_iswap,
    prepare_two_qubit_state_using_sqrt_iswap as prepare_two_qubit_state_using_sqrt_iswap,
    quantum_shannon_decomposition as quantum_shannon_decomposition,
    RouteCQC as RouteCQC,
    routed_circuit_with_mapping as routed_circuit_with_mapping,
    SqrtIswapTargetGateset as SqrtIswapTargetGateset,
    single_qubit_matrix_to_gates as single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations as single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z as single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz as single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form as single_qubit_op_to_framed_phase_form,
    stratified_circuit as stratified_circuit,
    synchronize_terminal_measurements as synchronize_terminal_measurements,
    TRANSFORMER as TRANSFORMER,
    TransformerContext as TransformerContext,
    TransformerLogger as TransformerLogger,
    three_qubit_matrix_to_operations as three_qubit_matrix_to_operations,
    transformer as transformer,
    two_qubit_matrix_to_cz_isometry as two_qubit_matrix_to_cz_isometry,
    two_qubit_matrix_to_cz_operations as two_qubit_matrix_to_cz_operations,
    two_qubit_matrix_to_diagonal_and_cz_operations as two_qubit_matrix_to_diagonal_and_cz_operations,
    two_qubit_matrix_to_ion_operations as two_qubit_matrix_to_ion_operations,
    two_qubit_matrix_to_sqrt_iswap_operations as two_qubit_matrix_to_sqrt_iswap_operations,
    two_qubit_gate_product_tabulation as two_qubit_gate_product_tabulation,
    TwoQubitCompilationTargetGateset as TwoQubitCompilationTargetGateset,
    TwoQubitGateTabulation as TwoQubitGateTabulation,
    TwoQubitGateTabulationResult as TwoQubitGateTabulationResult,
    toggle_tags as toggle_tags,
    unroll_circuit_op as unroll_circuit_op,
    unroll_circuit_op_greedy_earliest as unroll_circuit_op_greedy_earliest,
    unroll_circuit_op_greedy_frontier as unroll_circuit_op_greedy_frontier,
)

from cirq.qis import (
    bloch_vector_from_state_vector as bloch_vector_from_state_vector,
    choi_to_kraus as choi_to_kraus,
    choi_to_superoperator as choi_to_superoperator,
    CliffordTableau as CliffordTableau,
    density_matrix as density_matrix,
    density_matrix_from_state_vector as density_matrix_from_state_vector,
    dirac_notation as dirac_notation,
    entanglement_fidelity as entanglement_fidelity,
    eye_tensor as eye_tensor,
    fidelity as fidelity,
    kraus_to_choi as kraus_to_choi,
    kraus_to_superoperator as kraus_to_superoperator,
    one_hot as one_hot,
    operation_to_choi as operation_to_choi,
    operation_to_superoperator as operation_to_superoperator,
    QUANTUM_STATE_LIKE as QUANTUM_STATE_LIKE,
    QuantumState as QuantumState,
    QuantumStateRepresentation as QuantumStateRepresentation,
    quantum_state as quantum_state,
    STATE_VECTOR_LIKE as STATE_VECTOR_LIKE,
    StabilizerState as StabilizerState,
    superoperator_to_choi as superoperator_to_choi,
    superoperator_to_kraus as superoperator_to_kraus,
    to_valid_density_matrix as to_valid_density_matrix,
    to_valid_state_vector as to_valid_state_vector,
    validate_density_matrix as validate_density_matrix,
    validate_indices as validate_indices,
    validate_normalized_state_vector as validate_normalized_state_vector,
    validate_qid_shape as validate_qid_shape,
    von_neumann_entropy as von_neumann_entropy,
)

from cirq.sim import (
    CIRCUIT_LIKE as CIRCUIT_LIKE,
    ClassicalStateSimulator as ClassicalStateSimulator,
    CliffordSimulator as CliffordSimulator,
    CliffordState as CliffordState,
    CliffordSimulatorStepResult as CliffordSimulatorStepResult,
    CliffordTableauSimulationState as CliffordTableauSimulationState,
    CliffordTrialResult as CliffordTrialResult,
    DensityMatrixSimulationState as DensityMatrixSimulationState,
    DensityMatrixSimulator as DensityMatrixSimulator,
    DensityMatrixStepResult as DensityMatrixStepResult,
    DensityMatrixTrialResult as DensityMatrixTrialResult,
    measure_density_matrix as measure_density_matrix,
    measure_state_vector as measure_state_vector,
    final_density_matrix as final_density_matrix,
    final_state_vector as final_state_vector,
    sample as sample,
    sample_density_matrix as sample_density_matrix,
    sample_state_vector as sample_state_vector,
    sample_sweep as sample_sweep,
    SimulatesAmplitudes as SimulatesAmplitudes,
    SimulatesExpectationValues as SimulatesExpectationValues,
    SimulatesFinalState as SimulatesFinalState,
    SimulatesIntermediateState as SimulatesIntermediateState,
    SimulatesIntermediateStateVector as SimulatesIntermediateStateVector,
    SimulatesSamples as SimulatesSamples,
    SimulationProductState as SimulationProductState,
    SimulationState as SimulationState,
    SimulationStateBase as SimulationStateBase,
    SimulationTrialResult as SimulationTrialResult,
    SimulationTrialResultBase as SimulationTrialResultBase,
    Simulator as Simulator,
    SimulatorBase as SimulatorBase,
    SparseSimulatorStep as SparseSimulatorStep,
    StabilizerChFormSimulationState as StabilizerChFormSimulationState,
    StabilizerSampler as StabilizerSampler,
    StabilizerSimulationState as StabilizerSimulationState,
    StabilizerStateChForm as StabilizerStateChForm,
    StateVectorMixin as StateVectorMixin,
    StateVectorSimulationState as StateVectorSimulationState,
    StateVectorStepResult as StateVectorStepResult,
    StateVectorTrialResult as StateVectorTrialResult,
    StepResult as StepResult,
    StepResultBase as StepResultBase,
)

from cirq.study import (
    Concat as Concat,
    dict_to_product_sweep as dict_to_product_sweep,
    dict_to_zip_sweep as dict_to_zip_sweep,
    ExpressionMap as ExpressionMap,
    flatten as flatten,
    flatten_with_params as flatten_with_params,
    flatten_with_sweep as flatten_with_sweep,
    ResultDict as ResultDict,
    Linspace as Linspace,
    ListSweep as ListSweep,
    ParamDictType as ParamDictType,
    ParamMappingType as ParamMappingType,
    ParamResolver as ParamResolver,
    ParamResolverOrSimilarType as ParamResolverOrSimilarType,
    Points as Points,
    Product as Product,
    Sweep as Sweep,
    Sweepable as Sweepable,
    to_resolvers as to_resolvers,
    to_sweep as to_sweep,
    to_sweeps as to_sweeps,
    Result as Result,
    UnitSweep as UnitSweep,
    UNIT_SWEEP as UNIT_SWEEP,
    Zip as Zip,
    ZipLongest as ZipLongest,
)


from cirq.value import (
    ABCMetaImplementAnyOneOf as ABCMetaImplementAnyOneOf,
    alternative as alternative,
    big_endian_bits_to_int as big_endian_bits_to_int,
    big_endian_digits_to_int as big_endian_digits_to_int,
    big_endian_int_to_bits as big_endian_int_to_bits,
    big_endian_int_to_digits as big_endian_int_to_digits,
    canonicalize_half_turns as canonicalize_half_turns,
    chosen_angle_to_canonical_half_turns as chosen_angle_to_canonical_half_turns,
    chosen_angle_to_half_turns as chosen_angle_to_half_turns,
    ClassicalDataDictionaryStore as ClassicalDataDictionaryStore,
    ClassicalDataStore as ClassicalDataStore,
    ClassicalDataStoreReader as ClassicalDataStoreReader,
    Condition as Condition,
    Duration as Duration,
    DURATION_LIKE as DURATION_LIKE,
    KeyCondition as KeyCondition,
    LinearDict as LinearDict,
    MEASUREMENT_KEY_SEPARATOR as MEASUREMENT_KEY_SEPARATOR,
    MeasurementKey as MeasurementKey,
    MeasurementType as MeasurementType,
    PeriodicValue as PeriodicValue,
    RANDOM_STATE_OR_SEED_LIKE as RANDOM_STATE_OR_SEED_LIKE,
    state_vector_to_probabilities as state_vector_to_probabilities,
    SympyCondition as SympyCondition,
    Timestamp as Timestamp,
    TParamKey as TParamKey,
    TParamVal as TParamVal,
    TParamValComplex as TParamValComplex,
    validate_probability as validate_probability,
    value_equality as value_equality,
    KET_PLUS as KET_PLUS,
    KET_MINUS as KET_MINUS,
    KET_IMAG as KET_IMAG,
    KET_MINUS_IMAG as KET_MINUS_IMAG,
    KET_ZERO as KET_ZERO,
    KET_ONE as KET_ONE,
    PAULI_STATES as PAULI_STATES,
    ProductState as ProductState,
)

# pylint: disable=redefined-builtin
from cirq.protocols import (
    act_on as act_on,
    apply_channel as apply_channel,
    apply_mixture as apply_mixture,
    apply_unitaries as apply_unitaries,
    apply_unitary as apply_unitary,
    ApplyChannelArgs as ApplyChannelArgs,
    ApplyMixtureArgs as ApplyMixtureArgs,
    ApplyUnitaryArgs as ApplyUnitaryArgs,
    approx_eq as approx_eq,
    circuit_diagram_info as circuit_diagram_info,
    CircuitDiagramInfo as CircuitDiagramInfo,
    CircuitDiagramInfoArgs as CircuitDiagramInfoArgs,
    cirq_type_from_json as cirq_type_from_json,
    commutes as commutes,
    control_keys as control_keys,
    decompose as decompose,
    decompose_once as decompose_once,
    decompose_once_with_qubits as decompose_once_with_qubits,
    DecompositionContext as DecompositionContext,
    DEFAULT_RESOLVERS as DEFAULT_RESOLVERS,
    definitely_commutes as definitely_commutes,
    equal_up_to_global_phase as equal_up_to_global_phase,
    has_kraus as has_kraus,
    has_mixture as has_mixture,
    has_stabilizer_effect as has_stabilizer_effect,
    has_unitary as has_unitary,
    HasJSONNamespace as HasJSONNamespace,
    inverse as inverse,
    is_measurement as is_measurement,
    is_parameterized as is_parameterized,
    JsonResolver as JsonResolver,
    json_cirq_type as json_cirq_type,
    json_namespace as json_namespace,
    dataclass_json_dict as dataclass_json_dict,
    kraus as kraus,
    LabelEntity as LabelEntity,
    measurement_key_name as measurement_key_name,
    measurement_key_obj as measurement_key_obj,
    measurement_key_names as measurement_key_names,
    measurement_key_objs as measurement_key_objs,
    measurement_keys_touched as measurement_keys_touched,
    mixture as mixture,
    mul as mul,
    num_qubits as num_qubits,
    parameter_names as parameter_names,
    parameter_symbols as parameter_symbols,
    pauli_expansion as pauli_expansion,
    phase_by as phase_by,
    pow as pow,
    qasm as qasm,
    QasmArgs as QasmArgs,
    qid_shape as qid_shape,
    read_json_gzip as read_json_gzip,
    read_json as read_json,
    resolve_parameters as resolve_parameters,
    resolve_parameters_once as resolve_parameters_once,
    SerializableByKey as SerializableByKey,
    SupportsActOn as SupportsActOn,
    SupportsActOnQubits as SupportsActOnQubits,
    SupportsApplyChannel as SupportsApplyChannel,
    SupportsApplyMixture as SupportsApplyMixture,
    SupportsApproximateEquality as SupportsApproximateEquality,
    SupportsConsistentApplyUnitary as SupportsConsistentApplyUnitary,
    SupportsCircuitDiagramInfo as SupportsCircuitDiagramInfo,
    SupportsCommutes as SupportsCommutes,
    SupportsControlKey as SupportsControlKey,
    SupportsDecompose as SupportsDecompose,
    SupportsDecomposeWithQubits as SupportsDecomposeWithQubits,
    SupportsEqualUpToGlobalPhase as SupportsEqualUpToGlobalPhase,
    SupportsExplicitHasUnitary as SupportsExplicitHasUnitary,
    SupportsExplicitQidShape as SupportsExplicitQidShape,
    SupportsExplicitNumQubits as SupportsExplicitNumQubits,
    SupportsJSON as SupportsJSON,
    SupportsKraus as SupportsKraus,
    SupportsMeasurementKey as SupportsMeasurementKey,
    SupportsMixture as SupportsMixture,
    SupportsParameterization as SupportsParameterization,
    SupportsPauliExpansion as SupportsPauliExpansion,
    SupportsPhase as SupportsPhase,
    SupportsQasm as SupportsQasm,
    SupportsQasmWithArgs as SupportsQasmWithArgs,
    SupportsQasmWithArgsAndQubits as SupportsQasmWithArgsAndQubits,
    SupportsTraceDistanceBound as SupportsTraceDistanceBound,
    SupportsUnitary as SupportsUnitary,
    to_json_gzip as to_json_gzip,
    to_json as to_json,
    obj_to_dict_helper as obj_to_dict_helper,
    trace_distance_bound as trace_distance_bound,
    trace_distance_from_angle_list as trace_distance_from_angle_list,
    unitary as unitary,
    validate_mixture as validate_mixture,
    with_key_path as with_key_path,
    with_key_path_prefix as with_key_path_prefix,
    with_measurement_key_mapping as with_measurement_key_mapping,
    with_rescoped_keys as with_rescoped_keys,
)


from cirq.neutral_atoms import (
    is_native_neutral_atom_gate as is_native_neutral_atom_gate,
    is_native_neutral_atom_op as is_native_neutral_atom_op,
)

from cirq.vis import (
    Heatmap as Heatmap,
    TwoQubitInteractionHeatmap as TwoQubitInteractionHeatmap,
    get_state_histogram as get_state_histogram,
    integrated_histogram as integrated_histogram,
    plot_density_matrix as plot_density_matrix,
    plot_state_histogram as plot_state_histogram,
)

from cirq.work import (
    CircuitSampleJob as CircuitSampleJob,
    PauliSumCollector as PauliSumCollector,
    Sampler as Sampler,
    Collector as Collector,
    ZerosSampler as ZerosSampler,
)

# pylint: enable=redefined-builtin

# Unflattened sub-modules.

from cirq import testing

# Registers cirq-core's public classes for JSON serialization.
# pylint: disable=wrong-import-position
from cirq.protocols.json_serialization import _register_resolver
from cirq.json_resolver_cache import _class_resolver_dictionary


_register_resolver(_class_resolver_dictionary)

# contrib's json resolver cache depends on cirq.DEFAULT_RESOLVER

from cirq import contrib

# pylint: enable=wrong-import-position
