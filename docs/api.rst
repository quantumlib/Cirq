.. currentmodule:: cirq

API Reference
=============


Devices and Qubits
''''''''''''''''''

Classes for identifying the qubits and hardware you want to operate on.

.. autosummary::
    :toctree: generated/

    Device
    GridQubit
    LineQubit
    NamedQubit
    Qid
    UnconstrainedDevice


Single Qubit Unitary Gates
''''''''''''''''''''''''''

Unitary operations you can apply to a single qubit.

.. autosummary::
    :toctree: generated/

    H
    HPowGate
    measure
    measure_each
    MeasurementGate
    PhasedXPowGate
    Rx
    Ry
    Rz
    S
    SingleQubitMatrixGate
    T
    TwoQubitMatrixGate
    X
    XPowGate
    Y
    YPowGate
    Z
    ZPowGate


Two Qubit Unitary Gates
'''''''''''''''''''''''

Unitary operations you can apply to pairs of qubits.

.. autosummary::
    :toctree: generated/

    CNOT
    CNotPowGate
    CZ
    CZPowGate
    ISWAP
    ISwapPowGate
    MS
    SWAP
    SwapPowGate
    XX
    XXPowGate
    YY
    YYPowGate
    ZZ
    ZZPowGate


Three Qubit Unitary Gates
'''''''''''''''''''''''''

Unitary operations you can apply to triplets of qubits, with helpful
adjacency-respecting decompositions.

.. autosummary::
   :toctree: generated/

    CCX
    CCXPowGate
    CCZ
    CCZPowGate
    CSWAP
    CSwapGate
    FREDKIN
    TOFFOLI


Multiqubit Unitary Gates
''''''''''''''''''''''''

Some gates can be applied to arbitrary number of qubits

.. autosummary::
    :toctree: generated/

    I
    IdentityGate

Measurements
''''''''''''

Measurement can be on multiple qubits. Currently only measurement in
computational basis is supported.

.. autosummary::
    :toctree: generated/

    measure
    measure_each
    MeasurementGate

Channels and Mixture Gate
'''''''''''''''''''''''''

Non-unitary gates. Mixture gates are those that can be interpreted
as applying a unitary for a fixed probability while channel encompasses
the more general concept of a noisy open system quantum evolution.

.. autosummary::
    :toctree: generated/

    amplitude_damp
    AmplitudeDampingChannel
    asymmetric_depolarize
    AsymmetricDepolarizingChannel
    bit_flip
    BitFlipChannel
    depolarize
    DepolarizingChannel
    generalized_amplitude_damp
    GeneralizedAmplitudeDampingChannel
    phase_damp
    phase_flip
    PhaseDampingChannel
    PhaseFlipChannel


Other Gate and Operation Classes
''''''''''''''''''''''''''''''''

Generic classes for creating new kinds of gates and operations.

.. autosummary::
    :toctree: generated/

    ControlledGate
    ControlledOperation
    EigenGate
    Gate
    GateOperation
    InterchangeableQubitsGate
    LinearCombinationOfGates
    Operation
    SingleQubitGate
    ThreeQubitGate
    TwoQubitGate


Pauli and Clifford Group Concepts
'''''''''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    Pauli
    PauliInteractionGate
    PauliString
    PauliTransform
    SingleQubitCliffordGate


Displays
''''''''


.. autosummary::
    :toctree: generated/

    ApproxPauliStringExpectation
    pauli_string_expectation
    DensityMatrixDisplay
    PauliStringExpectation
    SamplesDisplay
    WaveFunctionDisplay


Circuits and Schedules
''''''''''''''''''''''

Utilities for representing and manipulating quantum computations via
Circuits, Operations, and Moments.

.. autosummary::
    :toctree: generated/

    Circuit
    CircuitDag
    flatten_op_tree
    freeze_op_tree
    GateOperation
    InsertStrategy
    Moment
    moment_by_moment_schedule
    op_gate_of_type
    OP_TREE
    Operation
    ParallelGateOperation
    QubitOrder
    QubitOrderOrList
    Schedule
    ScheduledOperation
    transform_op_tree
    Unique


Trials and Simulations
''''''''''''''''''''''

Classes for simulations and results.

.. autosummary::
    :toctree: generated/

    bloch_vector_from_state_vector
    density_matrix_from_state_vector
    DensityMatrixSimulator
    DensityMatrixSimulatorState
    DensityMatrixStepResult
    DensityMatrixTrialResult
    dirac_notation
    measure_density_matrix
    measure_state_vector
    sample
    sample_density_matrix
    sample_state_vector
    sample_sweep
    SimulatesFinalState
    SimulatesIntermediateState
    SimulatesIntermediateWaveFunction
    SimulatesSamples
    SimulationTrialResult
    Simulator
    SparseSimulatorStep
    StateVectorMixin
    StepResult
    TrialResult
    to_valid_density_matrix
    to_valid_state_vector
    validate_normalized_state
    validate_probability
    WaveFunctionSimulatorState
    WaveFunctionStepResult
    WaveFunctionTrialResult


Parameterization
''''''''''''''''

Handling of parameterized values.

.. autosummary::
    :toctree: generated/

    Linspace
    ParamResolver
    plot_state_histogram
    Points
    Sweep
    Sweepable
    to_resolvers
    UnitSweep

Magic Method Protocols
''''''''''''''''''''''

Utility methods for accessing generic functionality exposed by some gates,
operations, and other types.

.. autosummary::
    :toctree: generated/

    apply_channel
    apply_unitary
    approx_eq
    channel
    control
    circuit_diagram_info
    decompose
    decompose_once
    decompose_once_with_qubits
    has_channel
    has_mixture
    has_mixture_channel
    has_unitary
    inverse
    is_measurement
    is_parameterized
    measurement_key
    mixture
    mixture_channel
    mul
    pauli_expansion
    phase_by
    pow
    qasm
    resolve_parameters
    trace_distance_bound
    unitary
    validate_mixture

Magic Method Protocol Types
'''''''''''''''''''''''''''

Classes defining and used by the magic method protocols.

.. autosummary::
    :toctree: generated/

    ApplyChannelArgs
    ApplyUnitaryArgs
    CircuitDiagramInfo
    CircuitDiagramInfoArgs
    QasmArgs
    QasmOutput
    SupportsApplyChannel
    SupportsApplyUnitary
    SupportsApproximateEquality
    SupportsChannel
    SupportsCircuitDiagramInfo
    SupportsDecompose
    SupportsDecomposeWithQubits
    SupportsMixture
    SupportsParameterization
    SupportsPhase
    SupportsQasm
    SupportsQasmWithArgs
    SupportsQasmWithArgsAndQubits
    SupportsTraceDistanceBound
    SupportsUnitary


Optimization
''''''''''''

Classes and methods for rewriting circuits.

.. autosummary::
    :toctree: generated/

    ConvertToCzAndSingleGates
    DropEmptyMoments
    DropNegligible
    EjectPhasedPaulis
    EjectZ
    ExpandComposite
    google.optimized_for_xmon
    merge_single_qubit_gates_into_phased_x_z
    MergeInteractions
    MergeSingleQubitGates
    PointOptimizationSummary
    PointOptimizer
    single_qubit_matrix_to_gates
    single_qubit_matrix_to_pauli_rotations
    single_qubit_matrix_to_phased_x_z
    single_qubit_op_to_framed_phase_form
    two_qubit_matrix_to_operations


Utilities
'''''''''

General utility methods, mostly related to performing relevant linear algebra
operations and decompositions.

.. autosummary::
    :toctree: generated/

    allclose_up_to_global_phase
    apply_matrix_to_slices
    bidiagonalize_real_matrix_pair_with_symmetric_products
    bidiagonalize_unitary_with_special_orthogonals
    block_diag
    commutes
    canonicalize_half_turns
    chosen_angle_to_canonical_half_turns
    chosen_angle_to_half_turns
    commutes
    CONTROL_TAG
    diagonalize_real_symmetric_and_sorted_diagonal_matrices
    diagonalize_real_symmetric_matrix
    dot
    Duration
    expand_matrix_in_orthogonal_basis
    hilbert_schmidt_inner_product
    is_diagonal
    is_hermitian
    is_negligible_turn
    is_orthogonal
    is_special_orthogonal
    is_special_unitary
    is_unitary
    kak_canonicalize_vector
    kak_decomposition
    KakDecomposition
    kron
    kron_factor_4x4_to_2x2s
    kron_with_controls
    LinearDict
    map_eigenvalues
    match_global_phase
    matrix_from_basis_coefficients
    partial_trace
    PeriodicValue
    reflection_matrix_pow
    slice_for_qubits_equal_to
    so4_to_magic_su2s
    targeted_conjugate_about
    targeted_left_multiply
    TextDiagramDrawer
    Timestamp
    value_equality


Experiments
'''''''''''

Utilities for running experiments on hardware, or producing things required to
run experiments.

.. autosummary::
    :toctree: generated/

    generate_supremacy_circuit_google_v2
    generate_supremacy_circuit_google_v2_bristlecone
    generate_supremacy_circuit_google_v2_grid


Ion traps and neutral atoms
'''''''''''''''''''''''''''

Support for ion trap an neutral atom devices.

.. autosummary::
    :toctree: generated/

    ConvertToIonGates
    IonDevice
    MS
    two_qubit_matrix_to_ion_operations
    ConvertToNeutralAtomGates
    NeutralAtomDevice



Google
''''''

Functionality specific to quantum hardware and services from Google.

.. autosummary::
    :toctree: generated/

    google.AnnealSequenceSearchStrategy
    google.Bristlecone
    google.ConvertToXmonGates
    google.Engine
    google.engine_from_environment
    google.Foxtail
    google.gate_to_proto_dict
    google.GreedySequenceSearchStrategy
    google.is_native_xmon_op
    google.JobConfig
    google.line_on_device
    google.LinePlacementStrategy
    google.optimized_for_xmon
    google.pack_results
    google.schedule_from_proto_dicts
    google.schedule_to_proto_dicts
    google.unpack_results
    google.xmon_op_from_proto_dict
    google.XmonDevice
    google.XmonOptions
    google.XmonSimulator
    google.XmonStepResult


Testing
'''''''

Functionality for writing unit tests involving objects from Cirq, and also some
general testing utilities.

.. autosummary::
    :toctree: generated/

    testing.assert_allclose_up_to_global_phase
    testing.assert_circuits_with_terminal_measurements_are_equivalent
    testing.assert_decompose_is_consistent_with_unitary
    testing.assert_eigen_gate_has_consistent_apply_unitary
    testing.assert_eigengate_implements_consistent_protocols
    testing.assert_equivalent_repr
    testing.assert_has_consistent_apply_unitary
    testing.assert_has_consistent_apply_unitary_for_various_exponents
    testing.assert_has_diagram
    testing.assert_implements_consistent_protocols
    testing.assert_pauli_expansion_is_consistent_with_unitary
    testing.assert_phase_by_is_consistent_with_unitary
    testing.assert_qasm_is_consistent_with_unitary
    testing.assert_same_circuits
    testing.EqualsTester
    testing.highlight_text_differences
    testing.nonoptimal_toffoli_circuit
    testing.only_test_in_python3
    testing.OrderTester
    testing.random_circuit
    testing.random_orthogonal
    testing.random_special_orthogonal
    testing.random_special_unitary
    testing.random_superposition
    testing.random_unitary
    testing.TempDirectoryPath
    testing.TempFilePath


Contrib
'''''''

Contributed code that requires extra dependencies to be installed, code that may
be unstable, and code that may or may not be a fit for the main library. A
waiting area.

.. autosummary::
    :toctree: generated/

    contrib.acquaintance
    contrib.paulistring
    contrib.qcircuit
    contrib.quirk
    contrib.tpu
