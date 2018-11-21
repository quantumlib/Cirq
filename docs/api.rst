.. currentmodule:: cirq

API Reference
=============


Single Qubit Gates
''''''''''''''''''

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



Two Qubit Gates
''''''''''''''''

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


Three Qubit Gates
''''''''''''''''''

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


Other Gate and Operation Classes
''''''''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated

    ControlledGate
    EigenGate
    Gate
    GateOperation
    InterchangeableQubitsGate
    Operation
    ReversibleCompositeGate
    SingleQubitGate
    ThreeQubitGate
    TwoQubitGate


Circuits and Schedules
''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    Circuit
    flatten_op_tree
    freeze_op_tree
    InsertStrategy
    Moment
    moment_by_moment_schedule
    OP_TREE
    QubitOrder
    QubitOrderOrList
    Schedule
    ScheduledOperation
    transform_op_tree


Devices and Qubits
''''''''''''''''''

General classes for qubits and related concepts.

.. autosummary::
    :toctree: generated/

    Device
    GridQubit
    LineQubit
    NamedQubit
    QubitId
    UnconstrainedDevice


Trials and Simulations
''''''''''''''''''''''

Classes for parameterized circuits.

.. autosummary::
    :toctree: generated/

    bloch_vector_from_state_vector
    density_matrix_from_state_vector
    dirac_notation
    Linspace
    measure_state_vector
    ParamResolver
    plot_state_histogram
    Points
    sample_state_vector
    SimulatesSamples
    SimulationTrialResult
    Simulator
    SimulatorStep
    StepResult
    SimulatesFinalWaveFunction
    SimulatesIntermediateWaveFunction
    Sweep
    Sweepable
    to_valid_state_vector
    validate_normalized_state
    to_resolvers
    TrialResult
    UnitSweep


Magic Method Protocols
''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    apply_unitary
    circuit_diagram_info
    decompose
    decompose_once
    decompose_once_with_qubits
    inverse
    mul
    pow
    qasm
    is_parameterized
    resolve_parameters
    has_unitary
    unitary
    trace_distance_bound
    phase_by

Magic Method Protocol Types
'''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    CircuitDiagramInfo
    CircuitDiagramInfoArgs
    QasmArgs
    QasmOutput
    SupportsApplyUnitary
    SupportsCircuitDiagramInfo
    SupportsDecompose
    SupportsDecomposeWithQubits
    SupportsParameterization
    SupportsPhase
    SupportsQasm
    SupportsQasmWithArgs
    SupportsQasmWithArgsAndQubits
    SupportsTraceDistanceBound
    SupportsUnitary


Optimization
''''''''''''

Classes and methods for optimizing circuits.

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
    OptimizationPass
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
    canonicalize_half_turns
    chosen_angle_to_canonical_half_turns
    chosen_angle_to_half_turns
    slice_for_qubits_equal_to
    block_diag
    match_global_phase
    commutes
    CONTROL_TAG
    diagonalize_real_symmetric_and_sorted_diagonal_matrices
    diagonalize_real_symmetric_matrix
    dot
    Duration
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
    map_eigenvalues
    reflection_matrix_pow
    so4_to_magic_su2s
    Symbol
    targeted_left_multiply
    TextDiagramDrawer
    Timestamp
    Tolerance
    value_equality


Experiments
'''''''''''

.. autosummary::
    :toctree: generated/

    generate_supremacy_circuit_google_v2
    generate_supremacy_circuit_google_v2_bristlecone
    generate_supremacy_circuit_google_v2_grid




Google
''''''

Functionality specific to quantum hardware and services from Google.

    google.AnnealSequenceSearchStrategy
    google.GreedySequenceSearchStrategy
    google.Bristlecone
    google.ConvertToXmonGates
    google.Engine
    google.engine_from_environment
    google.Foxtail
    google.gate_to_proto_dict
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

    assert_allclose_up_to_global_phase
    assert_circuits_with_terminal_measurements_are_equivalent
    assert_decompose_is_consistent_with_unitary
    assert_eigen_gate_has_consistent_apply_unitary
    assert_equivalent_repr
    assert_has_consistent_apply_unitary
    assert_has_consistent_apply_unitary_for_various_exponents
    assert_has_diagram
    assert_phase_by_is_consistent_with_unitary
    assert_qasm_is_consistent_with_unitary
    assert_same_circuits
    EqualsTester
    highlight_text_differences
    nonoptimal_toffoli_circuit
    only_test_in_python3
    OrderTester
    random_circuit
    random_orthogonal
    random_special_orthogonal
    random_special_unitary
    random_unitary
    TempDirectoryPath
    TempFilePath


Work in Progress - Noisy Channels
'''''''''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    amplitude_damp
    AmplitudeDampingChannel
    asymmetric_depolarize
    AsymmetricDepolarizingChannel
    bit_flip
    BitFlipChannel
    channel
    depolarize
    DepolarizingChannel
    generalized_amplitude_damp
    GeneralizedAmplitudeDampingChannel
    phase_damp
    PhaseDampingChannel
    phase_flip
    PhaseFlipChannel
    rotation_error
    RotationErrorChannel
    SupportsChannel


Work in Progress - Stabilizers
''''''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    CircuitDag
    SingleQubitCliffordGate
    Pauli
    PauliInteractionGate
    PauliString
    PauliTransform
    Unique
