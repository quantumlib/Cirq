
API Reference
=============


Devices and Qubits
''''''''''''''''''

Classes for identifying the qubits and hardware you want to operate on.

.. autosummary::
    :toctree: generated/

    cirq.UNCONSTRAINED_DEVICE
    cirq.Device
    cirq.GridQid
    cirq.GridQubit
    cirq.LineQid
    cirq.LineQubit
    cirq.NamedQid
    cirq.NamedQubit
    cirq.Qid


Measurement
'''''''''''

Methods and classes for performing measurement operations.

.. autosummary::
    :toctree: generated/

    cirq.measure
    cirq.measure_each
    cirq.MeasurementGate


Unitary Gates and Operations
''''''''''''''''''''''''''''

Unitary effects that can be applied to one or more qubits.

.. autosummary::
    :toctree: generated/

    cirq.H
    cirq.I
    cirq.S
    cirq.T
    cirq.X
    cirq.Y
    cirq.Z
    cirq.CX
    cirq.CZ
    cirq.XX
    cirq.YY
    cirq.ZZ
    cirq.rx
    cirq.ry
    cirq.rz
    cirq.CCNOT
    cirq.CCX
    cirq.CCZ
    cirq.CNOT
    cirq.CSWAP
    cirq.FREDKIN
    cirq.ISWAP
    cirq.SWAP
    cirq.TOFFOLI
    cirq.cphase
    cirq.givens
    cirq.identity_each
    cirq.qft
    cirq.riswap
    cirq.wait
    .. autoclass:: cirq.CCNotPowGate
    cirq.CCXPowGate
    cirq.CCZPowGate
    .. autoclass:: cirq.CNotPowGate
    cirq.CSwapGate
    cirq.CXPowGate
    cirq.CZPowGate
    cirq.ControlledGate
    cirq.ControlledOperation
    cirq.EigenGate
    cirq.FSimGate
    cirq.Gate
    cirq.GlobalPhaseOperation
    cirq.HPowGate
    cirq.ISwapPowGate
    cirq.IdentityGate
    cirq.MatrixGate
    cirq.Operation
    cirq.PhaseGradientGate
    cirq.PhasedFSimGate
    cirq.PhasedISwapPowGate
    cirq.PhasedXPowGate
    cirq.PhasedXZGate
    cirq.QuantumFourierTransformGate
    cirq.QubitPermutationGate
    cirq.RandomGateChannel
    cirq.SingleQubitGate
    cirq.SwapPowGate
    cirq.TaggedOperation
    cirq.ThreeQubitDiagonalGate
    cirq.ThreeQubitGate
    cirq.TwoQubitDiagonalGate
    cirq.TwoQubitGate
    cirq.WaitGate
    cirq.XPowGate
    cirq.XXPowGate
    cirq.YPowGate
    cirq.YYPowGate
    cirq.ZPowGate
    cirq.ZZPowGate


Noisy Gates and Operations
''''''''''''''''''''''''''

Non-unitary gates. Mixture gates are those that can be interpreted
as applying a unitary for a fixed probability while channel encompasses
the more general concept of a noisy open system quantum evolution.

.. autosummary::
    :toctree: generated/

    cirq.NOISE_MODEL_LIKE
    cirq.NO_NOISE
    cirq.amplitude_damp
    cirq.asymmetric_depolarize
    cirq.bit_flip
    cirq.depolarize
    cirq.generalized_amplitude_damp
    cirq.phase_damp
    cirq.phase_flip
    cirq.reset
    cirq.AmplitudeDampingChannel
    cirq.AsymmetricDepolarizingChannel
    cirq.BitFlipChannel
    cirq.DepolarizingChannel
    cirq.GeneralizedAmplitudeDampingChannel
    cirq.NoiseModel
    cirq.PhaseDampingChannel
    cirq.PhaseFlipChannel
    cirq.ResetChannel
    cirq.VirtualTag


Pauli and Clifford Groups
'''''''''''''''''''''''''

Classes and methods related to representing and operating on states using sums
and products of Pauli operations.

.. autosummary::
    :toctree: generated/

    cirq.PAULI_BASIS
    cirq.PAULI_GATE_LIKE
    cirq.PAULI_STRING_LIKE
    cirq.pow_pauli_combination
    cirq.BaseDensePauliString
    cirq.CliffordState
    cirq.CliffordTableau
    cirq.DensePauliString
    cirq.MutableDensePauliString
    cirq.MutablePauliString
    cirq.Pauli
    cirq.PauliInteractionGate
    cirq.PauliString
    cirq.PauliStringGateOperation
    cirq.PauliStringPhasor
    cirq.PauliSum
    cirq.PauliSumLike
    cirq.PauliTransform
    cirq.SingleQubitCliffordGate
    cirq.StabilizerStateChForm


Circuits
''''''''

Utilities for representing and manipulating quantum computations via
Circuits, Operations, and Moments.

.. autosummary::
    :toctree: generated/

    cirq.OP_TREE
    cirq.flatten_op_tree
    cirq.freeze_op_tree
    cirq.transform_op_tree
    cirq.AbstractCircuit
    cirq.Circuit
    cirq.CircuitDag
    cirq.FrozenCircuit
    cirq.GateOperation
    cirq.InsertStrategy
    cirq.Moment
    cirq.ParallelGateOperation
    cirq.QubitOrder
    cirq.QubitOrderOrList
    cirq.Unique


Importing and Exporting
'''''''''''''''''''''''

Utilities for interoperating with other quantum software libraries and products.

.. autosummary::
    :toctree: generated/

    cirq.quirk_json_to_circuit
    cirq.quirk_url_to_circuit


Sampling, Simulations, and Data Collection
''''''''''''''''''''''''''''''''''''''''''

Objects for collecting data about a quantum circuit. Includes methods and
classes for defining parameter sweeps, performing simulations, and analyzing
results.

.. autosummary::
    :toctree: generated/

    cirq.CIRCUIT_LIKE
    cirq.RANDOM_STATE_OR_SEED_LIKE
    cirq.big_endian_bits_to_int
    cirq.big_endian_digits_to_int
    cirq.big_endian_int_to_bits
    cirq.big_endian_int_to_digits
    cirq.dict_to_product_sweep
    cirq.dict_to_zip_sweep
    cirq.final_density_matrix
    cirq.final_state_vector
    cirq.flatten
    cirq.flatten_to_ops
    cirq.flatten_to_ops_or_moments
    cirq.flatten_with_params
    cirq.flatten_with_sweep
    cirq.hog_score_xeb_fidelity_from_probabilities
    cirq.measure_density_matrix
    cirq.measure_state_vector
    cirq.sample
    cirq.sample_density_matrix
    cirq.sample_state_vector
    cirq.sample_sweep
    cirq.to_resolvers
    cirq.to_sweep
    cirq.to_sweeps
    cirq.validate_mixture
    cirq.validate_probability
    cirq.xeb_fidelity
    cirq.ActOnCliffordTableauArgs
    cirq.ActOnStabilizerCHFormArgs
    cirq.ActOnStateVectorArgs
    cirq.CircuitSampleJob
    cirq.CliffordSimulator
    cirq.CliffordSimulatorStepResult
    cirq.CliffordTrialResult
    cirq.Collector
    cirq.DensityMatrixSimulator
    cirq.DensityMatrixSimulatorState
    cirq.DensityMatrixStepResult
    cirq.DensityMatrixTrialResult
    cirq.ExpressionMap
    cirq.Linspace
    cirq.ListSweep
    .. autoclass:: cirq.ParamDictType
    cirq.ParamResolver
    cirq.ParamResolverOrSimilarType
    cirq.PauliSumCollector
    cirq.Points
    cirq.Product
    cirq.Result
    cirq.Sampler
    cirq.SimulatesAmplitudes
    cirq.SimulatesFinalState
    cirq.SimulatesIntermediateState
    cirq.SimulatesIntermediateStateVector
    cirq.SimulatesSamples
    cirq.SimulationTrialResult
    cirq.Simulator
    cirq.SparseSimulatorStep
    cirq.StabilizerSampler
    cirq.StateVectorMixin
    cirq.StateVectorSimulatorState
    cirq.StateVectorStepResult
    cirq.StateVectorTrialResult
    cirq.StepResult
    cirq.Sweep
    cirq.Sweepable
    cirq.UnitSweep
    cirq.ZerosSampler
    cirq.Zip


Visualization
'''''''''''''

Classes and methods for visualizing data.

.. autosummary::
    :toctree: generated/

    cirq.plot_state_histogram
    cirq.scatter_plot_normalized_kak_interaction_coefficients
    cirq.Heatmap
    cirq.TextDiagramDrawer


Magic Method Protocols
''''''''''''''''''''''

A magic method is a special named method, like `_unitary_`, that a class can
implement in order to indicate it supports certain functionality. There will be
a corresponding global method, such as `cirq.unitary`, for easily accessing this
functionality.

Classes that being with `Supports` are templates demonstrating and documenting
the magic methods that can be implemented.

.. autosummary::
    :toctree: generated/

    cirq.DEFAULT_RESOLVERS
    cirq.act_on
    cirq.apply_channel
    cirq.apply_mixture
    cirq.apply_unitaries
    cirq.apply_unitary
    cirq.approx_eq
    cirq.channel
    cirq.circuit_diagram_info
    cirq.compute_cphase_exponents_for_fsim_decomposition
    cirq.decompose
    cirq.decompose_cphase_into_two_fsim
    cirq.decompose_once
    cirq.decompose_once_with_qubits
    cirq.equal_up_to_global_phase
    cirq.has_channel
    cirq.has_mixture
    cirq.has_stabilizer_effect
    cirq.has_unitary
    cirq.inverse
    cirq.is_measurement
    cirq.is_parameterized
    cirq.measurement_key
    cirq.measurement_keys
    cirq.mixture
    cirq.mul
    cirq.num_qubits
    cirq.parameter_names
    cirq.parameter_symbols
    cirq.pauli_expansion
    cirq.phase_by
    cirq.pow
    cirq.qasm
    cirq.qid_shape
    cirq.quil
    cirq.read_json
    cirq.resolve_parameters
    cirq.to_json
    cirq.trace_distance_bound
    cirq.trace_distance_from_angle_list
    cirq.unitary
    cirq.with_measurement_key_mapping
    cirq.ApplyChannelArgs
    cirq.ApplyMixtureArgs
    cirq.ApplyUnitaryArgs
    cirq.CircuitDiagramInfo
    cirq.CircuitDiagramInfoArgs
    cirq.QasmArgs
    cirq.QasmOutput
    cirq.QuilFormatter
    cirq.QuilOutput
    cirq.SupportsActOn
    cirq.SupportsApplyChannel
    cirq.SupportsApplyMixture
    cirq.SupportsApproximateEquality
    cirq.SupportsChannel
    cirq.SupportsCircuitDiagramInfo
    cirq.SupportsCommutes
    cirq.SupportsConsistentApplyUnitary
    cirq.SupportsDecompose
    cirq.SupportsDecomposeWithQubits
    cirq.SupportsEqualUpToGlobalPhase
    cirq.SupportsExplicitHasUnitary
    cirq.SupportsExplicitNumQubits
    cirq.SupportsExplicitQidShape
    cirq.SupportsJSON
    cirq.SupportsMeasurementKey
    cirq.SupportsMixture
    cirq.SupportsParameterization
    cirq.SupportsPauliExpansion
    cirq.SupportsPhase
    cirq.SupportsQasm
    cirq.SupportsQasmWithArgs
    cirq.SupportsQasmWithArgsAndQubits
    cirq.SupportsTraceDistanceBound
    cirq.SupportsUnitary


Optimization
''''''''''''

Classes and methods for rewriting circuits.

.. autosummary::
    :toctree: generated/

    cirq.decompose_multi_controlled_rotation
    cirq.decompose_multi_controlled_x
    cirq.decompose_two_qubit_interaction_into_four_fsim_gates
    cirq.decompose_two_qubit_interaction_into_four_fsim_gates_via_b
    cirq.merge_single_qubit_gates_into_phased_x_z
    cirq.merge_single_qubit_gates_into_phxz
    cirq.single_qubit_matrix_to_gates
    cirq.single_qubit_matrix_to_pauli_rotations
    cirq.single_qubit_matrix_to_phased_x_z
    cirq.single_qubit_matrix_to_phxz
    cirq.single_qubit_op_to_framed_phase_form
    cirq.stratified_circuit
    cirq.two_qubit_matrix_to_operations
    cirq.ConvertToCzAndSingleGates
    cirq.DropEmptyMoments
    cirq.DropNegligible
    cirq.EjectPhasedPaulis
    cirq.EjectZ
    cirq.ExpandComposite
    cirq.MergeInteractions
    cirq.MergeSingleQubitGates
    cirq.PointOptimizationSummary
    cirq.PointOptimizer
    cirq.SynchronizeTerminalMeasurements


Experiments
'''''''''''

Utilities for running experiments on hardware, or producing things required to
run experiments.

.. autosummary::
    :toctree: generated/

    cirq.estimate_single_qubit_readout_errors
    cirq.generate_boixo_2018_supremacy_circuits_v2
    cirq.generate_boixo_2018_supremacy_circuits_v2_bristlecone
    cirq.generate_boixo_2018_supremacy_circuits_v2_grid
    cirq.least_squares_xeb_fidelity_from_expectations
    cirq.least_squares_xeb_fidelity_from_probabilities
    cirq.linear_xeb_fidelity
    cirq.linear_xeb_fidelity_from_probabilities
    cirq.log_xeb_fidelity
    cirq.log_xeb_fidelity_from_probabilities
    cirq.experiments.GRID_ALIGNED_PATTERN
    cirq.experiments.GRID_STAGGERED_PATTERN
    cirq.experiments.build_entangling_layers
    cirq.experiments.collect_grid_parallel_two_qubit_xeb_data
    cirq.experiments.compute_grid_parallel_two_qubit_xeb_results
    .. autofunction:: cirq.experiments.cross_entropy_benchmarking
    cirq.experiments.get_state_tomography_data
    cirq.experiments.purity_from_probabilities
    cirq.experiments.rabi_oscillations
    cirq.experiments.random_rotations_between_grid_interaction_layers_circuit
    cirq.experiments.single_qubit_randomized_benchmarking
    cirq.experiments.single_qubit_state_tomography
    cirq.experiments.state_tomography
    cirq.experiments.t1_decay
    cirq.experiments.t2_decay
    cirq.experiments.two_qubit_randomized_benchmarking
    cirq.experiments.two_qubit_state_tomography
    cirq.experiments.CrossEntropyResult
    cirq.experiments.CrossEntropyResultDict
    cirq.experiments.GridInteractionLayer
    cirq.experiments.RabiResult
    cirq.experiments.RandomizedBenchMarkResult
    cirq.experiments.SingleQubitReadoutCalibrationResult
    cirq.experiments.StateTomographyExperiment
    cirq.experiments.T1DecayResult
    cirq.experiments.T2DecayResult
    cirq.experiments.TomographyResult


Ion traps and neutral atoms
'''''''''''''''''''''''''''

Support for ion trap an neutral atom devices.

.. autosummary::
    :toctree: generated/

    cirq.ms
    cirq.is_native_neutral_atom_gate
    cirq.is_native_neutral_atom_op
    cirq.two_qubit_matrix_to_ion_operations
    cirq.ConvertToIonGates
    cirq.ConvertToNeutralAtomGates
    cirq.IonDevice
    cirq.NeutralAtomDevice



Google
''''''

Functionality specific to quantum hardware and services from Google.

.. autosummary::
    :toctree: generated/

    cirq.google.FSIM_GATESET
    cirq.google.NAMED_GATESETS
    cirq.google.SQRT_ISWAP_GATESET
    cirq.google.SYC
    cirq.google.SYC_GATESET
    cirq.google.XMON
    cirq.google.arg_from_proto
    cirq.google.get_engine
    cirq.google.get_engine_calibration
    cirq.google.get_engine_device
    cirq.google.get_engine_sampler
    cirq.google.line_on_device
    cirq.google.optimized_for_sycamore
    cirq.google.optimized_for_xmon
    cirq.google.AnnealSequenceSearchStrategy
    cirq.google.Bristlecone
    cirq.google.Calibration
    cirq.google.CalibrationLayer
    cirq.google.CalibrationResult
    cirq.google.CalibrationTag
    cirq.google.ConvertToSqrtIswapGates
    cirq.google.ConvertToSycamoreGates
    cirq.google.ConvertToXmonGates
    cirq.google.DeserializingArg
    cirq.google.Engine
    cirq.google.EngineJob
    cirq.google.EngineProcessor
    cirq.google.EngineProgram
    cirq.google.EngineTimeSlot
    cirq.google.Foxtail
    cirq.google.GateOpDeserializer
    cirq.google.GateOpSerializer
    cirq.google.GateTabulation
    cirq.google.GreedySequenceSearchStrategy
    cirq.google.LinePlacementStrategy
    cirq.google.PhysicalZTag
    cirq.google.ProtoVersion
    cirq.google.QuantumEngineSampler
    cirq.google.SerializableDevice
    cirq.google.SerializableGateSet
    cirq.google.SerializingArg
    cirq.google.Sycamore
    cirq.google.Sycamore23
    cirq.google.SycamoreGate
    cirq.google.XmonDevice


Contrib
'''''''

Contributed code that is not yet considered stable, may not yet fit well with
the main library, and may require extra dependencies to be installed (via
``python -m pip install cirq[contrib]``). A waiting area. All packages within
contrib may change without notice.

.. autosummary::
    :toctree: generated/

    cirq.contrib.acquaintance
    cirq.contrib.paulistring
    cirq.contrib.qcircuit
    cirq.contrib.quil_import
    cirq.contrib.quirk


Coding and Testing Tools
''''''''''''''''''''''''

These objects are not relevant when simply constructing and sampling circuits,
but are useful for customization tasks like defining and validating a custom
operation.


.. autosummary::
    :toctree: generated/

    cirq.alternative
    cirq.json_serializable_dataclass
    cirq.obj_to_dict_helper
    cirq.value_equality
    cirq.ABCMetaImplementAnyOneOf
    cirq.ArithmeticOperation
    cirq.InterchangeableQubitsGate
    cirq.JsonResolver
    cirq.LinearDict
    cirq.PeriodicValue
    cirq.testing.DEFAULT_GATE_DOMAIN
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary
    cirq.testing.assert_allclose_up_to_global_phase
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent
    cirq.testing.assert_commutes_magic_method_consistent_with_unitaries
    cirq.testing.assert_consistent_resolve_parameters
    cirq.testing.assert_decompose_is_consistent_with_unitary
    cirq.testing.assert_eigengate_implements_consistent_protocols
    cirq.testing.assert_equivalent_computational_basis_map
    cirq.testing.assert_equivalent_repr
    cirq.testing.assert_has_consistent_apply_unitary
    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents
    cirq.testing.assert_has_consistent_qid_shape
    cirq.testing.assert_has_consistent_trace_distance_bound
    cirq.testing.assert_has_diagram
    cirq.testing.assert_implements_consistent_protocols
    cirq.testing.assert_json_roundtrip_works
    cirq.testing.assert_logs
    cirq.testing.assert_pauli_expansion_is_consistent_with_unitary
    cirq.testing.assert_phase_by_is_consistent_with_unitary
    cirq.testing.assert_qasm_is_consistent_with_unitary
    cirq.testing.assert_same_circuits
    cirq.testing.assert_specifies_has_unitary_if_unitary
    cirq.testing.asyncio_pending
    cirq.testing.highlight_text_differences
    cirq.testing.nonoptimal_toffoli_circuit
    .. autofunction:: cirq.testing.random_circuit
    cirq.testing.random_density_matrix
    cirq.testing.random_orthogonal
    cirq.testing.random_special_orthogonal
    cirq.testing.random_special_unitary
    cirq.testing.random_superposition
    cirq.testing.random_unitary
    cirq.testing.EqualsTester
    cirq.testing.NoIdentifierQubit
    cirq.testing.OrderTester


Algebra and Representation
''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    cirq.CONTROL_TAG
    cirq.DURATION_LIKE
    cirq.all_near_zero
    cirq.all_near_zero_mod
    cirq.allclose_up_to_global_phase
    cirq.apply_matrix_to_slices
    cirq.axis_angle
    cirq.bidiagonalize_real_matrix_pair_with_symmetric_products
    cirq.bidiagonalize_unitary_with_special_orthogonals
    cirq.block_diag
    cirq.canonicalize_half_turns
    cirq.chosen_angle_to_canonical_half_turns
    cirq.chosen_angle_to_half_turns
    cirq.commutes
    cirq.deconstruct_single_qubit_matrix_into_angles
    cirq.definitely_commutes
    cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices
    cirq.diagonalize_real_symmetric_matrix
    cirq.dot
    cirq.expand_matrix_in_orthogonal_basis
    cirq.hilbert_schmidt_inner_product
    cirq.is_diagonal
    cirq.is_hermitian
    cirq.is_negligible_turn
    cirq.is_normal
    cirq.is_orthogonal
    cirq.is_special_orthogonal
    cirq.is_special_unitary
    cirq.is_unitary
    cirq.kak_canonicalize_vector
    cirq.kak_decomposition
    cirq.kak_vector
    cirq.kron
    cirq.kron_bases
    cirq.kron_factor_4x4_to_2x2s
    cirq.kron_with_controls
    cirq.map_eigenvalues
    cirq.match_global_phase
    cirq.matrix_commutes
    cirq.matrix_from_basis_coefficients
    cirq.num_cnots_required
    cirq.partial_trace
    cirq.partial_trace_of_state_vector_as_mixture
    cirq.reflection_matrix_pow
    cirq.slice_for_qubits_equal_to
    cirq.so4_to_magic_su2s
    cirq.sub_state_vector
    cirq.targeted_conjugate_about
    cirq.targeted_left_multiply
    cirq.to_special
    cirq.unitary_eig
    cirq.AxisAngleDecomposition
    cirq.Duration
    cirq.KakDecomposition
    cirq.Timestamp


Quantum Information Science
'''''''''''''''''''''''''''

.. autosummary::
    :toctree: generated/

    cirq.KET_IMAG
    cirq.KET_MINUS
    cirq.KET_MINUS_IMAG
    cirq.KET_ONE
    cirq.KET_PLUS
    cirq.KET_ZERO
    cirq.PAULI_STATES
    cirq.STATE_VECTOR_LIKE
    cirq.bloch_vector_from_state_vector
    cirq.density_matrix_from_state_vector
    cirq.dirac_notation
    cirq.eye_tensor
    cirq.fidelity
    cirq.one_hot
    cirq.to_valid_density_matrix
    cirq.to_valid_state_vector
    cirq.validate_density_matrix
    cirq.validate_indices
    cirq.validate_normalized_state_vector
    cirq.validate_qid_shape
    cirq.von_neumann_entropy
    cirq.ProductState


Internal Implementation Details
'''''''''''''''''''''''''''''''

Neither users nor developers will commonly refer to these objects, but they play
important roles in the internal machinery of the library.

.. autosummary::
    :toctree: generated/

    cirq.ConstantQubitNoiseModel
    cirq.LinearCombinationOfGates
    cirq.LinearCombinationOfOperations
    cirq.SingleQubitPauliStringGateOperation
    cirq.TParamKey
    cirq.TParamVal


Deprecated
''''''''''

These objects and methods will be removed in a future version of the library.

.. autosummary::
    :toctree: generated/

    cirq.QFT
    cirq.final_wavefunction
    cirq.has_mixture_channel
    cirq.mixture_channel
    cirq.subwavefunction
    cirq.validate_normalized_state
    cirq.wavefunction_partial_trace_as_mixture
    cirq.SimulatesIntermediateWaveFunction
    cirq.TrialResult
    cirq.WaveFunctionSimulatorState
    cirq.WaveFunctionStepResult
    cirq.WaveFunctionTrialResult
    cirq.google.engine_from_environment
