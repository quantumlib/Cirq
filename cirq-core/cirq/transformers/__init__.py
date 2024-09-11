# Copyright 2021 The Cirq Developers
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

"""Classes and methods for transforming circuits."""

from cirq.transformers.analytical_decompositions import (
    # pylint: disable=line-too-long
    compute_cphase_exponents_for_fsim_decomposition as compute_cphase_exponents_for_fsim_decomposition,
    decompose_cphase_into_two_fsim as decompose_cphase_into_two_fsim,
    decompose_clifford_tableau_to_operations as decompose_clifford_tableau_to_operations,
    decompose_multi_controlled_x as decompose_multi_controlled_x,
    decompose_multi_controlled_rotation as decompose_multi_controlled_rotation,
    decompose_two_qubit_interaction_into_four_fsim_gates as decompose_two_qubit_interaction_into_four_fsim_gates,
    is_negligible_turn as is_negligible_turn,
    parameterized_2q_op_to_sqrt_iswap_operations as parameterized_2q_op_to_sqrt_iswap_operations,
    prepare_two_qubit_state_using_cz as prepare_two_qubit_state_using_cz,
    prepare_two_qubit_state_using_iswap as prepare_two_qubit_state_using_iswap,
    prepare_two_qubit_state_using_sqrt_iswap as prepare_two_qubit_state_using_sqrt_iswap,
    quantum_shannon_decomposition as quantum_shannon_decomposition,
    single_qubit_matrix_to_gates as single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations as single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z as single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz as single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form as single_qubit_op_to_framed_phase_form,
    three_qubit_matrix_to_operations as three_qubit_matrix_to_operations,
    two_qubit_matrix_to_cz_isometry as two_qubit_matrix_to_cz_isometry,
    two_qubit_matrix_to_cz_operations as two_qubit_matrix_to_cz_operations,
    two_qubit_matrix_to_diagonal_and_cz_operations as two_qubit_matrix_to_diagonal_and_cz_operations,
    two_qubit_matrix_to_ion_operations as two_qubit_matrix_to_ion_operations,
    two_qubit_matrix_to_sqrt_iswap_operations as two_qubit_matrix_to_sqrt_iswap_operations,
    unitary_to_pauli_string as unitary_to_pauli_string,
)

from cirq.transformers.heuristic_decompositions import (
    TwoQubitGateTabulation as TwoQubitGateTabulation,
    TwoQubitGateTabulationResult as TwoQubitGateTabulationResult,
    two_qubit_gate_product_tabulation as two_qubit_gate_product_tabulation,
)

from cirq.transformers.routing import (
    AbstractInitialMapper as AbstractInitialMapper,
    HardCodedInitialMapper as HardCodedInitialMapper,
    LineInitialMapper as LineInitialMapper,
    MappingManager as MappingManager,
    RouteCQC as RouteCQC,
    routed_circuit_with_mapping as routed_circuit_with_mapping,
)

from cirq.transformers.target_gatesets import (
    create_transformer_with_kwargs as create_transformer_with_kwargs,
    CompilationTargetGateset as CompilationTargetGateset,
    CZTargetGateset as CZTargetGateset,
    SqrtIswapTargetGateset as SqrtIswapTargetGateset,
    TwoQubitCompilationTargetGateset as TwoQubitCompilationTargetGateset,
)

from cirq.transformers.align import align_left as align_left, align_right as align_right

from cirq.transformers.stratify import stratified_circuit as stratified_circuit

from cirq.transformers.expand_composite import expand_composite as expand_composite

from cirq.transformers.eject_phased_paulis import eject_phased_paulis as eject_phased_paulis

from cirq.transformers.optimize_for_target_gateset import (
    optimize_for_target_gateset as optimize_for_target_gateset,
)

from cirq.transformers.drop_empty_moments import drop_empty_moments as drop_empty_moments

from cirq.transformers.drop_negligible_operations import (
    drop_negligible_operations as drop_negligible_operations,
)

from cirq.transformers.dynamical_decoupling import (
    add_dynamical_decoupling as add_dynamical_decoupling,
)

from cirq.transformers.eject_z import eject_z as eject_z

from cirq.transformers.measurement_transformers import (
    defer_measurements as defer_measurements,
    dephase_measurements as dephase_measurements,
    drop_terminal_measurements as drop_terminal_measurements,
)

from cirq.transformers.merge_k_qubit_gates import merge_k_qubit_unitaries as merge_k_qubit_unitaries

from cirq.transformers.merge_single_qubit_gates import (
    merge_single_qubit_gates_to_phased_x_and_z as merge_single_qubit_gates_to_phased_x_and_z,
    merge_single_qubit_gates_to_phxz as merge_single_qubit_gates_to_phxz,
    merge_single_qubit_moments_to_phxz as merge_single_qubit_moments_to_phxz,
)

from cirq.transformers.qubit_management_transformers import (
    map_clean_and_borrowable_qubits as map_clean_and_borrowable_qubits,
)

from cirq.transformers.synchronize_terminal_measurements import (
    synchronize_terminal_measurements as synchronize_terminal_measurements,
)

from cirq.transformers.transformer_api import (
    LogLevel as LogLevel,
    TRANSFORMER as TRANSFORMER,
    TransformerContext as TransformerContext,
    TransformerLogger as TransformerLogger,
    transformer as transformer,
)

from cirq.transformers.transformer_primitives import (
    map_moments as map_moments,
    map_operations as map_operations,
    map_operations_and_unroll as map_operations_and_unroll,
    merge_k_qubit_unitaries_to_circuit_op as merge_k_qubit_unitaries_to_circuit_op,
    merge_moments as merge_moments,
    merge_operations as merge_operations,
    merge_operations_to_circuit_op as merge_operations_to_circuit_op,
    toggle_tags as toggle_tags,
    unroll_circuit_op as unroll_circuit_op,
    unroll_circuit_op_greedy_earliest as unroll_circuit_op_greedy_earliest,
    unroll_circuit_op_greedy_frontier as unroll_circuit_op_greedy_frontier,
)

from cirq.transformers.gauge_compiling import (
    CZGaugeTransformer as CZGaugeTransformer,
    ConstantGauge as ConstantGauge,
    Gauge as Gauge,
    GaugeSelector as GaugeSelector,
    GaugeTransformer as GaugeTransformer,
    ISWAPGaugeTransformer as ISWAPGaugeTransformer,
    SpinInversionGaugeTransformer as SpinInversionGaugeTransformer,
    SqrtCZGaugeTransformer as SqrtCZGaugeTransformer,
    SqrtISWAPGaugeTransformer as SqrtISWAPGaugeTransformer,
)

from cirq.transformers.randomized_measurements import (
    RandomizedMeasurements as RandomizedMeasurements,
)
