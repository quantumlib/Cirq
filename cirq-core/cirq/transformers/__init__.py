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

"""Circuit transformation utilities."""

from cirq.transformers.analytical_decompositions import (
    compute_cphase_exponents_for_fsim_decomposition,
    decompose_cphase_into_two_fsim,
    decompose_clifford_tableau_to_operations,
    decompose_multi_controlled_x,
    decompose_multi_controlled_rotation,
    decompose_two_qubit_interaction_into_four_fsim_gates,
    is_negligible_turn,
    prepare_two_qubit_state_using_cz,
    prepare_two_qubit_state_using_sqrt_iswap,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form,
    three_qubit_matrix_to_operations,
    two_qubit_matrix_to_diagonal_and_operations,
    two_qubit_matrix_to_operations,
    two_qubit_matrix_to_sqrt_iswap_operations,
)

from cirq.transformers.heuristic_decompositions import (
    TwoQubitGateTabulation,
    TwoQubitGateTabulationResult,
    two_qubit_gate_product_tabulation,
)

from cirq.transformers.transformer_primitives import (
    map_moments,
    map_operations,
    map_operations_and_unroll,
    merge_moments,
    merge_operations,
    unroll_circuit_op,
    unroll_circuit_op_greedy_earliest,
    unroll_circuit_op_greedy_frontier,
)
