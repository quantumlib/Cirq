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

"""Utilities for analytical decomposition of cirq gates."""

from cirq.transformers.analytical_decompositions.clifford_decomposition import (
    decompose_clifford_tableau_to_operations as decompose_clifford_tableau_to_operations,
)

from cirq.transformers.analytical_decompositions.controlled_gate_decomposition import (
    decompose_multi_controlled_x as decompose_multi_controlled_x,
    decompose_multi_controlled_rotation as decompose_multi_controlled_rotation,
)

from cirq.transformers.analytical_decompositions.cphase_to_fsim import (
    # pylint: disable=line-too-long
    compute_cphase_exponents_for_fsim_decomposition as compute_cphase_exponents_for_fsim_decomposition,
    decompose_cphase_into_two_fsim as decompose_cphase_into_two_fsim,
)

from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
    quantum_shannon_decomposition as quantum_shannon_decomposition,
)

from cirq.transformers.analytical_decompositions.single_qubit_decompositions import (
    is_negligible_turn as is_negligible_turn,
    single_qubit_matrix_to_gates as single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations as single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z as single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz as single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form as single_qubit_op_to_framed_phase_form,
)

from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
    three_qubit_matrix_to_operations as three_qubit_matrix_to_operations,
)

from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
    # pylint: disable=line-too-long
    two_qubit_matrix_to_cz_operations as two_qubit_matrix_to_cz_operations,
    two_qubit_matrix_to_diagonal_and_cz_operations as two_qubit_matrix_to_diagonal_and_cz_operations,
)

from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
    # pylint: disable=line-too-long
    decompose_two_qubit_interaction_into_four_fsim_gates as decompose_two_qubit_interaction_into_four_fsim_gates,
)

from cirq.transformers.analytical_decompositions.two_qubit_to_ms import (
    two_qubit_matrix_to_ion_operations as two_qubit_matrix_to_ion_operations,
)

from cirq.transformers.analytical_decompositions.two_qubit_to_sqrt_iswap import (
    parameterized_2q_op_to_sqrt_iswap_operations as parameterized_2q_op_to_sqrt_iswap_operations,
    two_qubit_matrix_to_sqrt_iswap_operations as two_qubit_matrix_to_sqrt_iswap_operations,
)

from cirq.transformers.analytical_decompositions.two_qubit_state_preparation import (
    prepare_two_qubit_state_using_cz as prepare_two_qubit_state_using_cz,
    prepare_two_qubit_state_using_sqrt_iswap as prepare_two_qubit_state_using_sqrt_iswap,
    prepare_two_qubit_state_using_iswap as prepare_two_qubit_state_using_iswap,
)

from cirq.transformers.analytical_decompositions.single_to_two_qubit_isometry import (
    two_qubit_matrix_to_cz_isometry as two_qubit_matrix_to_cz_isometry,
)

from cirq.transformers.analytical_decompositions.pauli_string_decomposition import (
    unitary_to_pauli_string as unitary_to_pauli_string,
)
