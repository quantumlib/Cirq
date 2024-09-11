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
"""Linear algebra methods and classes useful for quantum primitives.

Focuses on methods useful for analyzing and optimizing quantum circuits.
Avoids duplicating functionality present in numpy.
"""

from cirq.linalg.combinators import (
    block_diag as block_diag,
    CONTROL_TAG as CONTROL_TAG,
    dot as dot,
    kron as kron,
    kron_with_controls as kron_with_controls,
)

from cirq.linalg.decompositions import (
    # pylint: disable=line-too-long
    axis_angle as axis_angle,
    AxisAngleDecomposition as AxisAngleDecomposition,
    deconstruct_single_qubit_matrix_into_angles as deconstruct_single_qubit_matrix_into_angles,
    extract_right_diag as extract_right_diag,
    kak_canonicalize_vector as kak_canonicalize_vector,
    kak_decomposition as kak_decomposition,
    kak_vector as kak_vector,
    KakDecomposition as KakDecomposition,
    kron_factor_4x4_to_2x2s as kron_factor_4x4_to_2x2s,
    map_eigenvalues as map_eigenvalues,
    num_cnots_required as num_cnots_required,
    unitary_eig as unitary_eig,
    scatter_plot_normalized_kak_interaction_coefficients as scatter_plot_normalized_kak_interaction_coefficients,
    so4_to_magic_su2s as so4_to_magic_su2s,
)

from cirq.linalg.diagonalize import (
    # pylint: disable=line-too-long
    bidiagonalize_real_matrix_pair_with_symmetric_products as bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals as bidiagonalize_unitary_with_special_orthogonals,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices as diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix as diagonalize_real_symmetric_matrix,
)

from cirq.linalg.operator_spaces import (
    expand_matrix_in_orthogonal_basis as expand_matrix_in_orthogonal_basis,
    hilbert_schmidt_inner_product as hilbert_schmidt_inner_product,
    kron_bases as kron_bases,
    matrix_from_basis_coefficients as matrix_from_basis_coefficients,
    PAULI_BASIS as PAULI_BASIS,
    pow_pauli_combination as pow_pauli_combination,
)

from cirq.linalg.predicates import (
    allclose_up_to_global_phase as allclose_up_to_global_phase,
    is_cptp as is_cptp,
    is_diagonal as is_diagonal,
    is_hermitian as is_hermitian,
    is_normal as is_normal,
    is_orthogonal as is_orthogonal,
    is_special_orthogonal as is_special_orthogonal,
    is_special_unitary as is_special_unitary,
    is_unitary as is_unitary,
    matrix_commutes as matrix_commutes,
    slice_for_qubits_equal_to as slice_for_qubits_equal_to,
)

from cirq.linalg.tolerance import (
    all_near_zero as all_near_zero,
    all_near_zero_mod as all_near_zero_mod,
)

from cirq.linalg.transformations import (
    apply_matrix_to_slices as apply_matrix_to_slices,
    density_matrix_kronecker_product as density_matrix_kronecker_product,
    match_global_phase as match_global_phase,
    partial_trace as partial_trace,
    partial_trace_of_state_vector_as_mixture as partial_trace_of_state_vector_as_mixture,
    reflection_matrix_pow as reflection_matrix_pow,
    state_vector_kronecker_product as state_vector_kronecker_product,
    sub_state_vector as sub_state_vector,
    targeted_conjugate_about as targeted_conjugate_about,
    targeted_left_multiply as targeted_left_multiply,
    to_special as to_special,
    transpose_flattened_array as transpose_flattened_array,
    can_numpy_support_shape as can_numpy_support_shape,
)
