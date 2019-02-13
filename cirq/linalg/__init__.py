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

"""Types and methods related to performing linear algebra.

Focuses on methods useful for analyzing and optimizing quantum circuits.
Avoids duplicating functionality present in numpy.
"""

from cirq.linalg.combinators import (
    block_diag,
    CONTROL_TAG,
    dot,
    kron,
    kron_with_controls,
)
from cirq.linalg.decompositions import (
    deconstruct_single_qubit_matrix_into_angles,
    kak_canonicalize_vector,
    kak_decomposition,
    KakDecomposition,
    kron_factor_4x4_to_2x2s,
    map_eigenvalues,
    so4_to_magic_su2s,
)
from cirq.linalg.diagonalize import (
    bidiagonalize_real_matrix_pair_with_symmetric_products,
    bidiagonalize_unitary_with_special_orthogonals,
    diagonalize_real_symmetric_and_sorted_diagonal_matrices,
    diagonalize_real_symmetric_matrix,
)
from cirq.linalg.predicates import (
    allclose_up_to_global_phase,
    slice_for_qubits_equal_to,
    commutes,
    is_diagonal,
    is_hermitian,
    is_orthogonal,
    is_special_orthogonal,
    is_special_unitary,
    is_unitary,
)
from cirq.linalg.tolerance import (
    Tolerance,
)
from cirq.linalg.transformations import (
    apply_matrix_to_slices,
    match_global_phase,
    reflection_matrix_pow,
    targeted_left_multiply,
)
