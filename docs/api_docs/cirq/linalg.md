<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CONTROL_TAG"/>
<meta itemprop="property" content="PAULI_BASIS"/>
</div>

# Module: cirq.linalg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Types and methods related to performing linear algebra.


Focuses on methods useful for analyzing and optimizing quantum circuits.
Avoids duplicating functionality present in numpy.

## Modules

[`combinators`](../cirq/linalg/combinators.md) module: Utility methods for combining matrices.

[`decompositions`](../cirq/linalg/decompositions.md) module: Utility methods for breaking matrices into useful pieces.

[`diagonalize`](../cirq/linalg/diagonalize.md) module: Utility methods for diagonalizing matrices.

[`operator_spaces`](../cirq/linalg/operator_spaces.md) module: Utilities for manipulating linear operators as elements of vector space.

[`predicates`](../cirq/linalg/predicates.md) module: Utility methods for checking properties of matrices.

[`states`](../cirq/linalg/states.md) module: Utility methods for creating vectors and matrices.

[`tolerance`](../cirq/linalg/tolerance.md) module: Utility for testing approximate equality of matrices and scalars within

[`transformations`](../cirq/linalg/transformations.md) module: Utility methods for transforming matrices or vectors.

## Classes

[`class AxisAngleDecomposition`](../cirq/linalg/AxisAngleDecomposition.md): Represents a unitary operation as an axis, angle, and global phase.

[`class KakDecomposition`](../cirq/linalg/KakDecomposition.md): A convenient description of an arbitrary two-qubit operation.

## Functions

[`all_near_zero(...)`](../cirq/linalg/all_near_zero.md): Checks if the tensor's elements are all near zero.

[`all_near_zero_mod(...)`](../cirq/linalg/all_near_zero_mod.md): Checks if the tensor's elements are all near multiples of the period.

[`allclose_up_to_global_phase(...)`](../cirq/linalg/allclose_up_to_global_phase.md): Determines if a ~= b * exp(i t) for some t.

[`apply_matrix_to_slices(...)`](../cirq/linalg/apply_matrix_to_slices.md): Left-multiplies an NxN matrix onto N slices of a numpy array.

[`axis_angle(...)`](../cirq/linalg/axis_angle.md): Decomposes a single-qubit unitary into axis, angle, and global phase.

[`bidiagonalize_real_matrix_pair_with_symmetric_products(...)`](../cirq/linalg/bidiagonalize_real_matrix_pair_with_symmetric_products.md): Finds orthogonal matrices that diagonalize both mat1 and mat2.

[`bidiagonalize_unitary_with_special_orthogonals(...)`](../cirq/linalg/bidiagonalize_unitary_with_special_orthogonals.md): Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

[`block_diag(...)`](../cirq/linalg/block_diag.md): Concatenates blocks into a block diagonal matrix.

[`deconstruct_single_qubit_matrix_into_angles(...)`](../cirq/linalg/deconstruct_single_qubit_matrix_into_angles.md): Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

[`diagonalize_real_symmetric_and_sorted_diagonal_matrices(...)`](../cirq/linalg/diagonalize_real_symmetric_and_sorted_diagonal_matrices.md): Returns an orthogonal matrix that diagonalizes both given matrices.

[`diagonalize_real_symmetric_matrix(...)`](../cirq/linalg/diagonalize_real_symmetric_matrix.md): Returns an orthogonal matrix that diagonalizes the given matrix.

[`dot(...)`](../cirq/linalg/dot.md): Computes the dot/matrix product of a sequence of values.

[`expand_matrix_in_orthogonal_basis(...)`](../cirq/linalg/expand_matrix_in_orthogonal_basis.md): Computes coefficients of expansion of m in basis.

[`eye_tensor(...)`](../cirq/linalg/eye_tensor.md): THIS FUNCTION IS DEPRECATED.

[`hilbert_schmidt_inner_product(...)`](../cirq/linalg/hilbert_schmidt_inner_product.md): Computes Hilbert-Schmidt inner product of two matrices.

[`is_diagonal(...)`](../cirq/linalg/is_diagonal.md): Determines if a matrix is a approximately diagonal.

[`is_hermitian(...)`](../cirq/linalg/is_hermitian.md): Determines if a matrix is approximately Hermitian.

[`is_normal(...)`](../cirq/linalg/is_normal.md): Determines if a matrix is approximately normal.

[`is_orthogonal(...)`](../cirq/linalg/is_orthogonal.md): Determines if a matrix is approximately orthogonal.

[`is_special_orthogonal(...)`](../cirq/linalg/is_special_orthogonal.md): Determines if a matrix is approximately special orthogonal.

[`is_special_unitary(...)`](../cirq/linalg/is_special_unitary.md): Determines if a matrix is approximately unitary with unit determinant.

[`is_unitary(...)`](../cirq/linalg/is_unitary.md): Determines if a matrix is approximately unitary.

[`kak_canonicalize_vector(...)`](../cirq/linalg/kak_canonicalize_vector.md): Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

[`kak_decomposition(...)`](../cirq/linalg/kak_decomposition.md): Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

[`kak_vector(...)`](../cirq/linalg/kak_vector.md): Compute the KAK vectors of one or more two qubit unitaries.

[`kron(...)`](../cirq/linalg/kron.md): Computes the kronecker product of a sequence of values.

[`kron_bases(...)`](../cirq/linalg/kron_bases.md): Creates tensor product of bases.

[`kron_factor_4x4_to_2x2s(...)`](../cirq/linalg/kron_factor_4x4_to_2x2s.md): Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

[`kron_with_controls(...)`](../cirq/linalg/kron_with_controls.md): Computes the kronecker product of a sequence of values and control tags.

[`map_eigenvalues(...)`](../cirq/linalg/map_eigenvalues.md): Applies a function to the eigenvalues of a matrix.

[`match_global_phase(...)`](../cirq/linalg/match_global_phase.md): Phases the given matrices so that they agree on the phase of one entry.

[`matrix_commutes(...)`](../cirq/linalg/matrix_commutes.md): Determines if two matrices approximately commute.

[`matrix_from_basis_coefficients(...)`](../cirq/linalg/matrix_from_basis_coefficients.md): Computes linear combination of basis vectors with given coefficients.

[`one_hot(...)`](../cirq/linalg/one_hot.md): THIS FUNCTION IS DEPRECATED.

[`partial_trace(...)`](../cirq/linalg/partial_trace.md): Takes the partial trace of a given tensor.

[`partial_trace_of_state_vector_as_mixture(...)`](../cirq/linalg/partial_trace_of_state_vector_as_mixture.md): Returns a mixture representing a state vector with only some qubits kept.

[`pow_pauli_combination(...)`](../cirq/linalg/pow_pauli_combination.md): Computes non-negative integer power of single-qubit Pauli combination.

[`reflection_matrix_pow(...)`](../cirq/linalg/reflection_matrix_pow.md): Raises a matrix with two opposing eigenvalues to a power.

[`scatter_plot_normalized_kak_interaction_coefficients(...)`](../cirq/linalg/scatter_plot_normalized_kak_interaction_coefficients.md): Plots the interaction coefficients of many two-qubit operations.

[`slice_for_qubits_equal_to(...)`](../cirq/linalg/slice_for_qubits_equal_to.md): Returns an index corresponding to a desired subset of an np.ndarray.

[`so4_to_magic_su2s(...)`](../cirq/linalg/so4_to_magic_su2s.md): Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

[`sub_state_vector(...)`](../cirq/linalg/sub_state_vector.md): Attempts to factor a state vector into two parts and return one of them.

[`subwavefunction(...)`](../cirq/linalg/subwavefunction.md): THIS FUNCTION IS DEPRECATED.

[`targeted_conjugate_about(...)`](../cirq/linalg/targeted_conjugate_about.md): Conjugates the given tensor about the target tensor.

[`targeted_left_multiply(...)`](../cirq/linalg/targeted_left_multiply.md): Left-multiplies the given axes of the target tensor by the given matrix.

[`unitary_eig(...)`](../cirq/linalg/unitary_eig.md): Gives the guaranteed unitary eigendecomposition of a normal matrix.

[`wavefunction_partial_trace_as_mixture(...)`](../cirq/linalg/wavefunction_partial_trace_as_mixture.md): THIS FUNCTION IS DEPRECATED.

## Other Members

* `CONTROL_TAG` <a id="CONTROL_TAG"></a>
* `PAULI_BASIS` <a id="PAULI_BASIS"></a>
