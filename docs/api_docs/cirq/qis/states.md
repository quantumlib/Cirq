<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.states" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.qis.states

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utility methods for creating vectors and matrices.



## Functions

[`bloch_vector_from_state_vector(...)`](../../cirq/qis/bloch_vector_from_state_vector.md): Returns the bloch vector of a qubit.

[`density_matrix_from_state_vector(...)`](../../cirq/qis/density_matrix_from_state_vector.md): Returns the density matrix of the state vector.

[`dirac_notation(...)`](../../cirq/qis/dirac_notation.md): Returns the state vector as a string in Dirac notation.

[`eye_tensor(...)`](../../cirq/qis/eye_tensor.md): Returns an identity matrix reshaped into a tensor.

[`one_hot(...)`](../../cirq/qis/one_hot.md): Returns a numpy array with all 0s and a single non-zero entry(default 1).

[`to_valid_density_matrix(...)`](../../cirq/qis/to_valid_density_matrix.md): Verifies the density_matrix_rep is valid and converts it to ndarray form.

[`to_valid_state_vector(...)`](../../cirq/qis/to_valid_state_vector.md): Verifies the state_rep is valid and converts it to ndarray form.

[`validate_indices(...)`](../../cirq/qis/validate_indices.md): Validates that the indices have values within range of num_qubits.

[`validate_normalized_state(...)`](../../cirq/qis/validate_normalized_state.md): THIS FUNCTION IS DEPRECATED.

[`validate_normalized_state_vector(...)`](../../cirq/qis/validate_normalized_state_vector.md): Validates that the given state vector is a valid.

[`validate_qid_shape(...)`](../../cirq/qis/validate_qid_shape.md): Validates the size of the given `state_vector` against the given shape.

## Type Aliases

[`STATE_VECTOR_LIKE`](../../cirq/qis/STATE_VECTOR_LIKE.md)

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
