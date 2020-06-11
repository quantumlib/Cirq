<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.transformations" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RaiseValueErrorIfNotProvided"/>
<meta itemprop="property" content="TDefault"/>
</div>

# Module: cirq.linalg.transformations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utility methods for transforming matrices or vectors.



## Functions

[`apply_matrix_to_slices(...)`](../../cirq/linalg/apply_matrix_to_slices.md): Left-multiplies an NxN matrix onto N slices of a numpy array.

[`match_global_phase(...)`](../../cirq/linalg/match_global_phase.md): Phases the given matrices so that they agree on the phase of one entry.

[`partial_trace(...)`](../../cirq/linalg/partial_trace.md): Takes the partial trace of a given tensor.

[`partial_trace_of_state_vector_as_mixture(...)`](../../cirq/linalg/partial_trace_of_state_vector_as_mixture.md): Returns a mixture representing a state vector with only some qubits kept.

[`reflection_matrix_pow(...)`](../../cirq/linalg/reflection_matrix_pow.md): Raises a matrix with two opposing eigenvalues to a power.

[`sub_state_vector(...)`](../../cirq/linalg/sub_state_vector.md): Attempts to factor a state vector into two parts and return one of them.

[`subwavefunction(...)`](../../cirq/linalg/subwavefunction.md): THIS FUNCTION IS DEPRECATED.

[`targeted_conjugate_about(...)`](../../cirq/linalg/targeted_conjugate_about.md): Conjugates the given tensor about the target tensor.

[`targeted_left_multiply(...)`](../../cirq/linalg/targeted_left_multiply.md): Left-multiplies the given axes of the target tensor by the given matrix.

[`wavefunction_partial_trace_as_mixture(...)`](../../cirq/linalg/wavefunction_partial_trace_as_mixture.md): THIS FUNCTION IS DEPRECATED.

## Other Members

* `RaiseValueErrorIfNotProvided` <a id="RaiseValueErrorIfNotProvided"></a>
* `TDefault` <a id="TDefault"></a>
