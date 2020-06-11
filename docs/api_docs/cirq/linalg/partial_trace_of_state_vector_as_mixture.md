<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.partial_trace_of_state_vector_as_mixture" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.partial_trace_of_state_vector_as_mixture

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a mixture representing a state vector with only some qubits kept.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.partial_trace_of_state_vector_as_mixture`, `cirq.partial_trace_of_state_vector_as_mixture`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.partial_trace_of_state_vector_as_mixture(
    state_vector: np.ndarray,
    keep_indices: List[int],
    *,
    atol: Union[int, float] = 1e-08
) -> Tuple[Tuple[float, np.ndarray], ...]
</code></pre>



<!-- Placeholder for "Used in" -->

The input state vector must have shape `(2,) * n` or `(2 ** n)` where
`state_vector` is expressed over n qubits. States in the output mixture will
retain the same type of shape as the input state vector, either `(2 ** k)`
or `(2,) * k` where k is the number of qubits kept.

If the state vector cannot be factored into a pure state over `keep_indices`
then eigendecomposition is used and the output mixture will not be unique.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
The state vector to take the partial trace over.
</td>
</tr><tr>
<td>
`keep_indices`
</td>
<td>
Which indices to take the partial trace of the
state_vector on.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
The tolerance for determining that a factored state is pure.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A single-component mixture in which the factored state vector has
probability '1' if the partially traced state is pure, or else a
mixture of the default eigendecomposition of the mixed state's
partial trace.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the input `state_vector` is not an array of length
`(2 ** n)` or a tensor with a shape of `(2,) * n`
</td>
</tr>
</table>

