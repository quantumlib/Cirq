<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.to_valid_density_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.to_valid_density_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Verifies the density_matrix_rep is valid and converts it to ndarray form.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.qis.states.to_valid_density_matrix`, `cirq.to_valid_density_matrix`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.to_valid_density_matrix(
    density_matrix_rep: Union[int, np.ndarray],
    num_qubits: Optional[int] = None,
    *,
    qid_shape: Optional[Tuple[int, ...]] = None,
    dtype: Type[np.number] = np.complex64,
    atol: float = 1e-07
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

This method is used to support passing a matrix, a state vector,
or a computational basis state as a representation of a state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`density_matrix_rep`
</td>
<td>
If an numpy array, if it is of rank 2 (a matrix),
then this is the density matrix. If it is a numpy array of rank 1
(a vector) then this is a state vector. If this is an int,
then this is the computation basis state.
</td>
</tr><tr>
<td>
`num_qubits`
</td>
<td>
The number of qubits for the density matrix. The
density_matrix_rep must be valid for this number of qubits.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The qid shape of the state vector. Specify this argument
when using qudits.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype of the density matrix, will be used when creating
the state for a computational basis state (int), or validated
against if density_matrix_rep is a numpy array.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Numerical tolerance for verifying density matrix properties.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A numpy matrix corresponding to the density matrix on the given number
of qubits.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if the density_matrix_rep is not valid.
</td>
</tr>

</table>

