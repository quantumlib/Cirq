<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.to_valid_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.to_valid_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Verifies the state_rep is valid and converts it to ndarray form.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.qis.states.to_valid_state_vector`, `cirq.to_valid_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.to_valid_state_vector(
    state_rep: "cirq.STATE_VECTOR_LIKE",
    num_qubits: Optional[int] = None,
    *,
    qid_shape: Optional[Sequence[int]] = None,
    dtype: Type[np.number] = np.complex64,
    atol: float = 1e-07
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

This method is used to support passing in an integer representing a
computational basis state or a full state vector as a representation of
a pure state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_rep`
</td>
<td>
If an int, the state vector returned is the state vector
corresponding to a computational basis state. If an numpy array
this is the full state vector. Both of these are validated for
the given number of qubits, and the state must be properly
normalized and of the appropriate dtype.
</td>
</tr><tr>
<td>
`num_qubits`
</td>
<td>
The number of qubits for the state vector. The state_rep
must be valid for this number of qubits.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The expected qid shape of the state vector. Specify this
argument when using qudits.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype of the state vector, will be used when creating
the state for a computational basis state, or validated against if
state_rep is a numpy array.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Numerical tolerance for verifying that the norm of the state
vector is close to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A numpy ndarray corresponding to the state vector on the given number of
qubits.
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
if `state_vector` is not valid or
num_qubits != len(qid_shape).
</td>
</tr>
</table>

