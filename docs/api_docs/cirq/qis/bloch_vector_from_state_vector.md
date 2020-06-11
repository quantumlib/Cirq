<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.bloch_vector_from_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.bloch_vector_from_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the bloch vector of a qubit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.bloch_vector_from_state_vector`, `cirq.qis.states.bloch_vector_from_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.bloch_vector_from_state_vector(
    state_vector: Sequence,
    index: int,
    qid_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Calculates the bloch vector of the qubit at index in the state vector,
assuming state vector follows the standard Kronecker convention of
numpy.kron.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
A sequence representing a state vector in which
the ordering mapping to qubits follows the standard Kronecker
convention of numpy.kron (big-endian).
</td>
</tr><tr>
<td>
`index`
</td>
<td>
index of qubit who's bloch vector we want to find.
follows the standard Kronecker convention of numpy.kron.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
specifies the dimensions of the qudits for the input
`state_vector`.  If not specified, qubits are assumed and the
`state_vector` must have a dimension a power of two.
The qudit at `index` must be a qubit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A length 3 numpy array representing the qubit's bloch vector.
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
if the size of `state_vector `is not a power of 2 and the
shape is not given or if the shape is given and `state_vector` has
a size that contradicts this shape.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
if index is out of range for the number of qubits or qudits
corresponding to `state_vector`.
</td>
</tr>
</table>

