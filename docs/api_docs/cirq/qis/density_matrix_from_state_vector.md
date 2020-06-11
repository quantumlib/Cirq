<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.density_matrix_from_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.density_matrix_from_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the density matrix of the state vector.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.density_matrix_from_state_vector`, `cirq.qis.states.density_matrix_from_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.density_matrix_from_state_vector(
    state_vector: Sequence,
    indices: Optional[Iterable[int]] = None,
    qid_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Calculate the density matrix for the system on the given qubit indices,
with the qubits not in indices that are present in state vector traced out.
If indices is None the full density matrix for `state_vector` is returned.
We assume `state_vector` follows the standard Kronecker convention of
numpy.kron (big-endian).

#### For example:


state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
indices = None
gives us

    $$
    \rho = \begin{bmatrix}
            0.5 & 0.5 \\
            0.5 & 0.5
    \end{bmatrix}
    $$

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
`indices`
</td>
<td>
list containing indices for qubits that you would like
to include in the density matrix (i.e.) qubits that WON'T
be traced out. follows the standard Kronecker convention of
numpy.kron.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
specifies the dimensions of the qudits for the input
`state_vector`.  If not specified, qubits are assumed and the
`state_vector` must have a dimension a power of two.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A numpy array representing the density matrix.
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
if the size of `state_vector` is not a power of 2 and the
shape is not given or if the shape is given and `state_vector`
has a size that contradicts this shape.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
if the indices are out of range for the number of qubits
corresponding to `state_vector`.
</td>
</tr>
</table>

