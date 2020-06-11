<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.sample_density_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.sample_density_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Samples repeatedly from measurements in the computational basis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.sample_density_matrix`, `cirq.sim.density_matrix_utils.sample_density_matrix`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.sample_density_matrix(
    density_matrix: np.ndarray,
    indices: List[int],
    *,
    qid_shape: Optional[Tuple[int, ...]] = None,
    repetitions: int = 1,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Note that this does not modify the density_matrix.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`density_matrix`
</td>
<td>
The density matrix to be measured. This matrix is
assumed to be positive semidefinite and trace one. The matrix is
assumed to be of shape (2 ** integer, 2 ** integer) or
(2, 2, ..., 2).
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
Which qubits are measured. The density matrix rows and columns
are assumed to be supplied in big endian order. That is the
xth index of v, when expressed as a bitstring, has its largest
values in the 0th index.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The qid shape of the density matrix.  Specify this argument
when using qudits.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample the density matrix.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A seed for the pseudorandom number generator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Measurement results with True corresponding to the ``|1⟩`` state.
The outer list is for repetitions, and the inner corresponds to
measurements ordered by the supplied qubits. These lists
are wrapped as an numpy ndarray.
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
``repetitions`` is less than one or size of ``matrix`` is
not a power of 2.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
An index from ``indices`` is out of range, given the number
of qubits corresponding to the density matrix.
</td>
</tr>
</table>

