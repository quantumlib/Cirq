<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.single_qubit_matrix_to_pauli_rotations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.single_qubit_matrix_to_pauli_rotations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements a single-qubit operation with few rotations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.decompositions.single_qubit_matrix_to_pauli_rotations`, `cirq.single_qubit_matrix_to_pauli_rotations`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.single_qubit_matrix_to_pauli_rotations(
    mat: np.ndarray,
    atol: float = 0
) -> List[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, float]]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mat`
</td>
<td>
The 2x2 unitary matrix of the operation to implement.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
A limit on the amount of absolute error introduced by the
construction.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of (Pauli, half_turns) tuples that, when applied in order,
perform the desired operation.
</td>
</tr>

</table>

