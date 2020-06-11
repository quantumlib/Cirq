<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.two_qubit_matrix_to_operations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.two_qubit_matrix_to_operations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/two_qubit_decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a two-qubit operation into Z/XY/CZ gates.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.two_qubit_decompositions.two_qubit_matrix_to_operations`, `cirq.two_qubit_matrix_to_operations`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.two_qubit_matrix_to_operations(
    q0: "cirq.Qid",
    q1: "cirq.Qid",
    mat: np.ndarray,
    allow_partial_czs: bool,
    atol: float = 1e-08,
    clean_operations: bool = True
) -> List[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`q0`
</td>
<td>
The first qubit being operated on.
</td>
</tr><tr>
<td>
`q1`
</td>
<td>
The other qubit being operated on.
</td>
</tr><tr>
<td>
`mat`
</td>
<td>
Defines the operation to apply to the pair of qubits.
</td>
</tr><tr>
<td>
`allow_partial_czs`
</td>
<td>
Enables the use of Partial-CZ gates.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
A limit on the amount of absolute error introduced by the
construction.
</td>
</tr><tr>
<td>
`clean_operations`
</td>
<td>
Enables optimizing resulting operation list by
merging operations and ejecting phased Paulis and Z operations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of operations implementing the matrix.
</td>
</tr>

</table>

