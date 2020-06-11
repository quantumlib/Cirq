<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ion.two_qubit_matrix_to_ion_operations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ion.two_qubit_matrix_to_ion_operations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_decomposition.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a two-qubit operation into MS/single-qubit rotation gates.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ion.ion_decomposition.two_qubit_matrix_to_ion_operations`, `cirq.two_qubit_matrix_to_ion_operations`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ion.two_qubit_matrix_to_ion_operations(
    q0: "cirq.Qid",
    q1: "cirq.Qid",
    mat: np.ndarray,
    atol: float = 1e-08
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
`tolerance`
</td>
<td>
A limit on the amount of error introduced by the
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
A list of operations implementing the matrix.
</td>
</tr>

</table>

