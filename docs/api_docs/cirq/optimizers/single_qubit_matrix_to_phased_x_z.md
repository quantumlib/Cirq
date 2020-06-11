<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.single_qubit_matrix_to_phased_x_z" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.single_qubit_matrix_to_phased_x_z

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements a single-qubit operation with a PhasedX and Z gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.decompositions.single_qubit_matrix_to_phased_x_z`, `cirq.single_qubit_matrix_to_phased_x_z`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.single_qubit_matrix_to_phased_x_z(
    mat: np.ndarray,
    atol: float = 0
) -> List[<a href="../../cirq/ops/SingleQubitGate.md"><code>cirq.ops.SingleQubitGate</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

If one of the gates isn't needed, it will be omitted.

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
A list of gates that, when applied in order, perform the desired
operation.
</td>
</tr>

</table>

