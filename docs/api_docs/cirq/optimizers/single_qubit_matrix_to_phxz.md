<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.single_qubit_matrix_to_phxz" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.single_qubit_matrix_to_phxz

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements a single-qubit operation with a PhasedXZ gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.decompositions.single_qubit_matrix_to_phxz`, `cirq.single_qubit_matrix_to_phxz`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.single_qubit_matrix_to_phxz(
    mat: np.ndarray,
    atol: float = 0
) -> Optional[<a href="../../cirq/ops/PhasedXZGate.md"><code>cirq.ops.PhasedXZGate</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

Under the hood, this uses deconstruct_single_qubit_matrix_into_angles which
converts the given matrix to a series of three rotations around the Z, Y, Z
axes. This is then converted to a phased X rotation followed by a Z, in the
form of a single PhasedXZ gate.

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
A PhasedXZ gate that implements the given matrix, or None if it is
close to identity (trace distance <= atol).
</td>
</tr>

</table>

