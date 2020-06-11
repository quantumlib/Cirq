<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.single_qubit_op_to_framed_phase_form" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.single_qubit_op_to_framed_phase_form

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a 2x2 unitary M into U^-1 * diag(1, r) * U * diag(g, g).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.optimizers.decompositions.single_qubit_op_to_framed_phase_form`, `cirq.single_qubit_op_to_framed_phase_form`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.single_qubit_op_to_framed_phase_form(
    mat: np.ndarray
) -> Tuple[np.ndarray, complex, complex]
</code></pre>



<!-- Placeholder for "Used in" -->

U translates the rotation axis of M to the Z axis.
g fixes a global phase factor difference caused by the translation.
r's phase is the amount of rotation around M's rotation axis.

This decomposition can be used to decompose controlled single-qubit
rotations into controlled-Z operations bordered by single-qubit operations.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mat`
</td>
<td>
The qubit operation as a 2x2 unitary matrix.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 2x2 unitary U, the complex relative phase factor r, and the complex
global phase factor g. Applying M is equivalent (up to global phase) to
applying U, rotating around the Z axis to apply r, then un-applying U.
When M is controlled, the control must be rotated around the Z axis to
apply g.
</td>
</tr>

</table>

