<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.deconstruct_single_qubit_matrix_into_angles" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.deconstruct_single_qubit_matrix_into_angles

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.deconstruct_single_qubit_matrix_into_angles`, `cirq.linalg.decompositions.deconstruct_single_qubit_matrix_into_angles`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.deconstruct_single_qubit_matrix_into_angles(
    mat: np.ndarray
) -> Tuple[float, float, float]
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
The 2x2 unitary matrix to break down.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple containing the amount to phase around Z, then rotate around Y,
then phase around Z (all in radians).
</td>
</tr>

</table>

