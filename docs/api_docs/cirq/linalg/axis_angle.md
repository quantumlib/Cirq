<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.axis_angle" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.axis_angle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a single-qubit unitary into axis, angle, and global phase.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.axis_angle`, `cirq.linalg.decompositions.axis_angle`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.axis_angle(
    single_qubit_unitary: np.ndarray
) -> <a href="../../cirq/linalg/AxisAngleDecomposition.md"><code>cirq.linalg.AxisAngleDecomposition</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`single_qubit_unitary`
</td>
<td>
The unitary of the single-qubit operation to
decompose.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An AxisAngleDecomposition equivalent to the given unitary.
</td>
</tr>

</table>

