<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.math_utils.weyl_chamber_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.two_qubit_gates.math_utils.weyl_chamber_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/math_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Cubic mesh of points in the Weyl chamber.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.math_utils.weyl_chamber_mesh(
    spacing: float
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`spacing`
</td>
<td>
Euclidean distance between neighboring KAK vectors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
np.ndarray of shape (N,3) corresponding to the points in the Weyl
chamber.
</td>
</tr>

</table>

