<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kron_factor_4x4_to_2x2s" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kron_factor_4x4_to_2x2s

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kron_factor_4x4_to_2x2s`, `cirq.linalg.decompositions.kron_factor_4x4_to_2x2s`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kron_factor_4x4_to_2x2s(
    matrix: np.ndarray
) -> Tuple[complex, np.ndarray, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

Requires the matrix to be the kronecker product of two 2x2 unitaries.
Requires the matrix to have a non-zero determinant.
Giving an incorrect matrix will cause garbage output.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matrix`
</td>
<td>
The 4x4 unitary matrix to factor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar factor and a pair of 2x2 unit-determinant matrices. The
kronecker product of all three is equal to the given matrix.
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
The given matrix can't be tensor-factored into 2x2 pieces.
</td>
</tr>
</table>

