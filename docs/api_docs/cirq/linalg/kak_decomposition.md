<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kak_decomposition" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kak_decomposition

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kak_decomposition`, `cirq.linalg.decompositions.kak_decomposition`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kak_decomposition(
    unitary_object: Union[np.ndarray, 'cirq.SupportsUnitary'],
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> <a href="../../cirq/linalg/KakDecomposition.md"><code>cirq.linalg.KakDecomposition</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`unitary_object`
</td>
<td>
The value to decompose. Can either be a 4x4 unitary
matrix, or an object that has a 4x4 unitary matrix (via the
<a href="../../cirq/protocols/SupportsUnitary.md"><code>cirq.SupportsUnitary</code></a> protocol).
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Per-matrix-entry relative tolerance on equality.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Per-matrix-entry absolute tolerance on equality.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
If set, verifies that the input corresponds to a
4x4 unitary before decomposing.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../cirq/linalg/KakDecomposition.md"><code>cirq.KakDecomposition</code></a> canonicalized such that the interaction
coefficients x, y, z satisfy:

0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
if x2 = π/4, z2 >= 0
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
Bad matrix.
</td>
</tr><tr>
<td>
`ArithmeticError`
</td>
<td>
Failed to perform the decomposition.
</td>
</tr>
</table>



#### References:

'An Introduction to Cartan's KAK Decomposition for QC Programmers'
https://arxiv.org/abs/quant-ph/0507171
