<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.bidiagonalize_real_matrix_pair_with_symmetric_products" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.bidiagonalize_real_matrix_pair_with_symmetric_products

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/diagonalize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Finds orthogonal matrices that diagonalize both mat1 and mat2.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.bidiagonalize_real_matrix_pair_with_symmetric_products`, `cirq.linalg.diagonalize.bidiagonalize_real_matrix_pair_with_symmetric_products`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.bidiagonalize_real_matrix_pair_with_symmetric_products(
    mat1: np.ndarray,
    mat2: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> Tuple[np.ndarray, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

Requires mat1 and mat2 to be real.
Requires mat1.T @ mat2 to be symmetric.
Requires mat1 @ mat2.T to be symmetric.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mat1`
</td>
<td>
One of the real matrices.
</td>
</tr><tr>
<td>
`mat2`
</td>
<td>
The other real matrix.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Relative numeric error threshold.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute numeric error threshold.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
If set, verifies that the inputs are real, and that
mat1.T @ mat2 and mat1 @ mat2.T are both symmetric. Defaults to set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple (L, R) of two orthogonal matrices, such that both L @ mat1 @ R
and L @ mat2 @ R are diagonal matrices.
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
Matrices don't meet preconditions (e.g. not real).
</td>
</tr>
</table>

