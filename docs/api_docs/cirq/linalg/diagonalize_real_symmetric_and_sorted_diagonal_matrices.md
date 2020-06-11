<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.diagonalize_real_symmetric_and_sorted_diagonal_matrices" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.diagonalize_real_symmetric_and_sorted_diagonal_matrices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/diagonalize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an orthogonal matrix that diagonalizes both given matrices.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices`, `cirq.linalg.diagonalize.diagonalize_real_symmetric_and_sorted_diagonal_matrices`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.diagonalize_real_symmetric_and_sorted_diagonal_matrices(
    symmetric_matrix: np.ndarray,
    diagonal_matrix: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

The given matrices must commute.
Guarantees that the sorted diagonal matrix is not permuted by the
diagonalization (except for nearly-equal values).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`symmetric_matrix`
</td>
<td>
A real symmetric matrix.
</td>
</tr><tr>
<td>
`diagonal_matrix`
</td>
<td>
A real diagonal matrix with entries along the diagonal
sorted into descending order.
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
If set, verifies that the input matrices commute
and are respectively symmetric and diagonal descending.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An orthogonal matrix P such that P.T @ symmetric_matrix @ P is diagonal
and P.T @ diagonal_matrix @ P = diagonal_matrix (up to tolerance).
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
Matrices don't meet preconditions (e.g. not symmetric).
</td>
</tr>
</table>

