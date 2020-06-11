<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.diagonalize_real_symmetric_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.diagonalize_real_symmetric_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/diagonalize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an orthogonal matrix that diagonalizes the given matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.diagonalize_real_symmetric_matrix`, `cirq.linalg.diagonalize.diagonalize_real_symmetric_matrix`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.diagonalize_real_symmetric_matrix(
    matrix: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matrix`
</td>
<td>
A real symmetric matrix to diagonalize.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Relative error tolerance.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute error tolerance.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
If set, verifies that the input matrix is real and
symmetric.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An orthogonal matrix P such that P.T @ matrix @ P is diagonal.
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
Matrix isn't real symmetric.
</td>
</tr>
</table>

