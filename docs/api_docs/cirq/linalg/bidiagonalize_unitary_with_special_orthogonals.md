<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.bidiagonalize_unitary_with_special_orthogonals" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.bidiagonalize_unitary_with_special_orthogonals

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/diagonalize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.bidiagonalize_unitary_with_special_orthogonals`, `cirq.linalg.diagonalize.bidiagonalize_unitary_with_special_orthogonals`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.bidiagonalize_unitary_with_special_orthogonals(
    mat: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> Tuple[np.ndarray, np.array, np.ndarray]
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
A unitary matrix.
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
If set, verifies that the input is a unitary matrix
(to the given tolerances). Defaults to set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
be orthogonal matrices with determinant equal to 1.
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

