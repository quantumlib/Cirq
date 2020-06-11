<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.so4_to_magic_su2s" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.so4_to_magic_su2s

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.decompositions.so4_to_magic_su2s`, `cirq.so4_to_magic_su2s`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.so4_to_magic_su2s(
    mat: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> Tuple[np.ndarray, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

Mag is the magic basis matrix:

    1  0  0  i
    0  i  1  0
    0  i -1  0     (times sqrt(0.5) to normalize)
    1  0  0 -i

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mat`
</td>
<td>
A real 4x4 orthogonal matrix.
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
When set, the code verifies that the given
matrix is from SO(4). Defaults to set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
is approximately equal to the given matrix.
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
</tr>
</table>

