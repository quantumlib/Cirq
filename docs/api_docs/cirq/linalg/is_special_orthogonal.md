<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.is_special_orthogonal" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.is_special_orthogonal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/predicates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines if a matrix is approximately special orthogonal.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.is_special_orthogonal`, `cirq.linalg.predicates.is_special_orthogonal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.is_special_orthogonal(
    matrix: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

A matrix is special orthogonal if it is square and real and its transpose
is its inverse and its determinant is one.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matrix`
</td>
<td>
The matrix to check.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
The per-matrix-entry relative tolerance on equality.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
The per-matrix-entry absolute tolerance on equality.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Whether the matrix is special orthogonal within the given tolerance.
</td>
</tr>

</table>

