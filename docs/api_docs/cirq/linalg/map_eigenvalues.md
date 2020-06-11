<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.map_eigenvalues" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.map_eigenvalues

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies a function to the eigenvalues of a matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.decompositions.map_eigenvalues`, `cirq.map_eigenvalues`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.map_eigenvalues(
    matrix: np.ndarray,
    func: Callable[[complex], complex],
    *,
    atol: float = 1e-08
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matrix`
</td>
<td>
The matrix to modify with the function.
</td>
</tr><tr>
<td>
`func`
</td>
<td>
The function to apply to the eigenvalues of the matrix.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Relative threshold used when separating eigenspaces.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute threshold used when separating eigenspaces.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The transformed matrix.
</td>
</tr>

</table>

