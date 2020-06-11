<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.unitary_eig" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.unitary_eig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gives the guaranteed unitary eigendecomposition of a normal matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.decompositions.unitary_eig`, `cirq.unitary_eig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.unitary_eig(
    matrix: np.ndarray,
    check_preconditions: bool = True,
    atol: float = 1e-08
) -> Tuple[np.array, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

All hermitian and unitary matrices are normal matrices. This method was
introduced as for certain classes of unitary matrices (where the eigenvalues
are close to each other) the eigenvectors returned by `numpy.linalg.eig` are
not guaranteed to be orthogonal.
For more information, see https://github.com/numpy/numpy/issues/15461.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matrix`
</td>
<td>
a normal matrix. If not normal, this method is not
guaranteed to return correct eigenvalues.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
when true and matrix is not unitary,
a `ValueError` is raised
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
the absolute tolerance when checking whether the original matrix
was unitary
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`eigvals`
</td>
<td>
the eigenvalues of `matrix`
</td>
</tr><tr>
<td>
`V`
</td>
<td>
the unitary matrix with the eigenvectors as columns
</td>
</tr>
</table>

