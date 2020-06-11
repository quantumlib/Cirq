<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.expand_matrix_in_orthogonal_basis" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.expand_matrix_in_orthogonal_basis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/operator_spaces.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes coefficients of expansion of m in basis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.expand_matrix_in_orthogonal_basis`, `cirq.linalg.operator_spaces.expand_matrix_in_orthogonal_basis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.expand_matrix_in_orthogonal_basis(
    m: np.ndarray,
    basis: Dict[str, np.ndarray]
) -> value.LinearDict[str]
</code></pre>



<!-- Placeholder for "Used in" -->

We require that basis be orthogonal w.r.t. the Hilbert-Schmidt inner
product. We do not require that basis be orthonormal. Note that Pauli
basis (I, X, Y, Z) is orthogonal, but not orthonormal.