<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.matrix_commutes" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.matrix_commutes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/predicates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines if two matrices approximately commute.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.predicates.matrix_commutes`, `cirq.matrix_commutes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.matrix_commutes(
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

Two matrices A and B commute if they are square and have the same size and
AB = BA.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`m1`
</td>
<td>
One of the matrices.
</td>
</tr><tr>
<td>
`m2`
</td>
<td>
The other matrix.
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
Whether the two matrices have compatible sizes and a commutator equal
to zero within tolerance.
</td>
</tr>

</table>

