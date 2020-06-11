<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.dot" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.dot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/combinators.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the dot/matrix product of a sequence of values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.dot`, `cirq.linalg.combinators.dot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.dot(
    *values
) -> Union[float, complex, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

Performs the computation in serial order without regard to the matrix
sizes.  If you are using this for matrices of large and differing sizes,
consider using np.lingalg.multi_dot for better performance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*values`
</td>
<td>
The values to combine with the dot/matrix product.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The resulting value or matrix.
</td>
</tr>

</table>

