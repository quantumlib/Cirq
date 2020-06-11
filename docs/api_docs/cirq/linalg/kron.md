<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kron" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kron

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/combinators.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the kronecker product of a sequence of values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kron`, `cirq.linalg.combinators.kron`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kron(
    *factors,
    shape_len: int = 2
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

A *args version of lambda args: functools.reduce(np.kron, args).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*factors`
</td>
<td>
The matrices, tensors, and/or scalars to combine together
using np.kron.
</td>
</tr><tr>
<td>
`shape_len`
</td>
<td>
The expected number of dimensions in the output. Mainly
determines the behavior of the empty kron product.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The kronecker product of all the inputs.
</td>
</tr>

</table>

