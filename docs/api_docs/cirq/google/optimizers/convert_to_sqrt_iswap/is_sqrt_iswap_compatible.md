<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap_compatible" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap_compatible

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sqrt_iswap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check if the given operation is compatible with the sqrt_iswap gateset

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap_compatible(
    op: "cirq.Operation"
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->
gate set.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`op`
</td>
<td>
Input operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the operation is native to the gate set, false otherwise.
</td>
</tr>

</table>

