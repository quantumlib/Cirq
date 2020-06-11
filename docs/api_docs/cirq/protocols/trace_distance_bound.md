<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.trace_distance_bound" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.trace_distance_bound

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/trace_distance_bound.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a maximum on the trace distance between this effect's input

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.trace_distance_bound`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.trace_distance_bound(
    val: Any
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->
and output.

This method attempts a number of strategies to calculate this value.

#### Strategy 1:

Use the effect's `_trace_distance_bound_` method.



#### Strategy 2:

If the effect is unitary, calculate the trace distance bound from the
eigenvalues of the unitary matrix.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The effect of which the bound should be calculated
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If any of the strategies return a result that is not Notimplemented and
not None, that result is returned. Otherwise, 1.0 is returned.
Result is capped at a maximum of 1.0, even if the underlying function
produces a result greater than 1.0
</td>
</tr>

</table>

