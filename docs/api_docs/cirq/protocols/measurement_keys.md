<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.measurement_keys" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.measurement_keys

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/measurement_key_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the measurement keys of measurements within the given value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.measurement_keys`, `cirq.protocols.measurement_key_protocol.measurement_keys`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.measurement_keys(
    val: Any,
    *,
    allow_decompose: bool = True
) -> Tuple[str, ...]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value which has the measurement key.
</td>
</tr><tr>
<td>
`allow_decompose`
</td>
<td>
Defaults to True. When true, composite operations that
don't directly specify their measurement keys will be decomposed in
order to find measurement keys within the decomposed operations. If
not set, composite operations will appear to have no measurement
keys. Used by internal methods to stop redundant decompositions from
being performed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The measurement keys of the value. If the value has no measurement,
the result is the empty tuple.
</td>
</tr>

</table>

