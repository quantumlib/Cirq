<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.measurement_key" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.measurement_key

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/measurement_key_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get the single measurement key for the given value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.measurement_key`, `cirq.protocols.measurement_key_protocol.measurement_key`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.measurement_key(
    val: Any,
    default: Any = cirq.protocols.measurement_key_protocol.RaiseTypeErrorIfNotProvided
)
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
The value which has one measurement key.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
a measurement key. If `default` is not set, a TypeError is raised.
If default is set to a value, that value is returned if the value
does not have `_measurement_key_`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a `_measurement_key_` method and its result is not
`NotImplemented`, that result is returned. Otherwise, if a default
value was specified, the default value is returned.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
`val` doesn't have a _measurement_key_ method (or that method
returned NotImplemented) and also no default value was specified.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
`val` has multiple measurement keys.
</td>
</tr>
</table>

