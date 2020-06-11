<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.pow" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.pow

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/pow_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns `val**factor` of the given value, if defined.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.pow`, `cirq.protocols.pow_protocol.pow`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.pow(
    val: Any,
    exponent: Any,
    default: Any = cirq.protocols.pow_protocol.RaiseTypeErrorIfNotProvided
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

Values define an extrapolation by defining a __pow__(self, exponent) method.
Note that the method may return NotImplemented to indicate a particular
extrapolation can't be done.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value or iterable of values to invert.
</td>
</tr><tr>
<td>
`exponent`
</td>
<td>
The extrapolation factor. For example, if this is 0.5 and val
is a gate then the caller is asking for a square root of the gate.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
an extrapolation defined. If `default` is not set and that occurs,
a TypeError is raised instead.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a __pow__ method that returns something besides
NotImplemented, that result is returned. Otherwise, if a default value
was specified, the default value is returned.
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
`val` doesn't have a __pow__ method (or that method returned
NotImplemented) and no `default` value was specified.
</td>
</tr>
</table>

