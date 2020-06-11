<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.inverse" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.inverse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/inverse_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the inverse `val**-1` of the given value, if defined.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.inverse`, `cirq.protocols.inverse_protocol.inverse`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.inverse(
    val: Any,
    default: Any = cirq.protocols.inverse_protocol.RaiseTypeErrorIfNotProvided
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

An object can define an inverse by defining a __pow__(self, exponent) method
that returns something besides NotImplemented when given the exponent -1.
The inverse of iterables is by default defined to be the iterable's items,
each inverted, in reverse order.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value (or iterable of invertible values) to invert.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
an inverse defined. If `default` is not set, a TypeError is raised.
If `default` is set to a value, that value is returned.
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
NotImplemented when given an exponent of -1, that result is returned.
Otherwise, if `val` is iterable, the result is a tuple with the same
items as `val` but in reverse order and with each item inverted.
Otherwise, if a `default` argument was specified, it is returned.
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
`val` doesn't have a __pow__ method, or that method returned
NotImplemented when given -1. Furthermore `val` isn't an
iterable containing invertible items. Also, no `default` argument
was specified.
</td>
</tr>
</table>

