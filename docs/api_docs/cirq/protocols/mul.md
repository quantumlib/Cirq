<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.mul" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.mul

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/mul_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns lhs * rhs, or else a default if the operator is not implemented.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.mul`, `cirq.protocols.mul_protocol.mul`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.mul(
    lhs: Any,
    rhs: Any,
    default: Any = cirq.protocols.mul_protocol.RaiseTypeErrorIfNotProvided
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

This method is mostly used by __pow__ methods trying to return
NotImplemented instead of causing a TypeError.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lhs`
</td>
<td>
Left hand side of the multiplication.
</td>
</tr><tr>
<td>
`rhs`
</td>
<td>
Right hand side of the multiplication.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Default value to return if the multiplication is not defined.
If not default is specified, a type error is raised when the
multiplication fails.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The product of the two inputs, or else the default value if the product
is not defined, or else raises a TypeError if no default is defined.
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
lhs doesn't have __mul__ or it returned NotImplemented
AND lhs doesn't have __rmul__ or it returned NotImplemented
AND a default value isn't specified.
</td>
</tr>
</table>

