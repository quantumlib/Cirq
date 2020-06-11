<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.resolve_parameters" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.resolve_parameters

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/resolve_parameters.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Resolves symbol parameters in the effect using the param resolver.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.resolve_parameters`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.resolve_parameters(
    val: Any,
    param_resolver: "cirq.ParamResolverOrSimilarType"
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

This function will use the `_resolve_parameters_` magic method
of `val` to resolve any Symbols with concrete values from the given
parameter resolver.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The object to resolve (e.g. the gate, operation, etc)
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
the object to use for resolving all symbols
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a gate or operation of the same type, but with all Symbols
replaced with floats according to the given ParamResolver.
If `val` has no `_resolve_parameters_` method or if it returns
NotImplemented, `val` itself is returned.
</td>
</tr>

</table>

