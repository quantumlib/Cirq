<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.is_parameterized" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.is_parameterized

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/resolve_parameters.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns whether the object is parameterized with any Symbols.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.is_parameterized`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.is_parameterized(
    val: Any
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

A value is parameterized when it has an `_is_parameterized_` method and
that method returns a truthy value, or if the value is an instance of
sympy.Basic.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the gate has any unresolved Symbols
and False otherwise. If no implementation of the magic
method above exists or if that method returns NotImplemented,
this will default to False.
</td>
</tr>

</table>

