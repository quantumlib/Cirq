<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.ParamResolver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="value_of"/>
</div>

# cirq.study.ParamResolver

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Resolves sympy.Symbols to actual values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ParamResolver`, `cirq.study.resolver.ParamResolver`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.ParamResolver(
    param_dict: "cirq.ParamResolverOrSimilarType" = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

A Symbol is a wrapped parameter name (str). A ParamResolver is an object
that can be used to assign values for these keys.

ParamResolvers are hashable.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`param_dict`
</td>
<td>
A dictionary from the ParameterValue key (str) to its
assigned value.
</td>
</tr>
</table>



## Methods

<h3 id="value_of"><code>value_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>value_of(
    value: Union[sympy.Basic, float, str]
) -> "cirq.TParamVal"
</code></pre>

Attempt to resolve a Symbol, string, or float to its assigned value.

Floats are returned without modification.  Strings are resolved via
the parameter dictionary with exact match only.  Otherwise, strings
are considered to be sympy.Symbols with the name as the input string.

sympy.Symbols are first checked for exact match in the parameter
dictionary.  Otherwise, the symbol is resolved using sympy substitution.

Note that passing a formula to this resolver can be slow due to the
underlying sympy library.  For circuits relying on quick performance,
it is recommended that all formulas are flattened before-hand using
cirq.flatten or other means so that formula resolution is avoided.
If unable to resolve a sympy.Symbol, returns it unchanged.
If unable to resolve a name, returns a sympy.Symbol with that name.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
The sympy.Symbol or name or float to try to resolve into just
a float.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value of the parameter as resolved by this resolver.
</td>
</tr>

</table>



<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__() -> bool
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key: Union[sympy.Basic, float, str]
) -> "cirq.TParamVal"
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[Union[str, sympy.Symbol]]
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




