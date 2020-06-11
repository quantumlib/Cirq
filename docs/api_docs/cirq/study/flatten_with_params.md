<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.flatten_with_params" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.study.flatten_with_params

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/flatten_expressions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates a copy of `val` with any symbols or expressions replaced with

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.flatten_with_params`, `cirq.study.flatten_expressions.flatten_with_params`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.flatten_with_params(
    val: Any,
    params: <a href="../../cirq/study/ParamResolverOrSimilarType.md"><code>cirq.study.ParamResolverOrSimilarType</code></a>
) -> Tuple[Any, <a href="../../cirq/study/ParamDictType.md"><code>cirq.study.ParamDictType</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->
new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
type.  Also transforms a dictionary of symbol values for `val` to an
equivalent dictionary mapping the new symbols to their evaluated values.

`flatten_with_params` goes through every parameter in `val` and does the
following:
- If the parameter is a number, don't change it.
- If the parameter is a symbol, don't change it and use the same symbol with
    the same value in the new dictionary of symbol values.
- If the parameter is an expression, replace it with a symbol and use the
    new symbol with the evaluated value of the expression in the new
    dictionary of symbol values.  The new symbol will be
    `sympy.Symbol('<x + 1>')` if the expression was `sympy.Symbol('x') + 1`.
    In the unlikely case that an expression with a different meaning also
    has the string `'x + 1'`, a number is appended to the name to avoid
    collision: `sympy.Symbol('<x + 1>_1')`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to copy and substitute parameter expressions with
flattened symbols.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
A dictionary or `ParamResolver` where the keys are
`sympy.Symbol`s used by `val` and the values are numbers.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The tuple (new value, new params) where new value is `val` with
flattened expressions and new params is a dictionary mapping the
new symbols like `sympy.Symbol('<x + 1>')` to numbers like
`params['x'] + 1`.
</td>
</tr>

</table>

