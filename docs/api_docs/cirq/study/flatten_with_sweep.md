<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.flatten_with_sweep" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.study.flatten_with_sweep

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
<p>`cirq.flatten_with_sweep`, `cirq.study.flatten_expressions.flatten_with_sweep`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.flatten_with_sweep(
    val: Any,
    sweep: Union[<a href="../../cirq/study/Sweep.md"><code>cirq.study.Sweep</code></a>, List[<a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>]]
) -> Tuple[Any, <a href="../../cirq/study/Sweep.md"><code>cirq.study.Sweep</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->
new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
type.  Also transforms a sweep over the symbols in `val` to a sweep over the
new symbols.

`flatten_with_sweep` goes through every parameter in `val` and does the
following:
- If the parameter is a number, don't change it.
- If the parameter is a symbol, don't change it and use the same symbol with
    the same values in the new sweep.
- If the parameter is an expression, replace it with a symbol and use the
    new symbol with the evaluated value of the expression in the new sweep.
    The new symbol will be `sympy.Symbol('<x + 1>')` if the expression was
    `sympy.Symbol('x') + 1`.  In the unlikely case that an expression with a
    different meaning also has the string `'x + 1'`, a number is appended to
    the name to avoid collision: `sympy.Symbol('<x + 1>_1')`.

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
`sweep`
</td>
<td>
A sweep over parameters used by `val`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The tuple (new value, new sweep) where new value is `val` with flattened
expressions and new sweep is the equivalent sweep over it.
</td>
</tr>

</table>

