<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.flatten" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.study.flatten

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
<p>`cirq.flatten`, `cirq.study.flatten_expressions.flatten`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.flatten(
    val: Any
) -> Tuple[Any, 'ExpressionMap']
</code></pre>



<!-- Placeholder for "Used in" -->
new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
type.

`flatten` goes through every parameter in `val` and does the following:
- If the parameter is a number, don't change it.
- If the parameter is a symbol, don't change it.
- If the parameter is an expression, replace it with a symbol.  The new
    symbol will be `sympy.Symbol('<x + 1>')` if the expression was
    `sympy.Symbol('x') + 1`.  In the unlikely case that an expression with a
    different meaning also has the string `'x + 1'`, a number is appended to
    the name to avoid collision: `sympy.Symbol('<x + 1>_1')`.

This function also creates a dictionary mapping from expressions and symbols
in `val` to the new symbols in the flattened copy of `val`.  E.g
`cirq.ExpressionMap({sympy.Symbol('x')+1: sympy.Symbol('<x + 1>')})`.  This
`ExpressionMap` can be used to transform a sweep over the symbols in `val`
to a sweep over the flattened symbols e.g. a sweep over `sympy.Symbol('x')`
to a sweep over `sympy.Symbol('<x + 1>')`.

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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The tuple (new value, expression map) where new value and expression map
are described above.
</td>
</tr>

</table>



#### Examples:

>>> qubit = cirq.LineQubit(0)
>>> a = sympy.Symbol('a')
>>> circuit = cirq.Circuit(
...     cirq.X(qubit) ** (a/4),
...     cirq.Y(qubit) ** (1-a/2),
... )
>>> print(circuit)

* <b>`0`</b>: ───X^(a/4)───Y^(1 - a/2)───

```
>>> sweep = cirq.Linspace(a, start=0, stop=3, length=4)
>>> print(cirq.ListSweep(sweep))
* <b>`Sweep`</b>: {'a': 0.0}
{'a': 1.0}
{'a': 2.0}
{'a': 3.0}
```

```
>>> c_flat, expr_map = cirq.flatten(circuit)
>>> print(c_flat)
* <b>`0`</b>: ───X^(<a/4>)───Y^(<1 - a/2>)───
>>> expr_map
cirq.ExpressionMap({a/4: <a/4>, 1 - a/2: <1 - a/2>})
```

```
>>> new_sweep = expr_map.transform_sweep(sweep)
>>> print(new_sweep)
* <b>`Sweep`</b>: {'<a/4>': 0.0, '<1 - a/2>': 1.0}
{'<a/4>': 0.25, '<1 - a/2>': 0.5}
{'<a/4>': 0.5, '<1 - a/2>': 0.0}
{'<a/4>': 0.75, '<1 - a/2>': -0.5}
```

```
>>> for params in sweep:  # Original
...     print(circuit,
...           '=>',
...           cirq.resolve_parameters(circuit, params))
* <b>`0`</b>: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0───Y───
* <b>`0`</b>: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.25───Y^0.5───
* <b>`0`</b>: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.5───Y^0───
* <b>`0`</b>: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.75───Y^-0.5───
```

```
>>> for params in new_sweep:  # Flattened
...     print(c_flat, '=>', end=' ')
...     print(cirq.resolve_parameters(c_flat, params))
* <b>`0`</b>: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0───Y───
* <b>`0`</b>: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.25───Y^0.5───
* <b>`0`</b>: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.5───Y^0───
* <b>`0`</b>: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.75───Y^-0.5───
```
