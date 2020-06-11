<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kron_with_controls" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kron_with_controls

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/combinators.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the kronecker product of a sequence of values and control tags.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kron_with_controls`, `cirq.linalg.combinators.kron_with_controls`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kron_with_controls(
    *factors
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Use <a href="../../cirq.md#CONTROL_TAG"><code>cirq.CONTROL_TAG</code></a> to represent controls. Any entry of the output
corresponding to a situation where the control is not satisfied will
be overwritten by identity matrix elements.

The control logic works by imbuing NaN with the meaning "failed to meet one
or more controls". The normal kronecker product then spreads the per-item
NaNs to all the entries in the product that need to be replaced by identity
matrix elements. This method rewrites those NaNs. Thus CONTROL_TAG can be
the matrix [[NaN, 0], [0, 1]] or equivalently [[NaN, NaN], [NaN, 1]].

Because this method re-interprets NaNs as control-failed elements, it won't
propagate error-indicating NaNs from its input to its output in the way
you'd otherwise expect.

#### Examples:


```
result = cirq.kron_with_controls(
    cirq.CONTROL_TAG,
    cirq.unitary(cirq.X))
print(result.astype(np.int32))

# prints:
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 0 1]
#  [0 0 1 0]]
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*factors`
</td>
<td>
The matrices, tensors, scalars, and/or control tags to combine
together using np.kron.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The resulting matrix.
</td>
</tr>

</table>

