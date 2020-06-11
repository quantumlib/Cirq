<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_has_consistent_apply_unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_has_consistent_apply_unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/circuit_compare.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tests whether a value's _apply_unitary_ is correct.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.circuit_compare.assert_has_consistent_apply_unitary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_has_consistent_apply_unitary(
    val: Any,
    *,
    atol: float = 1e-08
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Contrasts the effects of the value's `_apply_unitary_` with the
matrix returned by the value's `_unitary_` method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value under test. Should have a `__pow__` method.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute error tolerance.
</td>
</tr>
</table>

