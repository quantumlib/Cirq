<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents

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
<p>`cirq.testing.circuit_compare.assert_has_consistent_apply_unitary_for_various_exponents`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
    val: Any,
    *,
    exponents=(0, 1, -1, 0.5, 0.25, -0.5, 0.1, sympy.Symbol('s'))
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Contrasts the effects of the value's `_apply_unitary_` with the
matrix returned by the value's `_unitary_` method. Attempts this after
attempting to raise the value to several exponents.

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
`exponents`
</td>
<td>
The exponents to try. Defaults to a variety of special and
arbitrary angles, as well as a parameterized angle (a symbol). If
the value's `__pow__` returns `NotImplemented` for any of these,
they are skipped.
</td>
</tr>
</table>

