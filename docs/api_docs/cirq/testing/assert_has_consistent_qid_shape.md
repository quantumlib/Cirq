<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_has_consistent_qid_shape" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_has_consistent_qid_shape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/circuit_compare.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tests whether a value's `_qid_shape_` and `_num_qubits_` are correct and

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.circuit_compare.assert_has_consistent_qid_shape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_has_consistent_qid_shape(
    val: Any
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
consistent.

Verifies that the entries in the shape are all positive integers and the
length of shape equals `_num_qubits_` (and also equals `len(qubits)` if
`val` has `qubits`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value under test. Should have `_qid_shape_` and/or
`num_qubits_` methods. Can optionally have a `qubits` property.
</td>
</tr>
</table>

