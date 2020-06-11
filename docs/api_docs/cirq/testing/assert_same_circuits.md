<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_same_circuits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_same_circuits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/circuit_compare.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Asserts that two circuits are identical, with a descriptive error.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.circuit_compare.assert_same_circuits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_same_circuits(
    actual: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    expected: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`actual`
</td>
<td>
A circuit computed by some code under test.
</td>
</tr><tr>
<td>
`expected`
</td>
<td>
The circuit that should have been computed.
</td>
</tr>
</table>

