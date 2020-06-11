<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_has_diagram" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_has_diagram

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/circuit_compare.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines if a given circuit has the desired text diagram.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.circuit_compare.assert_has_diagram`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_has_diagram(
    actual: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    desired: str,
    **kwargs
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
The circuit that was actually computed by some process.
</td>
</tr><tr>
<td>
`desired`
</td>
<td>
The desired text diagram as a string. Newlines at the
beginning and whitespace at the end are ignored.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to be passed to actual.to_text_diagram().
</td>
</tr>
</table>

