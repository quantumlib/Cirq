<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.line_qubit_from_proto_id" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.line_qubit_from_proto_id

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/program.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parse a proto id to a <a href="../../../../cirq/devices/LineQubit.md"><code>cirq.LineQubit</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.program.line_qubit_from_proto_id`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.line_qubit_from_proto_id(
    proto_id: str
) -> "cirq.LineQubit"
</code></pre>



<!-- Placeholder for "Used in" -->

Proto ids for line qubits are integer strings representing the `x`
attribute of the line qubit.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`proto_id`
</td>
<td>
The id to convert.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../cirq/devices/LineQubit.md"><code>cirq.LineQubit</code></a> corresponding to the proto id.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the string is not an integer.
</td>
</tr>
</table>

