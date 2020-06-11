<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.qubit_from_proto_id" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.qubit_from_proto_id

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/program.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a <a href="../../../../cirq/ops/Qid.md"><code>cirq.Qid</code></a> for a proto id.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.program.qubit_from_proto_id`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.qubit_from_proto_id(
    proto_id: str
) -> "cirq.Qid"
</code></pre>



<!-- Placeholder for "Used in" -->

Proto IDs of the form {int}_{int} are parsed as GridQubits.

Proto IDs of the form {int} are parsed as LineQubits.

All other proto IDs are parsed as NamedQubits. Note that this will happily
accept any string; for circuits which explicitly use Grid or LineQubits,
prefer one of the specialized methods below.

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
A <a href="../../../../cirq/ops/Qid.md"><code>cirq.Qid</code></a> corresponding to the proto id.
</td>
</tr>

</table>

