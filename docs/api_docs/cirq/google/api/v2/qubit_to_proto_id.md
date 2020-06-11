<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.qubit_to_proto_id" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.qubit_to_proto_id

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/program.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Return a proto id for a <a href="../../../../cirq/ops/Qid.md"><code>cirq.Qid</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.program.qubit_to_proto_id`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.qubit_to_proto_id(
    q: "cirq.Qid"
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->

For <a href="../../../../cirq/devices/GridQubit.md"><code>cirq.GridQubit</code></a>s this id `{row}_{col}` where `{row}` is the integer
row of the grid qubit, and `{col}` is the integer column of the qubit.

For <a href="../../../../cirq/ops/NamedQubit.md"><code>cirq.NamedQubit</code></a>s this id is the name.

For <a href="../../../../cirq/devices/LineQubit.md"><code>cirq.LineQubit</code></a>s this is string of the `x` attribute.