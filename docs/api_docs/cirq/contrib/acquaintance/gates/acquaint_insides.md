<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.gates.acquaint_insides" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.gates.acquaint_insides

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Acquaints each of the qubits with another set specified by an

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.gates.acquaint_insides(
    swap_gate: "cirq.Gate",
    acquaintance_gate: "cirq.Operation",
    qubits: Sequence['cirq.Qid'],
    before: bool,
    layers: <a href="../../../../cirq/contrib/acquaintance/gates/Layers.md"><code>cirq.contrib.acquaintance.gates.Layers</code></a>,
    mapping: Dict[<a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
acquaintance gate.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The list of qubits of which half are individually acquainted
with another list of qubits.
</td>
</tr><tr>
<td>
`layers`
</td>
<td>
The layers to put gates into.
</td>
</tr><tr>
<td>
`acquaintance_gate`
</td>
<td>
The acquaintance gate that acquaints the end qubit
with another list of qubits.
</td>
</tr><tr>
<td>
`before`
</td>
<td>
Whether the acquainting is done before the shift.
</td>
</tr><tr>
<td>
`swap_gate`
</td>
<td>
The gate used to swap logical indices.
</td>
</tr><tr>
<td>
`mapping`
</td>
<td>
The mapping from qubits to logical indices. Used to keep track
of the effect of inside-acquainting swaps.
</td>
</tr>
</table>

