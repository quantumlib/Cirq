<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.gates.acquaint_and_shift" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.gates.acquaint_and_shift

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Acquaints and shifts a pair of lists of qubits. The first part is

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.gates.acquaint_and_shift(
    parts: Tuple[List['cirq.Qid'], List['cirq.Qid']],
    layers: <a href="../../../../cirq/contrib/acquaintance/gates/Layers.md"><code>cirq.contrib.acquaintance.gates.Layers</code></a>,
    acquaintance_size: Optional[int],
    swap_gate: "cirq.Gate",
    mapping: Dict[<a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int]
)
</code></pre>



<!-- Placeholder for "Used in" -->
acquainted with every qubit individually in the second part, and vice
versa. Operations are grouped into several layers:
    * prior_interstitial: The first layer of acquaintance gates.
    * prior: The combination of acquaintance gates and swaps that acquaints
        the inner halves.
    * intra: The shift gate.
    * post: The combination of acquaintance gates and swaps that acquaints
        the outer halves.
    * posterior_interstitial: The last layer of acquaintance gates.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`parts`
</td>
<td>
The two lists of qubits to acquaint.
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
`acquaintance_size`
</td>
<td>
The number of qubits to acquaint at a time. If None,
after each pair of parts is shifted the union thereof is
acquainted.
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

