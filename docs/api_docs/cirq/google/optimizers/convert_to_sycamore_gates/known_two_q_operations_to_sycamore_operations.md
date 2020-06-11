<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.known_two_q_operations_to_sycamore_operations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.known_two_q_operations_to_sycamore_operations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Synthesize a known gate operation to a sycamore operation

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.known_two_q_operations_to_sycamore_operations(
    qubit_a: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    qubit_b: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    op: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>,
    tabulation: Optional[<a href="../../../../cirq/google/GateTabulation.md"><code>cirq.google.GateTabulation</code></a>] = None
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This function dispatches based on gate type

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubit_a`
</td>
<td>
first qubit of GateOperation
</td>
</tr><tr>
<td>
`qubit_b`
</td>
<td>
second qubit of GateOperation
</td>
</tr><tr>
<td>
`op`
</td>
<td>
operation to decompose
</td>
</tr><tr>
<td>
`tabulation`
</td>
<td>
A tabulation for the Sycamore gate to use for
decomposing gates.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
New operations iterable object
</td>
</tr>

</table>

