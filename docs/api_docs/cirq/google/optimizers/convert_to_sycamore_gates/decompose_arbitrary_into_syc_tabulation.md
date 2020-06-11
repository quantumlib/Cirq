<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_tabulation" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_tabulation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Synthesize an arbitrary 2 qubit operation to a sycamore operation using

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.decompose_arbitrary_into_syc_tabulation(
    qubit_a: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    qubit_b: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    op: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>,
    tabulation: <a href="../../../../cirq/google/GateTabulation.md"><code>cirq.google.GateTabulation</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
the given Tabulation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubit_a`
</td>
<td>
first qubit of the operation
</td>
</tr><tr>
<td>
`qubit_b`
</td>
<td>
second qubit of the operation
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
A tabulation for the Sycamore gate.
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

