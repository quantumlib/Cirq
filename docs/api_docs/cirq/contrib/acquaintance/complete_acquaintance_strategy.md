<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.complete_acquaintance_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.complete_acquaintance_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/strategies/complete.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an acquaintance strategy capable of executing a gate corresponding

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.strategies.complete.complete_acquaintance_strategy`, `cirq.contrib.acquaintance.strategies.complete_acquaintance_strategy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.complete_acquaintance_strategy(
    qubit_order: Sequence['cirq.Qid'],
    acquaintance_size: int = 0,
    swap_gate: "cirq.Gate" = cirq.ops.SWAP
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->
to any set of at most acquaintance_size qubits.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubit_order`
</td>
<td>
The qubits on which the strategy should be defined.
</td>
</tr><tr>
<td>
`acquaintance_size`
</td>
<td>
The maximum number of qubits to be acted on by
an operation.
</td>
</tr><tr>
<td>
`swap_gate`
</td>
<td>
The gate used to swap logical indices.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A circuit capable of implementing any set of k-local
operations.
</td>
</tr>

</table>

