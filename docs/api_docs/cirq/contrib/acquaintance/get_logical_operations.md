<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.get_logical_operations" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.get_logical_operations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the logical operations specified by the physical operations and

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.permutation.get_logical_operations`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.get_logical_operations(
    operations: "cirq.OP_TREE",
    initial_mapping: Dict[<a href="../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, <a href="../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
) -> Iterable['cirq.Operation']
</code></pre>



<!-- Placeholder for "Used in" -->
initial mapping.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`operations`
</td>
<td>
The physical operations.
</td>
</tr><tr>
<td>
`initial_mapping`
</td>
<td>
The initial mapping of physical to logical qubits.
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
A non-permutation physical operation acts on an unmapped
qubit.
</td>
</tr>
</table>

