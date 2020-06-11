<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.update_mapping" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.update_mapping

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Updates a mapping (in place) from qubits to logical indices according to

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.permutation.update_mapping`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.update_mapping(
    mapping: Dict[<a href="../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, <a href="../../../cirq/contrib/acquaintance/executor/LogicalIndex.md"><code>cirq.contrib.acquaintance.executor.LogicalIndex</code></a>],
    operations: "cirq.OP_TREE"
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
a set of permutation gates. Any gates other than permutation gates are
ignored.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mapping`
</td>
<td>
The mapping to update.
</td>
</tr><tr>
<td>
`operations`
</td>
<td>
The operations to update according to.
</td>
</tr>
</table>

