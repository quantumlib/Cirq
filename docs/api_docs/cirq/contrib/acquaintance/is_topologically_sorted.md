<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.is_topologically_sorted" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.is_topologically_sorted

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/topological_sort.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Whether a given order of operations is consistent with the DAG.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.topological_sort.is_topologically_sorted`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.is_topologically_sorted(
    dag: "cirq.CircuitDag",
    operations: "cirq.OP_TREE",
    equals: Callable[[<a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>, <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>], bool] = operator.eq
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

For example, suppose the (transitive reduction of the) circuit DAG is

     ╭─> Op2 ─╮
Op1 ─┤        ├─> Op4
     ╰─> Op3 ─╯

Then [Op1, Op2, Op3, Op4] and [Op1, Op3, Op2, Op4] (and any operations
tree that flattens to one of them) are topologically sorted according
to the DAG, and any other ordering of the four operations is not.

Evaluates to False when the set of operations is different from those
in the nodes of the DAG, regardless of the ordering.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dag`
</td>
<td>
The circuit DAG.
</td>
</tr><tr>
<td>
`operations`
</td>
<td>
The ordered operations.
</td>
</tr><tr>
<td>
`equals`
</td>
<td>
The function to determine equality of operations. Defaults to
`operator.eq`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Whether or not the operations given are topologically sorted
according to the DAG.
</td>
</tr>

</table>

