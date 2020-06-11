<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.transform_op_tree" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.transform_op_tree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/op_tree.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Maps transformation functions onto the nodes of an OP_TREE.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ops.op_tree.transform_op_tree`, `cirq.transform_op_tree`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.transform_op_tree(
    root: <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>,
    op_transformation: Callable[[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>], <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>] = (lambda e: e),
    iter_transformation: Callable[[Iterable[<a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>]], <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>] = (lambda e: e),
    preserve_moments: bool = False
) -> <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
The operation or tree of operations to transform.
</td>
</tr><tr>
<td>
`op_transformation`
</td>
<td>
How to transform the operations (i.e. leaves).
</td>
</tr><tr>
<td>
`iter_transformation`
</td>
<td>
How to transform the iterables (i.e. internal
nodes).
</td>
</tr><tr>
<td>
`preserve_moments`
</td>
<td>
Whether to leave Moments alone. If True, the
transformation functions will not be applied to Moments or the
operations within them.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A transformed operation tree.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
root isn't a valid OP_TREE.
</td>
</tr>
</table>

