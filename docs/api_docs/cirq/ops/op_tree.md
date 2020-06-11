<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.op_tree" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.ops.op_tree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/op_tree.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A recursive type describing trees of operations, and utility methods for it.



## Classes

[`class OpTree`](../../cirq/ops/op_tree/OpTree.md): The recursive type consumed by circuit builder methods.

## Functions

[`flatten_op_tree(...)`](../../cirq/ops/flatten_op_tree.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops(...)`](../../cirq/ops/flatten_to_ops.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops_or_moments(...)`](../../cirq/ops/flatten_to_ops_or_moments.md): Performs an in-order iteration OP_TREE, yielding ops and moments.

[`freeze_op_tree(...)`](../../cirq/ops/freeze_op_tree.md): Replaces all iterables in the OP_TREE with tuples.

[`transform_op_tree(...)`](../../cirq/ops/transform_op_tree.md): Maps transformation functions onto the nodes of an OP_TREE.

## Type Aliases

[`OP_TREE`](../../cirq/ops/OP_TREE.md)

