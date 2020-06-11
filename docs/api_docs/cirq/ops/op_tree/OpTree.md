<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.op_tree.OpTree" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.ops.op_tree.OpTree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/op_tree.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The recursive type consumed by circuit builder methods.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.op_tree.OpTree(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

An OpTree is a type protocol, satisfied by anything that can be recursively
flattened into Operations. We also define the Union type OP_TREE which
can be an OpTree or just a single Operation.

#### For example:


- An Operation is an OP_TREE all by itself.
- A list of operations is an OP_TREE.
- A list of tuples of operations is an OP_TREE.
- A list with a mix of operations and lists of operations is an OP_TREE.
- A generator yielding operations is an OP_TREE.

Note: once mypy supports recursive types this could be defined as an alias:

OP_TREE = Union[Operation, Iterable['OP_TREE']]

See: https://github.com/python/mypy/issues/731

## Methods

<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/op_tree.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[Union[<a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>, 'OpTree']]
</code></pre>






