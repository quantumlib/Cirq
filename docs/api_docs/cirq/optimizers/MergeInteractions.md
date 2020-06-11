<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.MergeInteractions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimization_at"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.optimizers.MergeInteractions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/merge_interactions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Combines series of adjacent one and two-qubit gates operating on a pair

Inherits From: [`PointOptimizer`](../../cirq/circuits/PointOptimizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.MergeInteractions`, `cirq.optimizers.merge_interactions.MergeInteractions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.MergeInteractions(
    tolerance: float = 1e-08,
    allow_partial_czs: bool = True,
    post_clean_up: Callable[[Sequence[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]], <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>] = (lambda op_list: op_list)
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
of qubits.

## Methods

<h3 id="optimization_at"><code>optimization_at</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/merge_interactions.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimization_at(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    index: int,
    op: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> Optional[<a href="../../cirq/circuits/PointOptimizationSummary.md"><code>cirq.circuits.PointOptimizationSummary</code></a>]
</code></pre>

Describes how to change operations near the given location.

For example, this method could realize that the given operation is an
X gate and that in the very next moment there is a Z gate. It would
indicate that they should be combined into a Y gate by returning
PointOptimizationSummary(clear_span=2,
                         clear_qubits=op.qubits,
                         new_operations=cirq.Y(op.qubits[0]))

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to improve.
</td>
</tr><tr>
<td>
`index`
</td>
<td>
The index of the moment with the operation to focus on.
</td>
</tr><tr>
<td>
`op`
</td>
<td>
The operation to focus improvements upon.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A description of the optimization to perform, or else None if no
change should be made.
</td>
</tr>

</table>



<h3 id="optimize_circuit"><code>optimize_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/optimization_pass.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimize_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/optimization_pass.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>

Call self as a function.




