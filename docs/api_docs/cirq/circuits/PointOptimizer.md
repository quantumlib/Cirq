<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.PointOptimizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimization_at"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.circuits.PointOptimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/optimization_pass.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Makes circuit improvements focused on a specific location.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.PointOptimizer`, `cirq.circuits.optimization_pass.PointOptimizer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.PointOptimizer(
    post_clean_up: Callable[[Sequence['cirq.Operation']], <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>] = (lambda op_list: op_list)
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="optimization_at"><code>optimization_at</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/optimization_pass.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>optimization_at(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    index: int,
    op: "cirq.Operation"
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




