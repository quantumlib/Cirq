<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.ConvertToCzAndSingleGates" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimization_at"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.optimizers.ConvertToCzAndSingleGates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/convert_to_cz_and_single_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Attempts to convert strange multi-qubit gates into CZ and single qubit

Inherits From: [`PointOptimizer`](../../cirq/circuits/PointOptimizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ConvertToCzAndSingleGates`, `cirq.optimizers.convert_to_cz_and_single_gates.ConvertToCzAndSingleGates`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.ConvertToCzAndSingleGates(
    ignore_failures: bool = False,
    allow_partial_czs: bool = False
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
gates.

First, checks if the operation has a unitary effect. If so, and the gate is
    a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
    operation.

Second, attempts to <a href="../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a> to the operation.

Third, if ignore_failures is set, gives up and returns the gate unchanged.
    Otherwise raises a TypeError.

## Methods

<h3 id="optimization_at"><code>optimization_at</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/convert_to_cz_and_single_gates.py">View source</a>

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




