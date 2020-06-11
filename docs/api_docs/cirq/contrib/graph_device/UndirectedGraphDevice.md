<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.graph_device.UndirectedGraphDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="get_device_edge_from_op"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_crosstalk"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.contrib.graph_device.UndirectedGraphDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A device whose properties are represented by an edge-labelled graph.

Inherits From: [`Device`](../../../cirq/devices/Device.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.graph_device.graph_device.UndirectedGraphDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.graph_device.UndirectedGraphDevice(
    device_graph: Optional[<a href="../../../cirq/contrib/graph_device/UndirectedHypergraph.md"><code>cirq.contrib.graph_device.UndirectedHypergraph</code></a>] = None,
    crosstalk_graph: Optional[<a href="../../../cirq/contrib/graph_device/UndirectedHypergraph.md"><code>cirq.contrib.graph_device.UndirectedHypergraph</code></a>] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Each (undirected) edge of the device graph is labelled by an
UndirectedGraphDeviceEdge or None. None indicates that any operation is
allowed and has zero duration.

Each (undirected) edge of the constraint graph is labelled either by a
function or None. The function takes as arguments operations on the
adjacent device edges and raises an error if they are not simultaneously
executable. If None, no such operations are allowed.

Note that
    * the crosstalk graph is allowed to have vertices (i.e. device edges)
        that do not exist in the graph device.
    * duration_of does not check that operation is valid.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`edges`
</td>
<td>

</td>
</tr><tr>
<td>
`labelled_edges`
</td>
<td>

</td>
</tr><tr>
<td>
`qubits`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="can_add_operation_into_moment"><code>can_add_operation_into_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>can_add_operation_into_moment(
    operation: "cirq.Operation",
    moment: "cirq.Moment"
) -> bool
</code></pre>

Determines if it's possible to add an operation into a moment.

For example, on the XmonDevice two CZs shouldn't be placed in the same
moment if they are on adjacent qubits.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation`
</td>
<td>
The operation being added.
</td>
</tr><tr>
<td>
`moment`
</td>
<td>
The moment being transformed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Whether or not the moment will validate after adding the operation.
</td>
</tr>

</table>



<h3 id="decompose_operation"><code>decompose_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_operation(
    operation: "cirq.Operation"
) -> "cirq.OP_TREE"
</code></pre>

Returns a device-valid decomposition for the given operation.

This method is used when adding operations into circuits with a device
specified, to avoid spurious failures due to e.g. using a Hadamard gate
that must be decomposed into native gates.

<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../cirq/value/Duration.md"><code>cirq.value.Duration</code></a>
</code></pre>




<h3 id="get_device_edge_from_op"><code>get_device_edge_from_op</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_device_edge_from_op(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../cirq/contrib/graph_device/graph_device/UndirectedGraphDeviceEdge.md"><code>cirq.contrib.graph_device.graph_device.UndirectedGraphDeviceEdge</code></a>
</code></pre>




<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_set() -> FrozenSet['cirq.Qid']
</code></pre>

Returns a set or frozenset of qubits on the device, if possible.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
If the device has a finite set of qubits, then a set or frozen set
of all qubits on the device is returned.

If the device has no well defined finite set of qubits (e.g.
`cirq.UnconstrainedDevice` has this property), then `None` is
returned.
</td>
</tr>

</table>



<h3 id="validate_circuit"><code>validate_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_circuit(
    circuit: "cirq.Circuit"
) -> None
</code></pre>

Raises an exception if a circuit is not valid.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to validate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The circuit isn't valid for this device.
</td>
</tr>
</table>



<h3 id="validate_crosstalk"><code>validate_crosstalk</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_crosstalk(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>,
    other_operations: Iterable[<a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
) -> None
</code></pre>




<h3 id="validate_moment"><code>validate_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_moment(
    moment: <a href="../../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
)
</code></pre>

Raises an exception if a moment is not valid.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moment`
</td>
<td>
The moment to validate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The moment isn't valid for this device.
</td>
</tr>
</table>



<h3 id="validate_operation"><code>validate_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> None
</code></pre>

Raises an exception if an operation is not valid.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation`
</td>
<td>
The operation to validate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The operation isn't valid for this device.
</td>
</tr>
</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




