<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.pasqal.PasqalDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="distance"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="is_pasqal_device_op"/>
<meta itemprop="property" content="qubit_list"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.pasqal.PasqalDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A Pasqal Device with qubits placed on a 3D grid.

Inherits From: [`Device`](../../cirq/devices/Device.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.pasqal.pasqal_device.PasqalDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.pasqal.PasqalDevice(
    control_radius: float,
    qubits: Iterable[<a href="../../cirq/pasqal/ThreeDGridQubit.md"><code>cirq.pasqal.ThreeDGridQubit</code></a>]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`control_radius`
</td>
<td>
the maximum distance between qubits for a controlled
gate. Distance is measured in units of the indices passed into
the qubit constructor.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits on the device, identified by their x, y, z position.
Must be of type ThreeDGridQubit.
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
if the wrong qubit type is provided or if invalid
parameter is provided for control_radius.
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_operation(
    operation: <a href="../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> "cirq.OP_TREE"
</code></pre>

Returns a device-valid decomposition for the given operation.

This method is used when adding operations into circuits with a device
specified, to avoid spurious failures due to e.g. using a Hadamard gate
that must be decomposed into native gates.

<h3 id="distance"><code>distance</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>distance(
    p: "cirq.Qid",
    q: "cirq.Qid"
) -> float
</code></pre>

Returns the distance between two qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`p`
</td>
<td>
qubit involved in the distance computation
</td>
</tr><tr>
<td>
`q`
</td>
<td>
qubit involved in the distance computation
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The distance between qubits p and q, in lattice spacing units.
</td>
</tr>

</table>



<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation: <a href="../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
)
</code></pre>

Provides the duration of the given operation on this device.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation`
</td>
<td>
the operation to get the duration of
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The duration of the given operation on this device
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
If the operation provided doesn't correspond to a native
gate
</td>
</tr>
</table>



<h3 id="is_pasqal_device_op"><code>is_pasqal_device_op</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>is_pasqal_device_op(
    op: <a href="../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> bool
</code></pre>




<h3 id="qubit_list"><code>qubit_list</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_list()
</code></pre>




<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_set() -> FrozenSet[<a href="../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
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



<h3 id="validate_moment"><code>validate_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_moment(
    moment: "cirq.Moment"
) -> None
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/pasqal/pasqal_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation: <a href="../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
)
</code></pre>

Raises an error if the given operation is invalid on this device.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation`
</td>
<td>
the operation to validate
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
If the operation is not valid
</td>
</tr>
</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






