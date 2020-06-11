<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ion.IonDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="at"/>
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_circuit"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="neighbors_of"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_gate"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.ion.IonDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A device with qubits placed on a line.

Inherits From: [`Device`](../../cirq/devices/Device.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.IonDevice`, `cirq.ion.ion_device.IonDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ion.IonDevice(
    measurement_duration: "cirq.DURATION_LIKE",
    twoq_gates_duration: "cirq.DURATION_LIKE",
    oneq_gates_duration: "cirq.DURATION_LIKE",
    qubits: Iterable[<a href="../../cirq/devices/LineQubit.md"><code>cirq.devices.LineQubit</code></a>]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Qubits have all-to-all connectivity.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`measurement_duration`
</td>
<td>
The maximum duration of a measurement.
</td>
</tr><tr>
<td>
`twoq_gates_duration`
</td>
<td>
The maximum duration of a two qubit operation.
</td>
</tr><tr>
<td>
`oneq_gates_duration`
</td>
<td>
The maximum duration of a single qubit
operation.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits on the device, identified by their x, y location.
</td>
</tr>
</table>



## Methods

<h3 id="at"><code>at</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>at(
    position: int
) -> Optional[<a href="../../cirq/devices/LineQubit.md"><code>cirq.devices.LineQubit</code></a>]
</code></pre>

Returns the qubit at the given position, if there is one, else None.
        

<h3 id="can_add_operation_into_moment"><code>can_add_operation_into_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>can_add_operation_into_moment(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>,
    moment: <a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
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



<h3 id="decompose_circuit"><code>decompose_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
) -> <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
</code></pre>




<h3 id="decompose_operation"><code>decompose_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_operation(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>

Returns a device-valid decomposition for the given operation.

This method is used when adding operations into circuits with a device
specified, to avoid spurious failures due to e.g. using a Hadamard gate
that must be decomposed into native gates.

<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation
)
</code></pre>




<h3 id="neighbors_of"><code>neighbors_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors_of(
    qubit: <a href="../../cirq/devices/LineQubit.md"><code>cirq.devices.LineQubit</code></a>
) -> Iterable[<a href="../../cirq/devices/LineQubit.md"><code>cirq.devices.LineQubit</code></a>]
</code></pre>

Returns the qubits that the given qubit can interact with.


<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_set() -> FrozenSet['cirq.LineQubit']
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
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



<h3 id="validate_gate"><code>validate_gate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_gate(
    gate: <a href="../../cirq/ops/Gate.md"><code>cirq.ops.Gate</code></a>
)
</code></pre>




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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/ion_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation
)
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






