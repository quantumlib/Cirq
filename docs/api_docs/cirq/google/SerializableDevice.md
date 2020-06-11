<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.SerializableDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.google.SerializableDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/serializable_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Device object generated from a device specification proto.

Inherits From: [`Device`](../../cirq/devices/Device.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.devices.SerializableDevice`, `cirq.google.devices.serializable_device.SerializableDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.SerializableDevice(
    qubits: List['cirq.Qid'],
    gate_definitions: Dict[Type['cirq.Gate'], List[_GateDefinition]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a device specification proto and a gate_set to translate the
serialized gate_ids to cirq Gates, this will generate a Device that can
verify operations and circuits for the hardware specified by the device.

Expected usage is through constructing this class through a proto using
the static function call from_proto().

This class only supports GridQubits and NamedQubits.  NamedQubits with names
that conflict (such as "4_3") may be converted to GridQubits on
deserialization.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
A list of valid Qid for the device.
</td>
</tr><tr>
<td>
`gate_definitions`
</td>
<td>
Maps cirq gates to device properties for that
gate.
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/serializable_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation: "cirq.Operation"
) -> <a href="../../cirq/value/Duration.md"><code>cirq.value.Duration</code></a>
</code></pre>




<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/serializable_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_proto(
    proto: <a href="../../cirq/google/api/v2/device_pb2/DeviceSpecification.md"><code>cirq.google.api.v2.device_pb2.DeviceSpecification</code></a>,
    gate_sets: Iterable[<a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a>]
) -> "SerializableDevice"
</code></pre>

Args:
    proto: A proto describing the qubits on the device, as well as the
        supported gates and timing information.
    gate_set: A SerializableGateSet that can translate the gate_ids
        into cirq Gates.

<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/serializable_device.py">View source</a>

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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/serializable_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation: "cirq.Operation"
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





