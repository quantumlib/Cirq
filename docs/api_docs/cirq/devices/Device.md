<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.devices.Device" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.devices.Device

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Hardware constraints for validating circuits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Device`, `cirq.devices.device.Device`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->


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

<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_set() -> Optional[AbstractSet['cirq.Qid']]
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/device.py">View source</a>

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





