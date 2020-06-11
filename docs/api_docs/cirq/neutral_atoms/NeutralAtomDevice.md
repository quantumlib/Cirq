<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.neutral_atoms.NeutralAtomDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="can_add_operation_into_moment"/>
<meta itemprop="property" content="decompose_operation"/>
<meta itemprop="property" content="distance"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="neighbors_of"/>
<meta itemprop="property" content="qubit_list"/>
<meta itemprop="property" content="qubit_set"/>
<meta itemprop="property" content="validate_circuit"/>
<meta itemprop="property" content="validate_gate"/>
<meta itemprop="property" content="validate_moment"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.neutral_atoms.NeutralAtomDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A device with qubits placed on a grid.

Inherits From: [`Device`](../../cirq/devices/Device.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.NeutralAtomDevice`, `cirq.neutral_atoms.neutral_atom_devices.NeutralAtomDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.neutral_atoms.NeutralAtomDevice(
    measurement_duration: "cirq.DURATION_LIKE",
    gate_duration: "cirq.DURATION_LIKE",
    control_radius: float,
    max_parallel_z: int,
    max_parallel_xy: int,
    max_parallel_c: int,
    qubits: Iterable[<a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`measurement_duration`
</td>
<td>
the maximum duration of a measurement.
</td>
</tr><tr>
<td>
`gate_duration`
</td>
<td>
the maximum duration of a gate
</td>
</tr><tr>
<td>
`control_radius`
</td>
<td>
the maximum distance between qubits for a controlled
gate. Distance is measured in units of the indices passed into
the GridQubit constructor.
</td>
</tr><tr>
<td>
`max_parallel_z`
</td>
<td>
The maximum number of qubits that can be acted on
in parallel by a Z gate
</td>
</tr><tr>
<td>
`max_parallel_xy`
</td>
<td>
The maximum number of qubits that can be acted on
in parallel by a local XY gate
</td>
</tr><tr>
<td>
`max_parallel_c`
</td>
<td>
the maximum number of qubits that can be acted on in
parallel by a controlled gate. Must be less than or equal to the
lesser of max_parallel_z and max_parallel_xy
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits on the device, identified by their x, y location.
Must be of type GridQubit
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
parallel parameters are provided
</td>
</tr>
</table>



## Methods

<h3 id="can_add_operation_into_moment"><code>can_add_operation_into_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>can_add_operation_into_moment(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>,
    moment: <a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
) -> bool
</code></pre>

Determines if it's possible to add an operation into a moment. An
operation can be added if the moment with the operation added is valid

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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If either of the given moment or operation is invalid
</td>
</tr>
</table>



<h3 id="decompose_operation"><code>decompose_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_operation(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>

Returns a device-valid decomposition for the given operation.

This method is used when adding operations into circuits with a device
specified, to avoid spurious failures due to e.g. using a Hadamard gate
that must be decomposed into native gates.

<h3 id="distance"><code>distance</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>distance(
    p: "cirq.Qid",
    q: "cirq.Qid"
) -> float
</code></pre>




<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
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



<h3 id="neighbors_of"><code>neighbors_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors_of(
    qubit: "cirq.GridQubit"
) -> Iterable['cirq.GridQubit']
</code></pre>

Returns the qubits that the given qubit can interact with.


<h3 id="qubit_list"><code>qubit_list</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_list()
</code></pre>




<h3 id="qubit_set"><code>qubit_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_set() -> FrozenSet['cirq.GridQubit']
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>

Raises an error if the given circuit is invalid on this device. A
circuit is invalid if any of its moments are invalid or if there is a
non-empty moment after a moment with a measurement.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to validate
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
If the given circuit can't be run on this device
</td>
</tr>
</table>



<h3 id="validate_gate"><code>validate_gate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_gate(
    gate: <a href="../../cirq/ops/Gate.md"><code>cirq.ops.Gate</code></a>
)
</code></pre>

Raises an error if the provided gate isn't part of the native gate set.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`gate`
</td>
<td>
the gate to validate
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
If the given gate is not part of the native gate set.
</td>
</tr>
</table>



<h3 id="validate_moment"><code>validate_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_moment(
    moment: <a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
)
</code></pre>

Raises an error if the given moment is invalid on this device


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moment`
</td>
<td>
The moment to validate
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
If the given moment is invalid
</td>
</tr>
</table>



<h3 id="validate_operation"><code>validate_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/neutral_atoms/neutral_atom_devices.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
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






