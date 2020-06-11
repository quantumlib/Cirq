<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.SerializableGateSet" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="deserialize"/>
<meta itemprop="property" content="deserialize_op"/>
<meta itemprop="property" content="is_supported_operation"/>
<meta itemprop="property" content="serialize"/>
<meta itemprop="property" content="serialize_op"/>
<meta itemprop="property" content="supported_gate_types"/>
<meta itemprop="property" content="with_added_gates"/>
</div>

# cirq.google.SerializableGateSet

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class for serializing and deserializing programs and operations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.serializable_gate_set.SerializableGateSet`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.SerializableGateSet(
    gate_set_name: str,
    serializers: Iterable[<a href="../../cirq/google/GateOpSerializer.md"><code>cirq.google.GateOpSerializer</code></a>],
    deserializers: Iterable[<a href="../../cirq/google/GateOpDeserializer.md"><code>cirq.google.GateOpDeserializer</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class is for cirq.google.api.v2. protos.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`gate_set_name`
</td>
<td>
The name used to identify the gate set.
</td>
</tr><tr>
<td>
`serializers`
</td>
<td>
The GateOpSerializers to use for serialization.
Multiple serializers for a given gate type are allowed and
will be checked for a given type in the order specified here.
This allows for a given gate type to be serialized into
different serialized form depending on the parameters of the
gate.
</td>
</tr><tr>
<td>
`deserializers`
</td>
<td>
The GateOpDeserializers to convert serialized
forms of gates to GateOperations.
</td>
</tr>
</table>



## Methods

<h3 id="deserialize"><code>deserialize</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>deserialize(
    proto: <a href="../../cirq/google/api/v2/program_pb2/Program.md"><code>cirq.google.api.v2.program_pb2.Program</code></a>,
    device: Optional['cirq.Device'] = None
) -> "cirq.Circuit"
</code></pre>

Deserialize a Circuit from a cirq.google.api.v2.Program.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`proto`
</td>
<td>
A dictionary representing a cirq.google.api.v2.Program proto.
</td>
</tr><tr>
<td>
`device`
</td>
<td>
If the proto is for a schedule, a device is required
Otherwise optional.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The deserialized Circuit, with a device if device was
not None.
</td>
</tr>

</table>



<h3 id="deserialize_op"><code>deserialize_op</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>deserialize_op(
    operation_proto: <a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>,
    *,
    arg_function_language: str = ''
) -> "cirq.Operation"
</code></pre>

Deserialize an Operation from a cirq.google.api.v2.Operation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation_proto`
</td>
<td>
A dictionary representing a
cirq.google.api.v2.Operation proto.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The deserialized Operation.
</td>
</tr>

</table>



<h3 id="is_supported_operation"><code>is_supported_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_supported_operation(
    op: "cirq.Operation"
) -> bool
</code></pre>

Whether or not the given gate can be serialized by this gate set.


<h3 id="serialize"><code>serialize</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize(
    program: "cirq.Circuit",
    msg: Optional[<a href="../../cirq/google/api/v2/program_pb2/Program.md"><code>cirq.google.api.v2.program_pb2.Program</code></a>] = None,
    *,
    arg_function_language: Optional[str] = None
) -> <a href="../../cirq/google/api/v2/program_pb2/Program.md"><code>cirq.google.api.v2.program_pb2.Program</code></a>
</code></pre>

Serialize a Circuit to cirq.google.api.v2.Program proto.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The Circuit to serialize.
</td>
</tr>
</table>



<h3 id="serialize_op"><code>serialize_op</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize_op(
    op: "cirq.Operation",
    msg: Optional[<a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>] = None,
    *,
    arg_function_language: Optional[str] = ''
) -> <a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>
</code></pre>

Serialize an Operation to cirq.google.api.v2.Operation proto.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`op`
</td>
<td>
The operation to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary corresponds to the cirq.google.api.v2.Operation proto.
</td>
</tr>

</table>



<h3 id="supported_gate_types"><code>supported_gate_types</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>supported_gate_types() -> Tuple
</code></pre>




<h3 id="with_added_gates"><code>with_added_gates</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/serializable_gate_set.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_added_gates(
    *,
    gate_set_name: Optional[str] = None,
    serializers: Iterable[<a href="../../cirq/google/GateOpSerializer.md"><code>cirq.google.GateOpSerializer</code></a>] = (),
    deserializers: Iterable[<a href="../../cirq/google/GateOpDeserializer.md"><code>cirq.google.GateOpDeserializer</code></a>] = ()
) -> "SerializableGateSet"
</code></pre>

Creates a new gateset with additional (de)serializers.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`gate_set_name`
</td>
<td>
Optional new name of the gateset. If not given, use
the same name as this gateset.
</td>
</tr><tr>
<td>
`serializers`
</td>
<td>
Serializers to add to those in this gateset.
</td>
</tr><tr>
<td>
`deserializers`
</td>
<td>
Deserializers to add to those in this gateset.
</td>
</tr>
</table>





