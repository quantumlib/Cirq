<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.GateOpSerializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="can_serialize_operation"/>
<meta itemprop="property" content="to_proto"/>
</div>

# cirq.google.GateOpSerializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_serializer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Describes how to serialize a GateOperation for a given Gate type.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.op_serializer.GateOpSerializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.GateOpSerializer(
    *,
    gate_type: Type[<a href="../../cirq/google/op_serializer/Gate.md"><code>cirq.google.op_serializer.Gate</code></a>] = (lambda x: True),
    serialized_gate_id: str,
    args: List[<a href="../../cirq/google/SerializingArg.md"><code>cirq.google.SerializingArg</code></a>],
    can_serialize_predicate: Callable[['cirq.Operation'], bool] = <function GateOpSerializer.<lambda> at 0x7f038e04d8c0>
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`gate_type`
</td>
<td>
The type of the gate that is being serialized.
</td>
</tr><tr>
<td>
`serialized_gate_id`
</td>
<td>
The string id of the gate when serialized.
</td>
</tr><tr>
<td>
`can_serialize_predicate`
</td>
<td>
Sometimes an Operation can only be
serialized for particular parameters. This predicate will be
checked before attempting to serialize the Operation. If the
predicate is False, serialization will result in a None value.
Default value is a lambda that always returns True.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A list of specification of the arguments to the gate when
serializing, including how to get this information from the
gate of the given gate type.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`gate_type`
</td>
<td>
The type of the gate that can be serialized.
</td>
</tr><tr>
<td>
`serialized_gate_id`
</td>
<td>
The id used when serializing the gate.
</td>
</tr>
</table>



## Methods

<h3 id="can_serialize_operation"><code>can_serialize_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_serializer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>can_serialize_operation(
    op: "cirq.Operation"
) -> bool
</code></pre>

Whether the given operation can be serialized by this serializer.

This checks that the gate is a subclass of the gate type for this
serializer, and that the gate returns true for
`can_serializer_predicate` called on the gate.

<h3 id="to_proto"><code>to_proto</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_serializer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_proto(
    op: "cirq.Operation",
    msg: Optional[<a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>] = None,
    *,
    arg_function_language: Optional[str] = ''
) -> Optional[<a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>]
</code></pre>

Returns the cirq.google.api.v2.Operation message as a proto dict.




