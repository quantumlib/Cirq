<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.GateOpDeserializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# cirq.google.GateOpDeserializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_deserializer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Describes how to deserialize a proto to a given Gate type.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.op_deserializer.GateOpDeserializer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.GateOpDeserializer(
    serialized_gate_id: str,
    gate_constructor: Callable,
    args: Sequence[<a href="../../cirq/google/DeserializingArg.md"><code>cirq.google.DeserializingArg</code></a>],
    num_qubits_param: Optional[str] = None,
    op_wrapper: Callable[['cirq.Operation', <a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>], 'cirq.Operation'] = (lambda x, y: x)
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`serialized_gate_id`
</td>
<td>
The serialized id of the gate that is being
deserialized.
</td>
</tr><tr>
<td>
`gate_constructor`
</td>
<td>
A function that produces the deserialized gate
given arguments from args.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A list of the arguments to be read from the serialized
gate and the information required to use this to construct
the gate using the gate_constructor above.
</td>
</tr><tr>
<td>
`num_qubits_param`
</td>
<td>
Some gate constructors require that the number
of qubits be passed to their constructor. This is the name
of the parameter in the constructor for this value. If None,
no number of qubits is passed to the constructor.
</td>
</tr><tr>
<td>
`op_wrapper`
</td>
<td>
An optional Callable to modify the resulting
GateOperation, for instance, to add tags
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`serialized_gate_id`
</td>
<td>
The id used when serializing the gate.
</td>
</tr>
</table>



## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_deserializer.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_proto(
    proto: <a href="../../cirq/google/api/v2/program_pb2/Operation.md"><code>cirq.google.api.v2.program_pb2.Operation</code></a>,
    *,
    arg_function_language: str = ''
) -> "cirq.Operation"
</code></pre>

Turns a cirq.google.api.v2.Operation proto into a GateOperation.




