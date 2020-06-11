<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.devices.known_devices.create_device_proto_for_qubits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.devices.known_devices.create_device_proto_for_qubits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/devices/known_devices.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create device spec for the given qubits and coupled pairs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.devices.known_devices.create_device_proto_for_qubits(
    qubits: Collection['cirq.Qid'],
    pairs: Collection[Tuple['cirq.Qid', 'cirq.Qid']],
    gate_sets: Optional[Iterable[<a href="../../../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a>]] = None,
    durations_picos: Optional[Dict[str, int]] = None,
    out: Optional[<a href="../../../../cirq/google/api/v2/device_pb2/DeviceSpecification.md"><code>cirq.google.api.v2.device_pb2.DeviceSpecification</code></a>] = None
) -> <a href="../../../../cirq/google/api/v2/device_pb2/DeviceSpecification.md"><code>cirq.google.api.v2.device_pb2.DeviceSpecification</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
Qubits that can perform single-qubit gates.
</td>
</tr><tr>
<td>
`pairs`
</td>
<td>
Pairs of coupled qubits that can perform two-qubit gates.
</td>
</tr><tr>
<td>
`gate_sets`
</td>
<td>
Gate sets that define the translation between gate ids and
cirq Gate objects.
</td>
</tr><tr>
<td>
`durations_picos`
</td>
<td>
A map from gate ids to gate durations in picoseconds.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
If given, populate this proto, otherwise create a new proto.
</td>
</tr>
</table>

