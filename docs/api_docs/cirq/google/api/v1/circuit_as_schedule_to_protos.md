<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v1.circuit_as_schedule_to_protos" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v1.circuit_as_schedule_to_protos

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v1/programs.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Convert a circuit into an iterable of protos.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v1.programs.circuit_as_schedule_to_protos`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v1.circuit_as_schedule_to_protos(
    circuit: "cirq.Circuit"
) -> Iterator[<a href="../../../../cirq/google/api/v1/operations_pb2/Operation.md"><code>cirq.google.api.v1.operations_pb2.Operation</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to convert to a proto. Must contain only
gates that can be cast to xmon gates.
</td>
</tr>
</table>



#### Yields:

An Operation proto.
