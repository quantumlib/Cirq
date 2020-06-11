<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.sweep_to_proto" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.sweep_to_proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/sweeps.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a Sweep to v2 protobuf message.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.sweeps.sweep_to_proto`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.sweep_to_proto(
    sweep: <a href="../../../../cirq/study/Sweep.md"><code>cirq.study.Sweep</code></a>,
    *,
    out: Optional[<a href="../../../../cirq/google/api/v2/run_context_pb2/Sweep.md"><code>cirq.google.api.v2.run_context_pb2.Sweep</code></a>] = None
) -> <a href="../../../../cirq/google/api/v2/run_context_pb2/Sweep.md"><code>cirq.google.api.v2.run_context_pb2.Sweep</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sweep`
</td>
<td>
The sweep to convert.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
Optional message to be populated. If not given, a new message will
be created.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Populated sweep protobuf message.
</td>
</tr>

</table>

