<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.results_to_proto" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.results_to_proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/results.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts trial results from multiple sweeps to v2 protobuf message.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.results.results_to_proto`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.results_to_proto(
    trial_sweeps: Iterable[Iterable[<a href="../../../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]],
    measurements: List[<a href="../../../../cirq/google/api/v2/MeasureInfo.md"><code>cirq.google.api.v2.MeasureInfo</code></a>],
    *,
    out: Optional[<a href="../../../../cirq/google/api/v2/result_pb2/Result.md"><code>cirq.google.api.v2.result_pb2.Result</code></a>] = None
) -> <a href="../../../../cirq/google/api/v2/result_pb2/Result.md"><code>cirq.google.api.v2.result_pb2.Result</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`trial_sweeps`
</td>
<td>
Iterable over sweeps and then over trial results within
each sweep.
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
List of info about measurements in the program.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
Optional message to populate. If not given, create a new message.
</td>
</tr>
</table>

