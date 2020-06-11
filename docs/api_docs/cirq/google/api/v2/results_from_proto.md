<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.results_from_proto" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.api.v2.results_from_proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/results.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a v2 result proto into List of list of trial results.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.results.results_from_proto`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.results_from_proto(
    msg: <a href="../../../../cirq/google/api/v2/result_pb2/Result.md"><code>cirq.google.api.v2.result_pb2.Result</code></a>,
    measurements: List[<a href="../../../../cirq/google/api/v2/MeasureInfo.md"><code>cirq.google.api.v2.MeasureInfo</code></a>] = None
) -> List[List[<a href="../../../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`msg`
</td>
<td>
v2 Result message to convert.
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
List of info about expected measurements in the program.
This may be used for custom ordering of the result. If no
measurement config is provided, then all results will be returned
in the order specified within the result.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list containing a list of trial results for each sweep.
</td>
</tr>

</table>

