<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line_on_device" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.line_on_device

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/line.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Searches for linear sequence of qubits on device.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.line.line_on_device`, `cirq.google.line.placement.line.line_on_device`, `cirq.google.line.placement.line_on_device`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line_on_device(
    device: "cirq.google.XmonDevice",
    length: int,
    method: <a href="../../cirq/google/LinePlacementStrategy.md"><code>cirq.google.LinePlacementStrategy</code></a> = greedy.GreedySequenceSearchStrategy()
) -> <a href="../../cirq/google/line/placement/GridQubitLineTuple.md"><code>cirq.google.line.placement.GridQubitLineTuple</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Google Xmon device instance.
</td>
</tr><tr>
<td>
`length`
</td>
<td>
Desired number of qubits making up the line.
</td>
</tr><tr>
<td>
`method`
</td>
<td>
Line placement method. Defaults to
cirq.greedy.GreedySequenceSearchMethod.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Line sequences search results.
</td>
</tr>

</table>

