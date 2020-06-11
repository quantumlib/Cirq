<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line.placement.chip.chip_as_adjacency_list" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.line.placement.chip.chip_as_adjacency_list

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/chip.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gives adjacency list representation of a chip.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line.placement.chip.chip_as_adjacency_list(
    device: "cirq.google.XmonDevice"
) -> Dict[<a href="../../../../../cirq/google/line/placement/anneal/LineSequence.md"><code>cirq.google.line.placement.anneal.LineSequence</code></a>, List[<a href="../../../../../cirq/google/line/placement/anneal/LineSequence.md"><code>cirq.google.line.placement.anneal.LineSequence</code></a>]]
</code></pre>



<!-- Placeholder for "Used in" -->

The adjacency list is constructed in order of above, left_of, below and
right_of consecutively.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Chip to be converted.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Map from nodes to list of qubits which represent all the neighbours of
given qubit.
</td>
</tr>

</table>

