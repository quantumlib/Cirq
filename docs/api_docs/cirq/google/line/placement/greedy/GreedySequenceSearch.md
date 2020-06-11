<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line.placement.greedy.GreedySequenceSearch" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_or_search"/>
</div>

# cirq.google.line.placement.greedy.GreedySequenceSearch

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/greedy.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for greedy search heuristics.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line.placement.greedy.GreedySequenceSearch(
    device: "cirq.google.XmonDevice",
    start: <a href="../../../../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Specialized greedy heuristics should implement abstrace _sequence_search
method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
Chip description.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
Starting qubit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When start qubit is not part of a chip.
</td>
</tr>
</table>



## Methods

<h3 id="get_or_search"><code>get_or_search</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/greedy.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_or_search() -> <a href="../../../../../cirq/google/line/placement/anneal/LineSequence.md"><code>cirq.google.line.placement.anneal.LineSequence</code></a>
</code></pre>

Starts the search or gives previously calculated sequence.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The linear qubit sequence found.
</td>
</tr>

</table>





