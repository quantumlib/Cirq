<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.GreedySequenceSearchStrategy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="place_line"/>
</div>

# cirq.google.GreedySequenceSearchStrategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/greedy.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Greedy search method for linear sequence of qubits on a chip.

Inherits From: [`LinePlacementStrategy`](../../cirq/google/LinePlacementStrategy.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.line.GreedySequenceSearchStrategy`, `cirq.google.line.placement.GreedySequenceSearchStrategy`, `cirq.google.line.placement.greedy.GreedySequenceSearchStrategy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.GreedySequenceSearchStrategy(
    algorithm: str = 'best'
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
    

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`algorithm`
</td>
<td>
Greedy algorithm to be used. Available options are:
best - runs all heuristics and chooses the best result,
largest_area - on every step takes the qubit which has connection
with the largest number of unassigned qubits, and
minimal_connectivity - on every step takes the qubit with minimal
number of unassigned neighbouring qubits.
</td>
</tr>
</table>



## Methods

<h3 id="place_line"><code>place_line</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/greedy.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>place_line(
    device: "cirq.google.XmonDevice",
    length: int
) -> <a href="../../cirq/google/line/placement/GridQubitLineTuple.md"><code>cirq.google.line.placement.GridQubitLineTuple</code></a>
</code></pre>

Runs line sequence search.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`device`
</td>
<td>
Chip description.
</td>
</tr><tr>
<td>
`length`
</td>
<td>
Required line length.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Linear sequences found on the chip.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If search algorithm passed on initialization is not
recognized.
</td>
</tr>
</table>





