<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line.placement.anneal.AnnealSequenceSearch" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="search"/>
</div>

# cirq.google.line.placement.anneal.AnnealSequenceSearch

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/anneal.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Simulated annealing search heuristic.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line.placement.anneal.AnnealSequenceSearch(
    device: "cirq.google.XmonDevice",
    seed=None
) -> None
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
Chip description.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Optional seed value for random number generator.
</td>
</tr>
</table>



## Methods

<h3 id="search"><code>search</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/anneal.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>search(
    trace_func: Callable[[List[<a href="../../../../../cirq/google/line/placement/anneal/LineSequence.md"><code>cirq.google.line.placement.anneal.LineSequence</code></a>], float, float, float, bool], None] = None
) -> List[<a href="../../../../../cirq/google/line/placement/anneal/LineSequence.md"><code>cirq.google.line.placement.anneal.LineSequence</code></a>]
</code></pre>

Issues new linear sequence search.

Each call to this method starts new search.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`trace_func`
</td>
<td>
Optional callable which will be called for each simulated
annealing step with arguments: solution candidate (list of linear
sequences on the chip), current temperature (float), candidate cost
(float), probability of accepting candidate (float), and acceptance
decision (boolean).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of linear sequences on the chip found by this method.
</td>
</tr>

</table>





