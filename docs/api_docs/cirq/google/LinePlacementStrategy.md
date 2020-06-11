<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.LinePlacementStrategy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="place_line"/>
</div>

# cirq.google.LinePlacementStrategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/place_strategy.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Choice and options for the line placement calculation method.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.line.LinePlacementStrategy`, `cirq.google.line.placement.LinePlacementStrategy`, `cirq.google.line.placement.place_strategy.LinePlacementStrategy`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Currently two methods are available: cirq.line.GreedySequenceSearchMethod
and cirq.line.AnnealSequenceSearchMethod.

## Methods

<h3 id="place_line"><code>place_line</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/place_strategy.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
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





