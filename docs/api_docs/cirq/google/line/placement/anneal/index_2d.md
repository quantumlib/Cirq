<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line.placement.anneal.index_2d" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.line.placement.anneal.index_2d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/anneal.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Finds the first index of a target item within a list of lists.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line.placement.anneal.index_2d(
    seqs: List[List[Any]],
    target: Any
) -> Tuple[int, int]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`seqs`
</td>
<td>
The list of lists to search.
</td>
</tr><tr>
<td>
`target`
</td>
<td>
The item to find.
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
Item is not present.
</td>
</tr>
</table>

