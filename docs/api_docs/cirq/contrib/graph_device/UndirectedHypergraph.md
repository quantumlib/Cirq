<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.graph_device.UndirectedHypergraph" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_edge"/>
<meta itemprop="property" content="add_edges"/>
<meta itemprop="property" content="add_vertex"/>
<meta itemprop="property" content="add_vertices"/>
<meta itemprop="property" content="random"/>
<meta itemprop="property" content="remove_vertex"/>
<meta itemprop="property" content="remove_vertices"/>
</div>

# cirq.contrib.graph_device.UndirectedHypergraph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>





<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.graph_device.hypergraph.UndirectedHypergraph`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.graph_device.UndirectedHypergraph(
    *,
    vertices: Optional[Iterable[Hashable]] = None,
    labelled_edges: Optional[Dict[Iterable[Hashable], Any]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`vertices`
</td>
<td>
The vertices.
</td>
</tr><tr>
<td>
`labelled_edges`
</td>
<td>
The labelled edges, as a mapping from (frozen) sets
of vertices to the corresponding labels. Vertices are
automatically added.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`edges`
</td>
<td>

</td>
</tr><tr>
<td>
`labelled_edges`
</td>
<td>

</td>
</tr><tr>
<td>
`vertices`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="add_edge"><code>add_edge</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edge(
    vertices: Iterable[Hashable],
    label: Any = None
) -> None
</code></pre>




<h3 id="add_edges"><code>add_edges</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edges(
    edges: Dict[Iterable[Hashable], Any]
)
</code></pre>




<h3 id="add_vertex"><code>add_vertex</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_vertex(
    vertex: Hashable
) -> None
</code></pre>




<h3 id="add_vertices"><code>add_vertices</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_vertices(
    vertices: Iterable[Hashable]
) -> None
</code></pre>




<h3 id="random"><code>random</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>random(
    vertices: Union[int, Iterable],
    edge_probs: Mapping[int, float]
) -> "UndirectedHypergraph"
</code></pre>

A random hypergraph.

Every possible edge is included with probability edge_prob[len(edge)].
All edges are labelled with None.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`vertices`
</td>
<td>
The vertex set. If an integer i, the vertex set is
{0, ..., i - 1}.
</td>
</tr><tr>
<td>
`edge_probs`
</td>
<td>
The probabilities of edges of given sizes. Non-positive
values mean the edge is never included and values at least 1
mean that it is always included.
</td>
</tr>
</table>



<h3 id="remove_vertex"><code>remove_vertex</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_vertex(
    vertex: Hashable
) -> None
</code></pre>




<h3 id="remove_vertices"><code>remove_vertices</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_vertices(
    vertices
)
</code></pre>




<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/hypergraph.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




